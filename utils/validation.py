from typing import Tuple, Dict, Optional
import re
from logger import log

def validate_response(
    response: str,
    context: str,
    brand_config: Optional[Dict] = None
) -> Tuple[bool, str]:
    """
    Generic response validator with brand-specific rules
    Args:
        response: Generated response text
        context: Retrieved context used for generation
        brand_config: Brand configuration dictionary
    Returns:
        Tuple of (is_valid, error_message)
    """
    # 1. Empty response check
    if not response.strip():
        return False, "Empty response"

    # 2. Apply brand-specific validation if config exists
    if brand_config:
        # Word limit check
        word_limit = brand_config.get("word_limit", 50)
        if len(response.split()) > word_limit:
            return False, f"Response exceeds {word_limit}-word limit"

        # Off-topic content check
        if should_use_offtopic_response(response, context, brand_config):
            return False, "Response triggered off-topic rules"

        # Brand-specific corrections
        if needs_correction(response, brand_config.get("corrections", {})):
            return False, "Response contains uncorrected terms"

    # 3. Generic quality checks (applies to all brands)
    if "[No relevant info]" in context and not is_appropriate_no_info_response(response):
        return False, "Missing disclaimer for missing information"

    return True, ""

def should_use_offtopic_response(response: str, context: str, brand_config: Dict) -> bool:
    """Determine if response should use the brand's off-topic message"""
    # Check if context explicitly indicates no info
    if "[No relevant info]" in context:
        return True

    # Check for brand-specific keywords if defined
    if "products" in brand_config:
        required_terms = "|".join(
            re.escape(product.lower()) 
            for product in brand_config.get("products")
        )
        if not re.search(required_terms, response.lower()):
            return True

    return False

def needs_correction(text: str, corrections: Dict[str, str]) -> bool:
    """Check if text contains terms needing correction"""
    for wrong_term in corrections:
        if wrong_term.lower() in text.lower():
            return True
    return False

def is_appropriate_no_info_response(response: str) -> bool:
    """Check if response properly handles missing info"""
    no_info_phrases = [
        "don't know",
        "don't have",
        "no information",
        "not sure"
    ]
    return any(phrase in response.lower() for phrase in no_info_phrases)