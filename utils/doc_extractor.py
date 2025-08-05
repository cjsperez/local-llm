import argparse
import json
import subprocess
import fitz  # PyMuPDF
import uuid
import requests
from bs4 import BeautifulSoup
import os
import re
import random
import logging
from typing import List, Union, Tuple
from pathlib import Path

# Set up logging
logger = logging.getLogger('doc_extractor')
logger.setLevel(logging.INFO)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file with error handling."""
    try:
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
        return ""

def extract_text_from_url(url: str) -> str:
    """Extract text from a webpage with robust error handling."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'noscript', 'iframe', 'nav', 'footer', 
                        'header', 'aside', 'form', 'button', 'input']):
            tag.decompose()
            
        # Get text with better paragraph handling
        text = '\n\n'.join(
            p.get_text().strip() 
            for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            if p.get_text().strip()
        )
        
        return text if text else soup.get_text(separator="\n", strip=True)
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return ""

def get_expected_faq_count(text_length: int) -> Tuple[int, int]:
    """Return minimum and maximum expected FAQs based on text length."""
    if text_length < 800:
        return (2, 4)
    elif text_length < 2000:
        return (4, 8)
    elif text_length < 5000:
        return (6, 12)
    else:
        return (8, 15)

def ask_ollama(text: str, model: str = "mistral:7b-instruct") -> str:
    """Send text to Ollama for processing with enhanced prompt engineering."""
    if not text.strip():
        return json.dumps({"error": "Empty input text"})
    
    min_faqs, max_faqs = get_expected_faq_count(len(text))
    
    prompt = f"""You are a strict FAQ extraction agent.

    **OBJECTIVE**:
    From the given text, extract ALL possible and **distinct** question-answer pairs in strict FAQ format.

    **RULES**:
    1. Parse the text carefully and extract every standalone **fact** as a separate Q&A.
    2. Do NOT combine unrelated facts into one entry. ONE FACT = ONE FAQ.
    3. Ensure 100% coverage: **every important detail** from the text must be included in the output.
    4. Each FAQ must focus on a **single detail only** (e.g., name, address, phone, pricing, services).
    5. No vague, broad, or redundant questions.
    6. Format the output as a valid **JSON array** only.
    7. Each object must have exactly these keys:
    - `"id"`: starts at 1, increments by 1
    - `"question"`: full question, ends with "?"
    - `"text"`: complete, standalone answer (minimum 2 sentences if possible)
    - `"category"`: one word category (e.g., general, contact, pricing, products, services)

    **STYLE GUIDE**:
    - Answers must give context so they stand alone.
    - Rephrase facts into natural questions. No bullet points.
    - Maintain clean JSON: no comments, trailing commas, or invalid formatting.
    - Use proper capitalization and punctuation.

    **EXAMPLE**:
    [
    {{
        "id": 1,
        "question": "What is the name of the company?",
        "text": "The company is called ParkNcharge, pronounced as 'Park and Charge.' It focuses on eMobility solutions in the Philippines.",
        "category": "general"
    }},
    {{
        "id": 2,
        "question": "What is the address of ParkNcharge?",
        "text": "ParkNcharge is located at the 7th Floor, V.A. Rufino Building, Ayala Avenue, Makati City, Philippines, Zip Code 1226.",
        "category": "contact"
    }}
    ]

    **INPUT TEXT**:
    {text[:10000]}
    """

  # Limit input size to prevent overload

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode('utf-8'),
            capture_output=True,
            text=False,
            check=True,
            timeout=300,
            bufsize=8192
        )
        raw_output = result.stdout.decode('utf-8', errors='replace')
        raw_output = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', raw_output)
        
        logger.debug(f"Raw Ollama output (first 500 chars):\n{raw_output[:500]}")
        return raw_output
        
    except subprocess.TimeoutExpired:
        logger.error("Ollama processing timeout after 300 seconds")
        return json.dumps({"error": "Processing timeout"})
    except Exception as e:
        logger.error(f"Ollama processing error: {str(e)}")
        return json.dumps({"error": f"Processing error: {str(e)}"})

def clean_llm_output(output: str) -> Union[dict, list]:
    """More robust cleaning with detailed error logging"""
    if not output:
        logger.error("Received empty output from LLM")
        return {"error": "Empty LLM output"}

    try:
        # First try direct JSON parse
        cleaned = output.strip().replace('```json', '').replace('```', '')
        data = json.loads(cleaned)
        
        if isinstance(data, list):
            if not data:
                logger.warning("Received empty list from LLM")
            return data
        elif isinstance(data, dict):
            return [data]  # Convert single object to list
        else:
            logger.error(f"Unexpected LLM output type: {type(data)}")
            return {"error": "Invalid output format"}
            
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed, trying recovery: {str(e)}")
        
        # Try to find JSON objects in the output
        json_objects = []
        for match in re.finditer(r'\{.*?\}', output, re.DOTALL):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict) and 'question' in obj and 'text' in obj:
                    json_objects.append(obj)
            except json.JSONDecodeError:
                continue
                
        if json_objects:
            logger.info(f"Recovered {len(json_objects)} valid FAQs from malformed output")
            return json_objects
            
        logger.error("Could not recover any valid JSON from LLM output")
        return {
            "error": "Invalid JSON format",
            "raw_output_sample": output[:500],
            "suggestion": "Check the Ollama prompt and model output format"
        }

def validate_qa_structure(data: Union[dict, list], text_length: int) -> dict:
    """More forgiving validation with detailed feedback"""
    if isinstance(data, dict) and "error" in data:
        return {
            "status": "failed",
            "error": data["error"],
            "details": data
        }

    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    validated = []
    errors = []
    
    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            errors.append(f"FAQ {idx}: Not a dictionary")
            continue
            
        # Check for required fields
        missing = [f for f in ['question', 'text'] if f not in item]
        if missing:
            errors.append(f"FAQ {idx}: Missing fields {missing}")
            continue
            
        # Normalize fields
        item.setdefault("id", str(uuid.uuid4()))
        item.setdefault("category", "general")
        item['question'] = item['question'].strip()
        if not item['question'].endswith('?'):
            item['question'] += '?'
            
        validated.append(item)
    
    min_expected, max_expected = get_expected_faq_count(text_length)
    faq_count = len(validated)
    
    return {
        "faqs": validated,
        "errors": errors,
        "expected_range": f"{min_expected}-{max_expected}",
        "status": "success" if faq_count >= min_expected else "partial",
        "is_acceptable": faq_count >= (min_expected // 2)  # Allow partial success
    }

def generate_question_variations(original_questions: List[dict]) -> List[dict]:
    """Generate natural language variations for questions with quality controls."""
    variations_list = []
    
    for q in original_questions:
        if "error" in q:
            continue
            
        original_q = q["question"]
        original_a = q["text"]
        category = q["category"]
        variations = []
        
        # Extract key terms
        terms = [
            word.strip('?.!').lower() 
            for word in original_q.split() 
            if word.lower() not in [
                'what', 'how', 'can', 'could', 'would', 
                'is', 'are', 'does', 'do', 'the', 'a', 'an'
            ]
        ]
        topic = ' '.join(terms[-3:])
        
        # Generate variations based on question type
        if original_q.startswith(('What is', 'What are')):
            variations.extend([
                f"Can you explain {topic}?",
                f"Describe {topic}",
                f"What does {topic} involve?",
                f"Tell me about {topic}"
            ])
        elif original_q.startswith(('How do', 'How can', 'How does')):
            action = ' '.join(terms)
            variations.extend([
                f"What's the process for {action}?",
                f"Steps to {action}",
                f"Guide me through {action}",
                f"Explain how to {action}"
            ])
        else:
            variations.extend([
                f"Tell me more about {topic}",
                f"Explain {topic} in detail",
                f"What should I know about {topic}?",
                f"Information about {topic}"
            ])
        
        # Add original question and ensure uniqueness
        variations.append(original_q)
        variations = list(set(variations))
        
        # Select 3-5 best variations
        selected = random.sample(variations, min(5, len(variations)))
        
        # Create variation entries
        for i, var_q in enumerate(selected, 1):
            variations_list.append({
                "id": f"{q['id']}.{i}",
                "question": var_q,
                "text": original_a,
                "category": category,
                "is_variation": True,
                "original_id": q['id']
            })
    
    return variations_list

def process_source(source: str, is_pdf: bool, model: str) -> dict:
    """Process a document source end-to-end with comprehensive error handling."""
    source_id = str(uuid.uuid4())
    logger.info(f"Processing {'PDF' if is_pdf else 'URL'}: {source}")
    
    try:
        # Text extraction
        raw_text = extract_text_from_pdf(source) if is_pdf else extract_text_from_url(source)
        if not raw_text:
            raise ValueError("Text extraction returned empty content")
            
        clean_text = preprocess_text(raw_text)
        if len(clean_text) > 10000:
            clean_text = clean_text[:10000]
            logger.warning(f"Truncated text to 10,000 characters")
        
        # FAQ extraction
        llm_output = ask_ollama(clean_text, model)
        parsed_data = clean_llm_output(llm_output)
        
        validation_result = validate_qa_structure(
            parsed_data, 
            len(clean_text)
        )
        
        # Generate variations if we have valid FAQs
        final_faqs = validation_result["faqs"]
        if validation_result["is_acceptable"]:
            valid_faqs = [q for q in final_faqs if "error" not in q]
            variations = generate_question_variations(valid_faqs)
            final_faqs.extend(variations)
        
        return {
            "status": "success" if validation_result["is_acceptable"] else "partial",
            "source": source,
            "faqs": final_faqs,
            "warnings": validation_result["warnings"],
            "text_length": len(clean_text),
            "expected_faqs": validation_result["expected_range"],
            "actual_faqs": len([q for q in final_faqs if "error" not in q])
        }
        
    except Exception as e:
        logger.error(f"Failed to process {source}: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "source": source
        }

def preprocess_text(text: str) -> str:
    """Clean and normalize text before processing."""
    # Remove unwanted characters and normalize
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('"', "'")  # Standardize quotes
    return text

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract FAQ-style information from documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--pdf', nargs='+', help='PDF file paths', default=[])
    parser.add_argument('--url', nargs='+', help='URLs to process', default=[])
    parser.add_argument('--model', default='mistral:7b-instruct', 
                       help='Ollama model to use')
    parser.add_argument('--output', default='output/faqs.json',
                       help='Output JSON file path')
    parser.add_argument('--min-text-length', type=int, default=300,
                       help='Minimum text length to process')
    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    all_results = []
    for pdf_path in args.pdf:
        result = process_source(pdf_path, True, args.model)
        all_results.append(result)
        
    for url in args.url:
        result = process_source(url, False, args.model)
        all_results.append(result)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    success = sum(1 for r in all_results if r['status'] == 'success')
    partial = sum(1 for r in all_results if r['status'] == 'partial')
    failed = sum(1 for r in all_results if r['status'] == 'failed')
    
    logger.info(
        f"Processing complete: {success} success, {partial} partial, {failed} failed\n"
        f"Results saved to {args.output}"
    )

if __name__ == "__main__":
    main()