import argparse
import json
import subprocess
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import os
import re
import random
import logging
from typing import List, Union

# Set up logging
logger = logging.getLogger('doc_extractor')
logger.setLevel(logging.INFO)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_url(url: str) -> str:
    """Extract text from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript', 'iframe', 'nav', 'footer']):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return ""

def ask_ollama(text: str, model: str = "phi3:latest") -> str:
    """Send text to Ollama for processing."""
    if not text.strip():
        return json.dumps({"error": "Empty input text"})
    
    prompt = f"""**STRICT INSTRUCTIONS**:
1. Output MUST be ONLY a valid JSON array
2. Format: [{{"id":1, "question":"...", "text":"...", "category":"..."}}]
3. No additional text, explanations, or markdown
4. Ensure proper JSON escaping for quotes

Example of ONLY acceptable output:
[{{"id":1, "question":"What is X?", "text":"X is...", "category":"general"}}]

Content to transform:
{text}"""

    try:
        # Add timeout and buffer size
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode('utf-8'),
            capture_output=True,
            text=False,
            check=True,
            timeout=300,
            bufsize=8192
        )
        raw_output = result.stdout.decode('utf-8', errors='ignore')
        
        # Log the first 500 chars of output for debugging
        logger.info(f"Raw Ollama output (first 500 chars):\n{raw_output[:500]}")
        
        return raw_output
        
    except subprocess.TimeoutExpired:
        logger.error("Ollama processing timeout after 300 seconds")
        return json.dumps({"error": "Processing timeout"})
    except Exception as e:
        logger.error(f"Ollama processing error: {str(e)}")
        return json.dumps({"error": f"Processing error: {str(e)}"})
    
def generate_question_variations(original_questions):
    """
    Generates 3-5 natural language variations for each question in the input list.
    Maintains the same answer and category while creating different phrasings.
    """
    action_question_frameworks = [
        "How can I {action}?",
        "What's the way to {action}?",
        "Where do I {action}?",
        "What's the process to {action}?",
        "Can you tell me how to {action}?"
    ]
    
    definition_phrases = [
        "What does {topic} mean?",
        "What is meant by {topic}?",
        "How would you define {topic}?",
        "Can you clarify what {topic} is?",
        "What's the definition of {topic}?"
    ]
    
    variations_list = []
    
    for question in original_questions:
        original_q = question["question"]
        original_a = question["text"]
        category = question["category"]
        variations = []
        
        # Extract key terms from question (simple approach)
        terms = [word.strip('?') for word in original_q.split() 
                if word.lower() not in ['what', 'how', 'can', 'could', 'would', 'is', 'are', 'does', 'do', 'the', 'a', 'an']]
        topic = ' '.join(terms[-3:])  # Take last few words as topic
        
        # Generate different types of variations based on question type
        if original_q.startswith('What is'):
            variations.extend([f.format(topic=topic) for f in definition_phrases])
        elif any(original_q.startswith(x) for x in ['How can', 'How do', 'How does']):
            action = ' '.join(terms)
            variations.extend([f.format(action=action) for f in action_question_frameworks])
        elif any(original_q.startswith(x) for x in ['Who is', 'Who are', 'Who was']):
            variations.extend([
                f"Can you tell me about {topic}?",
                f"What do you know about {topic}?",
                f"Describe {topic}",
                f"What's the role of {topic}?"
            ])
        elif original_q.startswith('When'):
            variations.extend([
                f"What time {topic}?",
                f"At what time {topic}?",
                f"Can you tell me when {topic}?",
                f"What's the timeline for {topic}?"
            ])
        else:
            # Generic variations for other question types
            variations.extend([
                f"Can you explain {topic}?",
                f"Tell me about {topic}",
                f"What do you know about {topic}?",
                f"I'd like to know about {topic}",
                f"Could you provide information about {topic}?"
            ])
        
        # Add the original question as one variation
        variations.append(original_q)
        
        # Remove duplicates
        variations = list(set(variations))
        
        # Select random variations (3-5)
        selected_variations = random.sample(
            variations, 
            min(random.randint(3,5), len(variations))
        )
        
        # Create new question objects for each variation
        for i, var_q in enumerate(selected_variations, 1):
            variations_list.append({
                "id": f"{question['id']}.{i}",  # Original ID with variation number
                "question": var_q,
                "text": original_a,  # Same answer
                "category": category   # Same category
            })
    
    return variations_list

def clean_llm_output(output: str) -> Union[dict, list]:
    """Clean and validate the LLM output with robust JSON extraction"""
    if not output:
        logger.warning("No output received from LLM")
        return {"error": "No output received"}
    
    # First try direct JSON parse
    try:
        data = json.loads(output)
        if isinstance(data, (list, dict)):
            return data
    except json.JSONDecodeError:
        pass
    
    # If direct parse fails, try extracting JSON
    logger.warning("Direct JSON parse failed, attempting extraction")
    
    # Try to find JSON array in the output
    json_match = re.search(r'(\[\s*\{.*?\}\s*\])', output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            logger.info("Successfully extracted JSON from output")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON extraction failed: {str(e)}")
    
    # Last resort: try line-by-line parsing
    logger.warning("Attempting line-by-line JSON recovery")
    cleaned_lines = []
    for line in output.splitlines():
        line = line.strip()
        if line and not line.startswith(('```', '<!--', '**')):
            cleaned_lines.append(line)
    
    try:
        data = json.loads(''.join(cleaned_lines))
        return data
    except json.JSONDecodeError:
        logger.error("All JSON parsing attempts failed")
        # Return the first 200 chars of output for debugging
        sample = output[:200].replace('\n', ' ')
        return {"error": "Invalid JSON format", "sample_output": sample}
    
def validate_qa_structure(data: Union[dict, list]) -> list:
    """Ensure all QA pairs have required fields"""
    if isinstance(data, dict):
        if "error" in data:
            return [data]
        # Convert single item to list
        data = [data]
    
    validated = []
    for item in data:
        if not isinstance(item, dict):
            continue
            
        # Ensure required fields exist
        if all(key in item for key in ["question", "text"]):
            # Set defaults for missing fields
            item.setdefault("id", len(validated) + 1)
            item.setdefault("category", "general")
            validated.append(item)
        elif "error" not in item:
            item["error"] = "Missing required fields"
            validated.append(item)
    
    return validated

# Add logging to the key functions being called from FastAPI
def process_source(source: str, is_pdf: bool, model: str) -> Union[dict, list]:
    """Process a single source with enhanced file handling"""
    source_id = str(uuid.uuid4())  # Generate unique ID for this processing run
    logger.info(f"Processing {'PDF' if is_pdf else 'URL'}: {source} (ID: {source_id})")
    
    try:
        # Extract text
        raw_text = extract_text_from_pdf(source) if is_pdf else extract_text_from_url(source)
        if not raw_text:
            logger.error(f"Failed to extract text from {source}")
            return {"error": "Failed to extract text", "source": source, "id": source_id}
        
        # Pre-process text
        clean_text = preprocess_text(raw_text)
        if len(clean_text) > 8000:
            logger.warning(f"Text too long ({len(clean_text)} chars), truncating")
            clean_text = clean_text[:8000]
        
        # Get LLM response
        llm_output = ask_ollama(clean_text, model)
        
        # Clean and validate
        parsed_data = clean_llm_output(llm_output)
        validated_data = validate_qa_structure(parsed_data)
        
        # Generate variations if we got valid QAs
        if validated_data and all("error" not in item for item in validated_data):
            final_data = generate_question_variations(validated_data)
            logger.info(f"Successfully processed {source} into {len(final_data)} QAs")
            return {
                "data": final_data,
                "source": source,
                "id": source_id,
                "status": "success"
            }
        
        return {
            "data": validated_data,
            "source": source,
            "id": source_id,
            "status": "completed_with_errors"
        }
        
    except Exception as e:
        logger.error(f"Error processing {source}: {str(e)}", exc_info=True)
        return {
            "error": f"Processing error: {str(e)}",
            "source": source,
            "id": source_id,
            "status": "failed"
        }
    
def preprocess_text(text: str) -> str:
    """Clean text before sending to LLM"""
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    # Normalize whitespace and clean special quotes
    text = ' '.join(text.replace('"', "'").replace('\n', ' ').split())
    return text

def parse_args():
    parser = argparse.ArgumentParser(description="Extract FAQ-style information from documents")
    parser.add_argument('--pdf', nargs='+', help='Paths to PDF files', default=[])
    parser.add_argument('--url', nargs='+', help='URLs to process', default=[])
    parser.add_argument('--model', default='llama3', help='Ollama model name')
    parser.add_argument('--output', default='output.json', help='Output JSON file')
    parser.add_argument('--separate-files', action='store_true', 
                       help='Save separate output file for each input source')
    return parser.parse_args()

def main():
    # Configure logging for command-line use
    if __name__ == "__main__":
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    args = parse_args()
    
    if not args.pdf and not args.url:
        logger.error("Please provide either --pdf or --url")
        return
    
    logger.info(f"Starting processing of {len(args.pdf)} PDF(s) and {len(args.url)} URL(s)")
    all_qa_pairs = []
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    for pdf_path in args.pdf:
        data = process_source(pdf_path, is_pdf=True, model=args.model)
        if args.separate_files:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_file = f"{base_name}_faqs.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(data if isinstance(data, list) else [data], f, indent=4, ensure_ascii=False)
            logger.info(f"Saved separate FAQs to {output_file}")
        if isinstance(data, list):
            all_qa_pairs.extend(data)
        elif isinstance(data, dict):
            all_qa_pairs.append(data)
    
    for url in args.url:
        data = process_source(url, is_pdf=False, model=args.model)
        
        # Generate output filename based on domain
        domain = re.sub(r'[^a-zA-Z0-9]', '_', url.split('//')[-1].split('/')[0])
        output_file = f"output/{domain}_faqs.json"
        
        # Write only the FAQ data to the output file
        if isinstance(data, dict) and "data" in data:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(data["data"], f, indent=4, ensure_ascii=False)
            logger.info(f"Saved FAQs to {output_file}")
        else:
            logger.error(f"Failed to generate valid FAQs for {url}")
        
        # Clean up any temporary files
        temp_files = [
            f for f in os.listdir()
            if f.startswith(data.get("id", "")) and not f.endswith('_faqs.json')
        ]
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
    
    # Ensure all questions end with question marks
    for item in all_qa_pairs:
        if isinstance(item, dict) and "question" in item:
            item["question"] = item["question"].rstrip('?') + '?'
    
    with open(args.output, "w", encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, indent=4, ensure_ascii=False)
    
    logger.info(f"All done! Saved {len(all_qa_pairs)} FAQ pairs to {args.output}")
    
    # Count valid QA pairs vs errors
    valid_entries = sum(1 for item in all_qa_pairs 
                       if isinstance(item, dict) and "question" in item and "text" in item)
    error_count = len(all_qa_pairs) - valid_entries
    
    logger.info(f"Summary: {valid_entries} valid Q&A pairs, {error_count} errors")
    
    if error_count > 0:
        logger.warning("Entries with errors:")
        for i, item in enumerate(all_qa_pairs):
            if isinstance(item, dict) and "error" in item:
                logger.warning(f"Item {i}: {item['error']}")

if __name__ == "__main__":
    main()