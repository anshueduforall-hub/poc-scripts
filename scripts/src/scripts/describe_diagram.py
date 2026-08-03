import os
import re
import base64
import json
import urllib.request
import urllib.error

# Configuration
OUTPUT_DIR = "./generated_107_notusellm"
MD_FILENAME = "local_parsed_document.md"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava:latest"

# Context extraction window (characters before and after image reference)
CONTEXT_WINDOW_SIZE = 500

PROMPT_TEMPLATE = (
    "Analyze this image in the context of the surrounding document text provided below.\n\n"
    "--- SURROUNDING DOCUMENT CONTEXT ---\n"
    "{context}\n"
    "--- END OF CONTEXT ---\n\n"
    "Task: Describe the figure, diagram, chart, or equation in detail. Explain its key components, "
    "data points, and relationship to the surrounding document context so a reader can understand it without seeing it."
)

def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and convert it to a clean base64 string without newlines."""
    with open(image_path, "rb") as img_file:
        raw_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        return raw_b64.replace("\n", "").replace("\r", "")

def query_ollama_vision(image_base64: str, prompt: str) -> str:
    """Send a request to local Ollama instance with sanitized image data."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }
    
    data = json.dumps(payload).encode("utf-8")
    
    req = urllib.request.Request(
        OLLAMA_URL, 
        data=data, 
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(data))
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"\n[Ollama Error {e.code}]: {error_body}")
        return ""
    except urllib.error.URLError as e:
        print(f"\n[Network Error]: {e.reason}")
        return ""

def extract_surrounding_context(full_content: str, start_idx: int, end_idx: int, window_size: int = CONTEXT_WINDOW_SIZE) -> str:
    """Extract text snippet around the image match index."""
    prefix_start = max(0, start_idx - window_size)
    suffix_end = min(len(full_content), end_idx + window_size)
    
    before_text = full_content[prefix_start:start_idx].strip()
    after_text = full_content[end_idx:suffix_end].strip()
    
    combined_context = f"{before_text}\n[IMAGE LOCATION]\n{after_text}"
    return combined_context

def process_markdown_and_describe_images(output_dir: str, md_filename: str):
    md_path = os.path.join(output_dir, md_filename)
    
    if not os.path.exists(md_path):
        print(f"Error: Markdown file not found at {md_path}")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match standard Markdown image tags: ![alt_text](image_path)
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    matches = list(image_pattern.finditer(content))

    if not matches:
        print("No image tags found in the Markdown file.")
        return

    print(f"Found {len(matches)} image(s) in {md_filename}. Processing with context via Ollama ({OLLAMA_MODEL})...")

    # Rebuild document content with context-aware descriptions
    new_content_parts = []
    last_end = 0

    for match in matches:
        start_idx, end_idx = match.span()
        alt_text = match.group(1)
        rel_img_path = match.group(2)
        
        # Append preceding unmodified text
        new_content_parts.append(content[last_end:start_idx])
        
        full_img_path = os.path.join(output_dir, rel_img_path)
        
        if not os.path.exists(full_img_path):
            print(f"Warning: Image file not found: {full_img_path}")
            new_content_parts.append(match.group(0))
            last_end = end_idx
            continue

        print(f"Processing: {rel_img_path} with surrounding context...")
        
        # Extract surrounding document text
        context_snippet = extract_surrounding_context(content, start_idx, end_idx)
        
        # Construct contextual prompt
        dynamic_prompt = PROMPT_TEMPLATE.format(context=context_snippet)
        
        img_b64 = encode_image_to_base64(full_img_path)
        description = query_ollama_vision(img_b64, dynamic_prompt)

        if description:
            formatted_output = (
                f"![{alt_text}]({rel_img_path})\n\n"
                f"> **Figure Description ({OLLAMA_MODEL}):**\n"
                f"> {description.replace(chr(10), chr(10) + '> ')}\n"
            )
            new_content_parts.append(formatted_output)
        else:
            new_content_parts.append(match.group(0))

        last_end = end_idx

    # Append remaining trailing text
    new_content_parts.append(content[last_end:])
    final_markdown = "".join(new_content_parts)

    # Save processed file
    processed_md_path = os.path.join(output_dir, "processed_notes_with_descriptions.md")
    with open(processed_md_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print(f"\nProcessing complete! Enhanced document saved to: {processed_md_path}")

if __name__ == "__main__":
    process_markdown_and_describe_images(OUTPUT_DIR, MD_FILENAME)