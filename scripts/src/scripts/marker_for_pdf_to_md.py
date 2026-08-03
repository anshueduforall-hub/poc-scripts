import os
import time
#import pypdfium2  # Import early to avoid layout warning configurations

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output
from marker.config.parser import ConfigParser

def convert_pdf_with_local_ollama(pdf_path, output_dir):
    script_start_time = time.perf_counter()
    # 1. Update routing parameters to target Ollama instead of external APIs
    config_dict = {
        "use_llm": True,
        "llm_service": "marker.services.ollama.OllamaService",
        "output_format": "markdown",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "gemma4:e2b",
        
        # 1. Enable image extraction
        "extract_images": True,
    }

    config_dict1 = {
        "use_llm": False,
        "output_format": "markdown",
        
        # 1. Enable image extraction
        "extract_images": True,
    }
    
    print("Parsing configurations...")
    config_parser = ConfigParser(config_dict1)
    
    print("Loading local artifact extraction models (Surya/OCR)...")
    model_dict = create_model_dict()
    
    # 2. Wire the parameters inside PdfConverter
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )

    # Measure conversion time (PDF to Markdown generation)
    t1 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Starting PDF conversion and Markdown generation...")
    rendered = converter(pdf_path)

    conversion_time = time.perf_counter() - t1
    print(f"[{time.strftime('%H:%M:%S')}] Markdown generated in {conversion_time:.2f} seconds ({conversion_time / 60:.2f} minutes).")
    
    # Measure output saving time
    t2 = time.perf_counter()
    # Ensure destination layout path directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Commit layout to file system
    save_output(rendered, output_dir, "local_parsed_document")
    print(f"[{time.strftime('%H:%M:%S')}] Output saved to disk in {time.perf_counter() - t2:.2f} seconds.")
    
    total_time = time.perf_counter() - script_start_time
    print(f"[{time.strftime('%H:%M:%S')}] Total elapsed time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")

if __name__ == "__main__":
    pdf_file = "/home/kanshu/Downloads/keph107.pdf"
    output_path = "./generated_107_notusellm"
    
    convert_pdf_with_local_ollama(pdf_file, output_path)