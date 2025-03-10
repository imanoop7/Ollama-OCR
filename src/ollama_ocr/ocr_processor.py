import json
from typing import Dict, Any, List, Union
import os
import base64
import requests
from contextlib import suppress
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import cv2
import pymupdf  # Import the pymupdf library
import numpy as np

class OCRProcessor:
    def __init__(self, model_name: str = "llama3.2-vision:11b", 
                 base_url: str = "http://localhost:11434/api/generate",
                 max_workers: int = 1):
        
        self.model_name = model_name
        self.base_url = base_url
        self.max_workers = max_workers

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert each page of a PDF to an image using pymupdf.
        Saves each page as a temporary image.
        Returns a list of image paths.
        """
        try:
            doc = pymupdf.open(pdf_path)
            image_paths = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap()  # Render page to an image
                temp_path = f"{pdf_path}_page{page_num}.png"  # Define output image path
                pix.save(temp_path)  # Save the image
                image_paths.append(temp_path)
            doc.close()
            return image_paths
        except Exception as e:
            raise ValueError(f"Could not convert PDF to images: {e}")

    def _preprocess_image(self, image_path: str, language: str = "en") -> str:
        """
        Preprocess image before OCR:
        - Convert PDF to image if needed (using pymupdf)
        - Language-specific preprocessing (if applicable)
        - Enhance contrast
        - Reduce noise
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Language-specific thresholding
        if language.lower() in ["japanese", "chinese", "zh", "korean"]:
            # For some CJK and similar languages adaptive thresholding may work better
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2)
            thresh = cv2.bitwise_not(thresh)
        else:
            # Default: Otsu thresholding
            thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            thresh = cv2.bitwise_not(thresh)

        # Save preprocessed image
        preprocessed_path = f"{image_path}_preprocessed.jpg"
        cv2.imwrite(preprocessed_path, thresh)

        return preprocessed_path

    def process_image(self, image_path: str, format_type: str = "markdown", preprocess: bool = True, 
                      custom_prompt: str = None, language: str = "en") -> str:
        """
        Process an image (or PDF) and extract text in the specified format

        Args:
            image_path: Path to the image file or PDF file
            format_type: One of ["markdown", "text", "json", "structured", "key_value","custom"]
            preprocess: Whether to apply image preprocessing
            custom_prompt: If provided, this prompt overrides the default based on format_type
            language: Language code to apply language specific OCR preprocessing
        """
        try:
            # If the input is a PDF, process all pages
            if image_path.lower().endswith('.pdf'):
                image_pages = self._pdf_to_images(image_path)
                print("No. of pages in the PDF", len(image_pages))
                responses = []
                for idx, page_file in enumerate(image_pages):
                    # Process each page with preprocessing if enabled
                    if preprocess:
                        preprocessed_path = self._preprocess_image(page_file, language)
                    else:
                        preprocessed_path = page_file

                    image_base64 = self._encode_image(preprocessed_path)

                    if custom_prompt and custom_prompt.strip():
                        prompt = custom_prompt
                        print("Using custom prompt:", prompt)  # Debug print
                    else:
                        prompts = {
                            "markdown": f"""Extract all text content from this image in {language} **exactly as it appears**, without modification, summarization, or omission.
                                Format the output in markdown:
                                - Use headers (#, ##, ###) **only if they appear in the image**
                                - Preserve original lists (-, *, numbered lists) as they are
                                - Maintain all text formatting (bold, italics, underlines) exactly as seen
                                - **Do not add, interpret, or restructure any content**
                            """,
                            "text": f"""Extract all visible text from this image in {language} **without any changes**.
                                - **Do not summarize, paraphrase, or infer missing text.**
                                - Retain all spacing, punctuation, and formatting exactly as in the image.
                                - If text is unclear or partially visible, extract as much as possible without guessing.
                                - **Include all text, even if it seems irrelevant or repeated.** 
                                """,


                           "json": f"""Extract all text from this image in {language} and format it as JSON, **strictly preserving** the structure.
                                - **Do not summarize, add, or modify any text.**
                                - Maintain hierarchical sections and subsections as they appear.
                                - Use keys that reflect the document's actual structure (e.g., "title", "body", "footer").
                                - Include all text, even if fragmented, blurry, or unclear.
                                """,


                            "structured": f"""Extract all text from this image in {language}, **ensuring complete structural accuracy**:
                                - Identify and format tables **without altering content**.
                                - Preserve list structures (bulleted, numbered) **exactly as shown**.
                                - Maintain all section headings, indents, and alignments.
                                - **Do not add, infer, or restructure the content in any way.**
                                """,


                           "key_value": f"""Extract all key-value pairs from this image in {language} **exactly as they appear**:
                                - Identify and extract labels and their corresponding values without modification.
                                - Maintain the exact wording, punctuation, and order.
                                - Format each pair as 'key: value' **only if clearly structured that way in the image**.
                                - **Do not infer missing values or add any extra text.**
                                """,

                            "table": f"""Extract all tabular data from this image in {language} **exactly as it appears**, without modification, summarization, or omission.
                                - **Preserve the table structure** (rows, columns, headers) as closely as possible.
                                - **Do not add missing values or infer content**—if a cell is empty, leave it empty.
                                - Maintain all numerical, textual, and special character formatting.
                                - If the table contains merged cells, indicate them clearly without altering their meaning.
                                - Output the table in a structured format such as Markdown, CSV, or JSON, based on the intended use.
                                """,


                        }
                        prompt = prompts.get(format_type, prompts["text"])
                        print("Using default prompt:", prompt)  # Debug print

                    # Prepare the request payload
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "images": [image_base64]
                    }

                    # Make the API call to Ollama
                    response = requests.post(self.base_url, json=payload)
                    response.raise_for_status()
                    res = response.json().get("response", "")
                    print("Page No. Processed", idx)
                    # Prefix result with page number
                    responses.append(f"Page {idx + 1}:\n{res}")

                    # Clean up temporary files
                    if preprocess and preprocessed_path.endswith('_preprocessed.jpg'):
                        os.remove(preprocessed_path)
                    if page_file.endswith('.png'):
                        os.remove(page_file)

                final_result = "\n".join(responses)
                if format_type == "json":
                    try:
                        json_data = json.loads(final_result)
                        return json.dumps(json_data, indent=2)
                    except json.JSONDecodeError:
                        return final_result
                return final_result

            # Process non-PDF images as before.
            if preprocess:
                image_path = self._preprocess_image(image_path, language)

            image_base64 = self._encode_image(image_path)

            # Clean up temporary files
            if image_path.endswith(('_preprocessed.jpg', '_temp.jpg')):
                os.remove(image_path)

            if custom_prompt and custom_prompt.strip():
                prompt = custom_prompt
                print("Using custom prompt:", prompt)
            else:
                prompts = {
                            "markdown": f"""Extract all text content from this image in {language} **exactly as it appears**, without modification, summarization, or omission.
                                Format the output in markdown:
                                - Use headers (#, ##, ###) **only if they appear in the image**
                                - Preserve original lists (-, *, numbered lists) as they are
                                - Maintain all text formatting (bold, italics, underlines) exactly as seen
                                - **Do not add, interpret, or restructure any content**
                            """,
                            "text": f"""Extract all visible text from this image in {language} **without any changes**.
                                - **Do not summarize, paraphrase, or infer missing text.**
                                - Retain all spacing, punctuation, and formatting exactly as in the image.
                                - If text is unclear or partially visible, extract as much as possible without guessing.
                                - **Include all text, even if it seems irrelevant or repeated.** 
                                """,


                           "json": f"""Extract all text from this image in {language} and format it as JSON, **strictly preserving** the structure.
                                - **Do not summarize, add, or modify any text.**
                                - Maintain hierarchical sections and subsections as they appear.
                                - Use keys that reflect the document's actual structure (e.g., "title", "body", "footer").
                                - Include all text, even if fragmented, blurry, or unclear.
                                """,


                            "structured": f"""Extract all text from this image in {language}, **ensuring complete structural accuracy**:
                                - Identify and format tables **without altering content**.
                                - Preserve list structures (bulleted, numbered) **exactly as shown**.
                                - Maintain all section headings, indents, and alignments.
                                - **Do not add, infer, or restructure the content in any way.**
                                """,


                           "key_value": f"""Extract all key-value pairs from this image in {language} **exactly as they appear**:
                                - Identify and extract labels and their corresponding values without modification.
                                - Maintain the exact wording, punctuation, and order.
                                - Format each pair as 'key: value' **only if clearly structured that way in the image**.
                                - **Do not infer missing values or add any extra text.**
                                """,

                            "table": f"""Extract all tabular data from this image in {language} **exactly as it appears**, without modification, summarization, or omission.
                                - **Preserve the table structure** (rows, columns, headers) as closely as possible.
                                - **Do not add missing values or infer content**—if a cell is empty, leave it empty.
                                - Maintain all numerical, textual, and special character formatting.
                                - If the table contains merged cells, indicate them clearly without altering their meaning.
                                - Output the table in a structured format such as Markdown, CSV, or JSON, based on the intended use.
                                """,
                }
                prompt = prompts.get(format_type, prompts["text"])
                print("Using default prompt:", prompt)  # Debug print

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "images": [image_base64]
            }

            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()

            result = response.json().get("response", "")

            if format_type == "json":
                try:
                    json_data = json.loads(result)
                    return json.dumps(json_data, indent=2)
                except json.JSONDecodeError:
                    return result

            return result
        except Exception as e:
            msg = f"Error processing image: {str(e)}"
            with suppress(Exception):
                msg += f" [extra_info]: {response.json().get('error')}"
            return msg
            
    def process_batch(
        self,
        input_path: Union[str, List[str]],
        format_type: str = "markdown",
        recursive: bool = False,
        preprocess: bool = True,
        custom_prompt: str = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Process multiple images in batch
        
        Args:
            input_path: Path to directory or list of image paths
            format_type: Output format type
            recursive: Whether to search directories recursively
            preprocess: Whether to apply image preprocessing
            custom_prompt: If provided, this prompt overrides the default for each image
            language: Language code to apply language specific OCR preprocessing
            
        Returns:
            Dictionary with results and statistics
        """
        # Collect all image paths
        image_paths = []
        if isinstance(input_path, str):
            base_path = Path(input_path)
            if base_path.is_dir():
                pattern = '**/*' if recursive else '*'
                for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.tiff']:
                    image_paths.extend(base_path.glob(f'{pattern}{ext}'))
            else:
                image_paths = [base_path]
        else:
            image_paths = [Path(p) for p in input_path]

        results = {}
        errors = {}
        
        # Process images in parallel with progress bar
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_image, str(path), format_type, preprocess, custom_prompt, language): path
                    for path in image_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[str(path)] = future.result()
                    except Exception as e:
                        errors[str(path)] = str(e)
                    pbar.update(1)

        return {
            "results": results,
            "errors": errors,
            "statistics": {
                "total": len(image_paths),
                "successful": len(results),
                "failed": len(errors)
            }
        }
