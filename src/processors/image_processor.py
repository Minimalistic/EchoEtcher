"""
Image content processor.
Extracts text from images using OCR and optionally uses vision models for description.
"""
from pathlib import Path
from typing import Dict, Optional
import logging

from .base_processor import BaseContentProcessor

# Try to import OCR libraries (optional dependencies)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL/Pillow not available. Image processing will be limited.")

# Try to import HEIF support for HEIC files
HEIF_AVAILABLE = False
if PIL_AVAILABLE:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        HEIF_AVAILABLE = True
        logging.info("HEIC/HEIF support enabled via pillow-heif")
    except ImportError:
        HEIF_AVAILABLE = False
        logging.debug("pillow-heif not available. HEIC/HEIF files will not be processable. Install with: pip install pillow-heif")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. OCR will not work. Install with: pip install pytesseract")


class ImageProcessor(BaseContentProcessor):
    """Processor for image files using OCR and optional vision models."""
    
    def get_supported_extensions(self) -> list:
        """Return supported image file extensions."""
        return ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif']
    
    def extract_content(self, file_path: Path) -> Dict:
        """
        Extract text content from an image file using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logging.info(f"Processing image file: {file_path}")
        
        text = ""
        metadata = {
            'image_file': str(file_path),
            'ocr_available': TESSERACT_AVAILABLE and PIL_AVAILABLE,
        }
        
        # Check if this is a HEIC file (requires special handling)
        is_heic = file_path.suffix.lower() in ['.heic', '.heif']
        
        # Get image dimensions and try OCR if PIL is available
        if PIL_AVAILABLE:
            try:
                # Check if HEIC support is needed and available
                if is_heic and not HEIF_AVAILABLE:
                    logging.warning(f"HEIC/HEIF file detected but pillow-heif not available: {file_path.name}")
                    metadata['format'] = 'HEIC/HEIF'
                    metadata['heic_unsupported'] = True
                    text = "HEIC/HEIF image file - processing not available. Install pillow-heif: pip install pillow-heif"
                    metadata['ocr_success'] = False
                    metadata['ocr_error'] = "HEIC format not supported - pillow-heif library required"
                else:
                    # Open and process the image (works for both regular images and HEIC if pillow-heif is installed)
                    with Image.open(file_path) as img:
                        metadata['width'] = img.width
                        metadata['height'] = img.height
                        metadata['format'] = img.format
                        metadata['mode'] = img.mode
                        
                        if is_heic:
                            metadata['heic_processed'] = True
                            logging.info(f"Successfully opened HEIC/HEIF file: {file_path.name}")
                        
                        # Try OCR if available
                        if TESSERACT_AVAILABLE:
                            try:
                                logging.info("Attempting OCR on image...")
                                # Convert to RGB if necessary (tesseract needs RGB)
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # Perform OCR
                                text = pytesseract.image_to_string(img)
                                text = text.strip()
                                
                                if text:
                                    logging.info(f"OCR extracted {len(text)} characters from image")
                                    metadata['ocr_success'] = True
                                    metadata['text_length'] = len(text)
                                else:
                                    logging.info("OCR found no text in image")
                                    metadata['ocr_success'] = False
                                    text = "No text found in image via OCR. This may be a photo without text, or the text may not be clearly visible."
                                    
                            except Exception as e:
                                logging.warning(f"OCR failed: {e}")
                                metadata['ocr_success'] = False
                                metadata['ocr_error'] = str(e)
                                # Don't set text to error message - let it be empty so Ollama can describe the image
                                text = ""
                        else:
                            # OCR not available
                            metadata['ocr_success'] = False
                            text = "Image file (OCR not available - install pytesseract for text extraction)"
                            
            except Exception as e:
                logging.warning(f"Could not read image: {e}")
                # If we can't read it, it's likely an unsupported format or corrupted
                if is_heic and not HEIF_AVAILABLE:
                    text = f"HEIC/HEIF file could not be processed: pillow-heif library required. Install with: pip install pillow-heif"
                    metadata['heic_unsupported'] = True
                else:
                    text = f"Image file could not be processed: {str(e)}. The file may be in an unsupported format or corrupted."
                metadata['ocr_success'] = False
                metadata['ocr_error'] = str(e)
        else:
            # PIL not available
            metadata['ocr_success'] = False
            if is_heic:
                text = "HEIC/HEIF image file (PIL/Pillow and pillow-heif required for processing)"
            else:
                text = "Image file (PIL/Pillow not available for processing)"
        
        # If we have no text content and OCR failed, leave text empty
        # The vision API will be used to analyze the image
        if not text or text.startswith("Image file") or "could not be processed" in text.lower() or "no text" in text.lower() or "please describe" in text.lower():
            # Clear the text - vision API will analyze the image
            text = ""
        
        return {
            'text': text,
            'metadata': metadata,
            'attachments': [str(file_path)],  # Image file is the attachment
            'source_type': 'image',
            'language': None,  # Could detect language from OCR text later
        }

