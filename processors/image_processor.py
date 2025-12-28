from pathlib import Path
from typing import Dict
import logging
from .base_processor import BaseContentProcessor

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class ImageProcessor(BaseContentProcessor):
    def get_supported_extensions(self) -> list:
        return ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif']
    
    def extract_content(self, file_path: Path) -> Dict:
        text = ""
        metadata = {'image_file': str(file_path), 'ocr_available': PIL_AVAILABLE and TESSERACT_AVAILABLE}
        
        if PIL_AVAILABLE:
            try:
                with Image.open(file_path) as img:
                    metadata['width'] = img.width
                    metadata['height'] = img.height
                    metadata['format'] = img.format
                    
                    if TESSERACT_AVAILABLE:
                        try:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            text = pytesseract.image_to_string(img).strip()
                            metadata['ocr_success'] = True
                        except Exception as e:
                            metadata['ocr_success'] = False
                            metadata['ocr_error'] = str(e)
            except Exception as e:
                metadata['error'] = str(e)
        else:
             text = "(PIL not installed, cannot process image)"

        return {
            'text': text,
            'metadata': metadata,
            'attachments': [str(file_path)],
            'source_type': 'image',
            'language': None
        }
