from pathlib import Path
from typing import Dict
import logging
from .base_processor import BaseContentProcessor

class TextProcessor(BaseContentProcessor):
    def get_supported_extensions(self) -> list:
        return ['.txt', '.md', '.markdown', '.text']
    
    def extract_content(self, file_path: Path) -> Dict:
        try:
            text = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                text = file_path.read_text(encoding='latin-1')
            except:
                text = file_path.read_text(encoding='utf-8', errors='ignore')
        
        metadata = {
            'file_size': file_path.stat().st_size,
            'line_count': len(text.splitlines()),
            'original_file': str(file_path),
        }
        
        return {
            'text': text,
            'metadata': metadata,
            'attachments': [],
            'source_type': 'text'
        }
