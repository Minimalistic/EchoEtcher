"""
Text content processor.
Reads text directly from text files.
"""
from pathlib import Path
from typing import Dict
import logging

from .base_processor import BaseContentProcessor


class TextProcessor(BaseContentProcessor):
    """Processor for text files (.txt, .md, etc.)."""
    
    def get_supported_extensions(self) -> list:
        """Return supported text file extensions."""
        return ['.txt', '.md', '.markdown', '.text']
    
    def extract_content(self, file_path: Path) -> Dict:
        """
        Extract text content from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with text content and metadata
        """
        logging.info(f"Reading text file: {file_path}")
        
        try:
            # Try UTF-8 first
            text = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1 (handles most cases)
            logging.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
            try:
                text = file_path.read_text(encoding='latin-1')
            except UnicodeDecodeError:
                # Last resort: ignore errors
                logging.warning(f"latin-1 decode failed for {file_path}, using error handling")
                text = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Extract metadata
        metadata = {
            'file_size': file_path.stat().st_size,
            'line_count': len(text.splitlines()),
            'word_count': len(text.split()),
            'original_file': str(file_path),
        }
        
        # If it's markdown, note that
        if file_path.suffix.lower() in ['.md', '.markdown']:
            metadata['format'] = 'markdown'
        
        return {
            'text': text,
            'metadata': metadata,
            'attachments': [],  # Text files don't have attachments
            'source_type': 'text',
            'language': None,  # Could add language detection later
        }

