"""
Folder content processor.
Processes entire folders as collective notes, combining content from multiple files.
"""
from pathlib import Path
from typing import Dict, List
import logging
import time

from .base_processor import BaseContentProcessor
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor
from .image_processor import ImageProcessor


class FolderProcessor(BaseContentProcessor):
    """Processor for folders containing multiple files."""
    
    def __init__(self, content_processor_manager=None):
        """
        Initialize folder processor.
        
        Args:
            content_processor_manager: Reference to ContentProcessorManager for getting individual processors
        """
        self.content_processor_manager = content_processor_manager
        self.audio_processor = AudioProcessor()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        super().__init__()
    
    def get_supported_extensions(self) -> list:
        """Folders don't have extensions, but we'll handle them specially."""
        return []  # Folders are handled by path checking, not extension
    
    def can_process(self, file_path: Path) -> bool:
        """Check if this is a directory."""
        return file_path.is_dir()
    
    def extract_content(self, folder_path: Path) -> Dict:
        """
        Extract content from all files in a folder.
        
        Args:
            folder_path: Path to the folder
            
        Returns:
            Dictionary with combined content from all files
        """
        logging.info(f"Processing folder: {folder_path}")
        
        all_text_parts = []
        all_attachments = []
        metadata_parts = []
        detected_languages = []
        
        # Get all files in the folder (non-recursive)
        files = sorted([f for f in folder_path.iterdir() if f.is_file()])
        
        if not files:
            logging.warning(f"Folder {folder_path} is empty")
            return {
                'text': "Empty folder",
                'metadata': {'folder_path': str(folder_path), 'file_count': 0},
                'attachments': [],
                'source_type': 'folder',
                'language': None,
            }
        
        logging.info(f"Found {len(files)} file(s) in folder")
        
        # Process each file
        for file_path in files:
            try:
                # Skip hidden files and system files
                if file_path.name.startswith('.'):
                    logging.debug(f"Skipping hidden file: {file_path.name}")
                    continue
                
                # Skip iCloud placeholder files
                if file_path.name.endswith('.icloud'):
                    logging.debug(f"Skipping iCloud placeholder: {file_path.name}")
                    continue
                
                # Get the appropriate processor
                processor = None
                if self.content_processor_manager:
                    processor = self.content_processor_manager.get_processor(file_path)
                
                if not processor:
                    # Try direct processor matching
                    if file_path.suffix.lower() in self.audio_processor.get_supported_extensions():
                        processor = self.audio_processor
                    elif file_path.suffix.lower() in self.text_processor.get_supported_extensions():
                        processor = self.text_processor
                    elif file_path.suffix.lower() in self.image_processor.get_supported_extensions():
                        processor = self.image_processor
                
                if processor:
                    file_type = processor.get_source_type()
                    logging.info(f"Processing {file_type} file: {file_path.name}")
                    
                    # Extract content from this file
                    extracted = processor.extract_content(file_path)
                    
                    # Combine text with separator
                    text = extracted.get('text', '').strip()
                    # Skip placeholder text that will be handled by vision API
                    if text and "please describe" not in text.lower() and not (text.startswith("Image file") and "no text" in text.lower()):
                        # Add file header
                        all_text_parts.append(f"\n\n## File: {file_path.name}\n")
                        all_text_parts.append(text)
                    elif file_type == "image":
                        # For images with no OCR text, add a note that vision will analyze it
                        all_text_parts.append(f"\n\n## File: {file_path.name}\n")
                        all_text_parts.append("(Image - will be analyzed with vision AI)")
                    
                    # Collect attachments
                    attachments = extracted.get('attachments', [])
                    if attachments:
                        all_attachments.extend(attachments)
                    
                    # Collect metadata
                    file_metadata = extracted.get('metadata', {})
                    if file_metadata:
                        metadata_parts.append({
                            'filename': file_path.name,
                            'type': file_type,
                            'metadata': file_metadata
                        })
                    
                    # Collect language if detected
                    lang = extracted.get('language')
                    if lang and lang not in detected_languages:
                        detected_languages.append(lang)
                    
                else:
                    logging.warning(f"No processor available for file: {file_path.name}")
                    # Still add as attachment even if we can't process it
                    all_attachments.append(str(file_path))
                    
            except Exception as e:
                logging.error(f"Error processing file {file_path.name} in folder: {e}")
                logging.exception("Full error trace:")
                # Continue processing other files
        
        # Combine all text parts
        combined_text = "\n".join(all_text_parts).strip()
        if not combined_text:
            combined_text = f"Folder contains {len(files)} file(s) but no extractable text content."
        
        # Determine primary language
        primary_language = detected_languages[0] if detected_languages else None
        
        return {
            'text': combined_text,
            'metadata': {
                'folder_path': str(folder_path),
                'folder_name': folder_path.name,
                'file_count': len(files),
                'files': metadata_parts,
            },
            'attachments': all_attachments,
            'source_type': 'folder',
            'language': primary_language,
        }
    
    def get_source_type(self) -> str:
        """Return the source type identifier."""
        return 'folder'

