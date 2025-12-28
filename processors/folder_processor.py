from pathlib import Path
from typing import Dict
import logging
from .base_processor import BaseContentProcessor
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor
from .image_processor import ImageProcessor

class FolderProcessor(BaseContentProcessor):
    def __init__(self, content_processor_manager=None):
        self.content_processor_manager = content_processor_manager
        self.audio_processor = AudioProcessor()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        super().__init__()
    
    def get_supported_extensions(self) -> list:
        return []
    
    def can_process(self, file_path: Path) -> bool:
        return file_path.is_dir()
    
    def extract_content(self, folder_path: Path) -> Dict:
        # Simplified implementation for MVP
        files = sorted([f for f in folder_path.iterdir() if f.is_file()])
        combined_text = []
        attachments = []
        
        for file_path in files:
            if file_path.name.startswith('.'): continue
            
            processor = None
            if self.content_processor_manager:
                processor = self.content_processor_manager.get_processor(file_path)
            
            if not processor:
                if file_path.suffix.lower() in self.audio_processor.get_supported_extensions(): processor = self.audio_processor
                elif file_path.suffix.lower() in self.text_processor.get_supported_extensions(): processor = self.text_processor
                elif file_path.suffix.lower() in self.image_processor.get_supported_extensions(): processor = self.image_processor
            
            if processor:
                content = processor.extract_content(file_path)
                text = content.get('text', '').strip()
                if text:
                    combined_text.append(f"\n\n## File: {file_path.name}\n{text}")
                attachments.extend(content.get('attachments', []))
            else:
                 attachments.append(str(file_path))
                 
        return {
            'text': "\n".join(combined_text),
            'metadata': {'folder_path': str(folder_path), 'file_count': len(files)},
            'attachments': attachments,
            'source_type': 'folder'
        }
