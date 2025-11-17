"""
Content Processor Manager
Manages different content processors and routes files to the appropriate processor.
"""
from pathlib import Path
from typing import Dict, Optional
import logging

from .processors import AudioProcessor, TextProcessor, ImageProcessor, FolderProcessor, BaseContentProcessor


class ContentProcessorManager:
    """Manages content processors for different file types."""
    
    def __init__(self):
        self.processors: Dict[str, BaseContentProcessor] = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize all available content processors."""
        # Register processors
        audio_processor = AudioProcessor()
        text_processor = TextProcessor()
        image_processor = ImageProcessor()
        folder_processor = FolderProcessor(content_processor_manager=self)
        
        # Register by source type
        self.processors['audio'] = audio_processor
        self.processors['text'] = text_processor
        self.processors['image'] = image_processor
        self.processors['folder'] = folder_processor
        
        # Also register by extension for quick lookup
        self._extension_map = {}
        for processor in [audio_processor, text_processor, image_processor]:
            for ext in processor.get_supported_extensions():
                self._extension_map[ext.lower()] = processor
                logging.debug(f"Registered extension {ext} -> {processor.get_source_type()}")
        
        logging.info(f"Initialized {len(self.processors)} content processors")
        logging.info(f"Supported extensions: {', '.join(sorted(self._extension_map.keys()))}")
        logging.info("Folder processing enabled - folders will be processed as collective notes")
    
    def get_processor(self, file_path: Path) -> Optional[BaseContentProcessor]:
        """
        Get the appropriate processor for a file or folder.
        
        Args:
            file_path: Path to the file or folder
            
        Returns:
            Content processor that can handle the file/folder, or None if no processor found
        """
        # Check if it's a folder first
        if file_path.is_dir():
            return self.processors.get('folder')
        
        # Otherwise check by extension
        ext = file_path.suffix.lower()
        return self._extension_map.get(ext)
    
    def can_process(self, file_path: Path) -> bool:
        """
        Check if we can process a file or folder.
        
        Args:
            file_path: Path to the file or folder
            
        Returns:
            True if a processor exists for this file/folder type
        """
        # Folders are always processable
        if file_path.is_dir():
            return True
        
        # Check by extension for files
        return file_path.suffix.lower() in self._extension_map
    
    def get_supported_extensions(self) -> list:
        """Get list of all supported file extensions."""
        return list(self._extension_map.keys())
    
    def unload_models(self):
        """Unload models from processors that support it (e.g., audio processor)."""
        for processor in self.processors.values():
            if hasattr(processor, 'unload_model'):
                processor.unload_model()

