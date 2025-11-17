"""
Base class for content processors.
All content processors should inherit from this class.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
import logging


class BaseContentProcessor(ABC):
    """
    Abstract base class for content processors.
    
    Each processor handles a specific content type (audio, text, image, etc.)
    and extracts text content and metadata from files of that type.
    """
    
    def __init__(self):
        self.supported_extensions = self.get_supported_extensions()
    
    @abstractmethod
    def get_supported_extensions(self) -> list:
        """
        Return list of file extensions this processor supports.
        Example: ['.mp3', '.wav', '.m4a'] for audio
        """
        pass
    
    @abstractmethod
    def extract_content(self, file_path: Path) -> Dict:
        """
        Extract content from a file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary with standardized format:
            {
                'text': str,  # Main text content
                'metadata': dict,  # Type-specific metadata
                'attachments': list,  # Any embedded files (paths)
                'source_type': str,  # 'audio', 'text', 'image', etc.
                'language': Optional[str],  # Detected language if applicable
            }
        """
        pass
    
    def can_process(self, file_path: Path) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if this processor can handle the file
        """
        return file_path.suffix.lower() in self.supported_extensions
    
    def get_source_type(self) -> str:
        """
        Return the source type identifier for this processor.
        Used in the standardized output format.
        """
        return self.__class__.__name__.replace('Processor', '').lower()

