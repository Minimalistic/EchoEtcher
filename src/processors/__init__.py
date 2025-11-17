"""
Content processors for different file types.
Each processor extracts content from a specific file type and returns standardized data.
"""

from .base_processor import BaseContentProcessor
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor
from .image_processor import ImageProcessor
from .folder_processor import FolderProcessor

__all__ = ['BaseContentProcessor', 'AudioProcessor', 'TextProcessor', 'ImageProcessor', 'FolderProcessor']

