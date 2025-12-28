from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

class BaseContentProcessor(ABC):
    def __init__(self):
        self.supported_extensions = self.get_supported_extensions()
    
    @abstractmethod
    def get_supported_extensions(self) -> list:
        pass
    
    @abstractmethod
    def extract_content(self, file_path: Path) -> Dict:
        pass
    
    def can_process(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions
    
    def get_source_type(self) -> str:
        return self.__class__.__name__.replace('Processor', '').lower()
