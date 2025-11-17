"""
Audio content processor.
Wraps the WhisperTranscriber to extract text from audio files.
"""
from pathlib import Path
from typing import Dict
import logging
import os

from .base_processor import BaseContentProcessor
from ..transcriber import WhisperTranscriber


class AudioProcessor(BaseContentProcessor):
    """Processor for audio files using Whisper transcription."""
    
    def __init__(self):
        self.transcriber = None
        self._transcriber_loaded = False
        super().__init__()
    
    def get_supported_extensions(self) -> list:
        """Return supported audio file extensions."""
        return ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    
    def _ensure_transcriber_loaded(self):
        """Lazy load the transcriber to save memory."""
        if not self._transcriber_loaded:
            logging.info("Loading Whisper transcriber for audio processing...")
            self.transcriber = WhisperTranscriber()
            self._transcriber_loaded = True
    
    def extract_content(self, file_path: Path) -> Dict:
        """
        Extract text content from an audio file using Whisper.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with transcription data and metadata
        """
        self._ensure_transcriber_loaded()
        
        logging.info(f"Transcribing audio file: {file_path}")
        transcription_data = self.transcriber.transcribe(file_path)
        
        # Extract metadata
        metadata = {
            'language': transcription_data.get('language'),
            'segments': transcription_data.get('segments', []),
            'audio_file': str(file_path),
        }
        
        # Count low confidence segments
        low_confidence_segments = [
            s for s in transcription_data.get('segments', [])
            if s.get('confidence', 0) < -1.0 and s.get('no_speech_prob', 0) < 0.5
        ]
        if low_confidence_segments:
            metadata['low_confidence_segments'] = len(low_confidence_segments)
        
        return {
            'text': transcription_data.get('text', ''),
            'metadata': metadata,
            'attachments': [str(file_path)],  # Audio file is the attachment
            'source_type': 'audio',
            'language': transcription_data.get('language'),
            'raw_transcription_data': transcription_data,  # Keep for backward compatibility
        }
    
    def unload_model(self):
        """Unload the Whisper model to free memory."""
        if self.transcriber and self._transcriber_loaded:
            self.transcriber.unload_model()
            self._transcriber_loaded = False

