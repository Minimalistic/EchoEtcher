from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import logging
from .base_processor import BaseContentProcessor
from ..transcriber import WhisperTranscriber

try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3NoHeaderError
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

class AudioProcessor(BaseContentProcessor):
    def __init__(self):
        self.transcriber = None
        self._transcriber_loaded = False
        super().__init__()
    
    def get_supported_extensions(self) -> list:
        return ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    
    def _extract_recording_date(self, file_path: Path) -> Optional[datetime]:
        """
        Extract recording date from audio file metadata.
        Tries multiple metadata fields depending on file format.
        
        Returns:
            datetime object if recording date found, None otherwise
        """
        if not MUTAGEN_AVAILABLE:
            logging.debug("mutagen not available, skipping recording date extraction")
            return None
        
        try:
            audio_file = MutagenFile(str(file_path))
            if audio_file is None:
                return None
            
            recording_date = None
            
            # Try different metadata fields depending on format
            # MP3 (ID3 tags)
            if hasattr(audio_file, 'get'):
                # TDRC = Recording time (ID3v2.4)
                # TDRL = Release time (ID3v2.4)
                # TDTG = Tagging time (ID3v2.4)
                # For older ID3v2.3, might be TYER + TDAT
                for tag in ['TDRC', 'TDRL', 'TDTG']:
                    if tag in audio_file:
                        try:
                            date_str = str(audio_file[tag][0])
                            # Try parsing various date formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y']:
                                try:
                                    recording_date = datetime.strptime(date_str[:len(fmt)], fmt)
                                    logging.info(f"Found recording date from {tag}: {recording_date}")
                                    break
                                except ValueError:
                                    continue
                            if recording_date:
                                break
                        except (ValueError, IndexError, AttributeError):
                            continue
            
            # M4A/MP4 (QuickTime metadata)
            if not recording_date and hasattr(audio_file, 'tags'):
                # Try creation date or recording date
                for key in ['©day', '©rec', 'creation_date', 'recording_date']:
                    if key in audio_file.tags:
                        try:
                            date_str = str(audio_file.tags[key][0])
                            # M4A dates are often in ISO format
                            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y']:
                                try:
                                    recording_date = datetime.strptime(date_str[:len(fmt)], fmt)
                                    logging.info(f"Found recording date from {key}: {recording_date}")
                                    break
                                except ValueError:
                                    continue
                            if recording_date:
                                break
                        except (ValueError, IndexError, AttributeError):
                            continue
            
            # FLAC/Vorbis comments
            if not recording_date and hasattr(audio_file, 'tags'):
                for key in ['DATE', 'RECORDINGDATE', 'RECORDING_DATE']:
                    if key in audio_file.tags:
                        try:
                            date_str = str(audio_file.tags[key][0])
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y']:
                                try:
                                    recording_date = datetime.strptime(date_str[:len(fmt)], fmt)
                                    logging.info(f"Found recording date from {key}: {recording_date}")
                                    break
                                except ValueError:
                                    continue
                            if recording_date:
                                break
                        except (ValueError, IndexError, AttributeError):
                            continue
            
            return recording_date
            
        except Exception as e:
            logging.debug(f"Error extracting recording date from {file_path}: {e}")
            return None
    
    def _ensure_transcriber_loaded(self):
        if not self._transcriber_loaded:
            logging.info("Loading Whisper transcriber for audio processing...")
            self.transcriber = WhisperTranscriber()
            self._transcriber_loaded = True
    
    def extract_content(self, file_path: Path) -> Dict:
        self._ensure_transcriber_loaded()
        
        logging.info(f"Transcribing audio file: {file_path}")
        transcription_data = self.transcriber.transcribe(file_path)
        
        # Extract recording date from audio metadata
        recording_date = self._extract_recording_date(file_path)
        
        metadata = {
            'language': transcription_data.get('language'),
            'segments': transcription_data.get('segments', []),
            'audio_file': str(file_path),
        }
        
        # Add recording date to metadata if found
        if recording_date:
            metadata['recording_date'] = recording_date
        
        return {
            'text': transcription_data.get('text', ''),
            'metadata': metadata,
            'attachments': [str(file_path)],
            'source_type': 'audio',
            'language': transcription_data.get('language')
        }
    
    def unload_model(self):
        if self.transcriber and self._transcriber_loaded:
            self.transcriber.unload_model()
            self._transcriber_loaded = False
