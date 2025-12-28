import whisper
from pathlib import Path
import torch
import warnings
import logging
import os
import json
import subprocess
import tempfile

# Filter out specific Whisper warnings about Triton/CUDA
warnings.filterwarnings('ignore', message='Failed to launch Triton kernels')

class WhisperTranscriber:
    def __init__(self, model_size=None, lazy_load=False):
        if model_size is None:
            model_size = os.getenv('WHISPER_MODEL_SIZE', 'medium')
        
        self.device = self._detect_best_device()
        self.model_size = model_size
        self.model = None
        
        # Chunking configuration
        self.chunk_threshold_seconds = float(os.getenv('WHISPER_CHUNK_THRESHOLD', '240'))
        self.chunk_duration_seconds = float(os.getenv('WHISPER_CHUNK_DURATION', '30'))
        self.chunk_overlap_seconds = float(os.getenv('WHISPER_CHUNK_OVERLAP', '5'))
        
        self.default_prompt = (
            "This is a personal note or journal entry. "
            "The speaker is sharing thoughts, ideas, or reflections "
            "in a natural, conversational style. "
            "Use proper capitalization, punctuation, and sentence structure. "
        )
        
        if not lazy_load:
            self._load_model()
    
    def _detect_best_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_model(self):
        if self.model is not None:
            return
        logging.info(f"Loading Whisper model: {self.model_size}")
        self.model = whisper.load_model(self.model_size).to(self.device)
        logging.info(f"Whisper model loaded on device: {self.device}")
    
    def unload_model(self):
        if self.model is None:
            return
        if self.device == "cuda":
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
        del self.model
        self.model = None
        import gc
        gc.collect()

    def transcribe(self, audio_path: Path):
        self.ensure_loaded()
        # Simple transcription for now to minimize complexity in porting
        # Assuming single file for MVP or short clips
        duration = self._get_audio_duration(audio_path)
        if duration > self.chunk_threshold_seconds and duration != float('inf'):
             logging.info(f"Audio file > {self.chunk_threshold_seconds}s, using simple transcription anyway for MVP stability.")
        
        return self._transcribe_single(audio_path)

    def ensure_loaded(self):
        if self.model is None:
            self._load_model()

    def _get_audio_duration(self, audio_path: Path) -> float:
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(audio_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
        except Exception as e:
            logging.warning(f"Could not determine audio duration: {e}")
            return float('inf')

    def _transcribe_single(self, audio_path: Path) -> dict:
        use_fp16 = self.device == "cuda"
        options = {
            "fp16": use_fp16,
            "beam_size": 5,
            "best_of": 3,
            "temperature": [0.0, 0.2],
            "task": "transcribe",
            "initial_prompt": self.default_prompt,
            "condition_on_previous_text": False,
            "word_timestamps": True,
        }
        
        result = self.model.transcribe(str(audio_path), **options)
        
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", [])
        }
