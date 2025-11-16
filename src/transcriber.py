import whisper
from pathlib import Path
import torch
import re
import warnings

# Filter out specific Whisper warnings about Triton/CUDA
warnings.filterwarnings('ignore', message='Failed to launch Triton kernels')

class WhisperTranscriber:
    def __init__(self, model_size=None):
        import os
        if model_size is None:
            model_size = os.getenv('WHISPER_MODEL_SIZE', 'medium')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size).to(device)
        self.device = device
        self.model_size = model_size
        # Default initial prompt for personal note-taking context
        self.default_prompt = (
            "This is a personal note or journal entry. "
            "The speaker is sharing thoughts, ideas, or reflections "
            "in a natural, conversational style. "
            "Maintain proper sentence structure and punctuation."
        )

    def transcribe(self, audio_path: Path) -> dict:
        """
        Transcribe an audio file using Whisper
        
        Args:
            audio_path (Path): Path to the audio file
            
        Returns:
            dict: Dictionary containing transcribed text and metadata
        """
        try:
            # Use better settings for GPU processing
            options = {
                "fp16": False,  # Use full precision for better quality
                "beam_size": 5,  # Increase beam size for better accuracy
                "best_of": 3,    # Reduced from 5 to prevent over-analysis
                "temperature": [0.0],  # Single temperature to prevent variation
                "task": "transcribe",
                "initial_prompt": self.default_prompt,
                "condition_on_previous_text": False,  # Disabled to prevent context loop
                "compression_ratio_threshold": 1.8,  # More aggressive threshold to prevent repetition
                "no_speech_threshold": 0.6,  # More aggressive filtering of non-speech
                "word_timestamps": True,    # Enable word timestamps for better segmentation
            }
            
            result = self.model.transcribe(str(audio_path), **options)
            
            # Extract and structure the metadata
            processed_result = {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": []
            }
            
            # Process each segment
            for segment in result.get("segments", []):
                processed_segment = {
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": segment.get("avg_logprob", 0),
                    "no_speech_prob": segment.get("no_speech_prob", 0),
                    "words": []
                }
                
                # Process word-level information if available
                for word in segment.get("words", []):
                    processed_segment["words"].append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("probability", 0)
                    })
                
                processed_result["segments"].append(processed_segment)
            
            # Clean up the main text
            processed_result["text"] = self._clean_text(processed_result["text"])
            
            return processed_result
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean up the transcribed text."""
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
        # Remove any repeated phrases (3 or more words that repeat)
        text = self._remove_repeated_phrases(text)
        return text

    def _remove_repeated_phrases(self, text):
        """Remove repeated phrases of 3 or more words from the text."""
        words = text.split()
        if len(words) < 6:  # Too short to have meaningful repetition
            return text
            
        # Build cleaned text by checking for repetitions
        cleaned_words = []
        i = 0
        while i < len(words):
            # Skip if we're too close to the end
            if i > len(words) - 3:
                cleaned_words.extend(words[i:])
                break
                
            # Check for repeating phrases of different lengths (3 to 6 words)
            repeated = False
            for phrase_length in range(3, 7):
                if i + phrase_length > len(words):
                    continue
                    
                phrase = words[i:i + phrase_length]
                # Look ahead for the same phrase
                next_pos = i + phrase_length
                if next_pos + phrase_length <= len(words):
                    next_phrase = words[next_pos:next_pos + phrase_length]
                    if phrase == next_phrase:
                        repeated = True
                        i += phrase_length  # Skip the repeated phrase
                        break
            
            if not repeated:
                cleaned_words.append(words[i])
                i += 1
        
        # Remove any remaining excessive whitespace
        return ' '.join(cleaned_words).strip()
