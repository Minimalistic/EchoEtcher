import whisper
from pathlib import Path
import torch
import re
import warnings
import logging
import subprocess
import os
import tempfile
import json

# Filter out specific Whisper warnings about Triton/CUDA
warnings.filterwarnings('ignore', message='Failed to launch Triton kernels')

class WhisperTranscriber:
    def __init__(self, model_size=None, lazy_load=False):
        import os
        import platform
        if model_size is None:
            model_size = os.getenv('WHISPER_MODEL_SIZE', 'medium')
        
        # Auto-detect best available device
        self.device = self._detect_best_device()
        self.model_size = model_size
        self.model = None
        self._device_info = None
        
        # Chunking configuration for large files
        # Default: chunk files longer than 4 minutes (240 seconds)
        self.chunk_threshold_seconds = float(os.getenv('WHISPER_CHUNK_THRESHOLD', '240'))
        # Default: 30-second chunks with 5-second overlap
        self.chunk_duration_seconds = float(os.getenv('WHISPER_CHUNK_DURATION', '30'))
        self.chunk_overlap_seconds = float(os.getenv('WHISPER_CHUNK_OVERLAP', '5'))
        
        # Default initial prompt for personal note-taking context
        # Enhanced prompt with more context for better transcription accuracy
        self.default_prompt = (
            "This is a personal note or journal entry. "
            "The speaker is sharing thoughts, ideas, or reflections "
            "in a natural, conversational style. "
            "Use proper capitalization, punctuation, and sentence structure. "
            "Transcribe spoken words accurately, including proper nouns, technical terms, "
            "and common phrases. Maintain natural flow and readability."
        )
        
        # Load model immediately unless lazy loading is requested
        if not lazy_load:
            self._load_model()
    
    def _load_model(self):
        """Load the Whisper model into memory."""
        if self.model is not None:
            return  # Already loaded
        
        logging.info(f"Loading Whisper model: {self.model_size}")
        self.model = whisper.load_model(self.model_size).to(self.device)
        
        # Log device information
        self._device_info = self._get_device_info()
        logging.info(f"Whisper model loaded on device: {self._device_info}")
    
    def unload_model(self):
        """Unload the Whisper model from memory to free up resources."""
        if self.model is None:
            return  # Already unloaded
        
        logging.info(f"Unloading Whisper model to free memory...")
        
        # Move model to CPU and delete
        if self.device == "cuda":
            self.model = self.model.cpu()
            torch.cuda.empty_cache()  # Clear CUDA cache
        
        del self.model
        self.model = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logging.info("Whisper model unloaded successfully")
    
    def ensure_loaded(self):
        """Ensure the model is loaded, loading it if necessary."""
        if self.model is None:
            self._load_model()
    
    def _detect_best_device(self):
        """
        Auto-detect the best available device for processing.
        Priority: CUDA (NVIDIA GPU) > CPU
        
        Note: MPS (Apple Silicon GPU) is skipped due to compatibility issues
        with Whisper's sparse tensor operations. CPU is used on macOS instead.
        """
        # Check for CUDA (NVIDIA GPU on Windows/Linux)
        if torch.cuda.is_available():
            return "cuda"
        
        # Skip MPS on macOS - Whisper has compatibility issues with MPS backend
        # due to unsupported sparse tensor operations. CPU works reliably.
        # if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     return "mps"
        
        # Fall back to CPU (including on macOS)
        return "cpu"
    
    def _get_device_info(self):
        """Get detailed information about the device being used."""
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"CUDA ({gpu_name}, {gpu_memory:.1f}GB)"
        elif self.device == "mps":
            return "MPS (Apple Silicon GPU)"
        else:
            import platform
            cpu_info = platform.processor() or "Unknown"
            return f"CPU ({cpu_info})"
    
    def is_loaded(self):
        """Check if the model is currently loaded in memory."""
        return self.model is not None

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get the duration of an audio file in seconds using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'json', str(audio_path)
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=True
            )
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
            logging.warning(f"Could not determine audio duration: {e}. Assuming file needs chunking.")
            return float('inf')  # Return infinity to trigger chunking
        except FileNotFoundError:
            logging.warning("ffprobe not found. Cannot determine audio duration. Assuming file needs chunking.")
            return float('inf')

    def _create_audio_chunk(self, audio_path: Path, start_time: float, duration: float, output_path: Path) -> bool:
        """Create a chunk of audio using ffmpeg."""
        # First try with codec copy (faster, no re-encoding)
        try:
            cmd = [
                'ffmpeg', '-i', str(audio_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-acodec', 'copy',  # Copy codec to avoid re-encoding (faster)
                '-y',  # Overwrite output file
                str(output_path)
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=True
            )
            return True
        except subprocess.CalledProcessError:
            # If codec copy fails, try re-encoding (slower but more compatible)
            logging.debug(f"Codec copy failed for chunk, trying re-encoding...")
            try:
                cmd = [
                    'ffmpeg', '-i', str(audio_path),
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-acodec', 'aac',  # Re-encode to AAC (widely compatible)
                    '-b:a', '128k',    # Audio bitrate
                    '-y',  # Overwrite output file
                    str(output_path)
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120, check=True
                )
                return True
            except subprocess.CalledProcessError as e:
                logging.error(f"Error creating audio chunk (both copy and re-encode failed): {e.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout creating audio chunk")
            return False

    def _transcribe_chunked(self, audio_path: Path) -> dict:
        """Transcribe a large audio file by splitting it into chunks."""
        duration = self._get_audio_duration(audio_path)
        
        if duration == float('inf'):
            # If we can't determine duration, try to process as single file
            logging.warning("Could not determine audio duration, attempting single-file transcription")
            return self._transcribe_single(audio_path)
        
        logging.info(f"Audio file duration: {duration:.1f} seconds. Using chunked transcription.")
        
        # Calculate number of chunks needed
        effective_chunk_duration = self.chunk_duration_seconds - self.chunk_overlap_seconds
        num_chunks = int((duration + effective_chunk_duration - 1) // effective_chunk_duration) + 1
        
        logging.info(f"Processing {num_chunks} chunks (chunk size: {self.chunk_duration_seconds}s, overlap: {self.chunk_overlap_seconds}s)")
        
        all_segments = []
        detected_language = None
        temp_dir = None
        
        try:
            # Create temporary directory for chunks
            temp_dir = tempfile.mkdtemp(prefix='whisper_chunks_')
            temp_dir_path = Path(temp_dir)
            
            for chunk_idx in range(num_chunks):
                start_time = chunk_idx * effective_chunk_duration
                
                # Don't process beyond the actual duration
                if start_time >= duration:
                    break
                
                # Adjust chunk duration for the last chunk
                chunk_duration = min(self.chunk_duration_seconds, duration - start_time)
                
                if chunk_duration <= 0:
                    break
                
                logging.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (time: {start_time:.1f}s - {start_time + chunk_duration:.1f}s)")
                
                # Create chunk file (use same extension as original, or .m4a as fallback)
                chunk_ext = audio_path.suffix if audio_path.suffix else '.m4a'
                chunk_path = temp_dir_path / f"chunk_{chunk_idx:04d}{chunk_ext}"
                if not self._create_audio_chunk(audio_path, start_time, chunk_duration, chunk_path):
                    logging.warning(f"Failed to create chunk {chunk_idx + 1}, skipping...")
                    continue
                
                # Transcribe chunk
                try:
                    chunk_result = self._transcribe_single(chunk_path)
                    
                    # Store language from first chunk
                    if detected_language is None and chunk_result.get("language"):
                        detected_language = chunk_result["language"]
                    
                    # Adjust segment timestamps to account for chunk offset
                    for segment in chunk_result.get("segments", []):
                        segment["start"] += start_time
                        segment["end"] += start_time
                        # Adjust word timestamps too
                        for word in segment.get("words", []):
                            word["start"] += start_time
                            word["end"] += start_time
                    
                    all_segments.extend(chunk_result.get("segments", []))
                    
                except Exception as e:
                    logging.error(f"Error transcribing chunk {chunk_idx + 1}: {e}")
                    continue
                finally:
                    # Clean up chunk file
                    try:
                        chunk_path.unlink()
                    except:
                        pass
            
            # Merge segments intelligently
            merged_segments = self._merge_segments(all_segments)
            
            # Combine text from all segments
            full_text = " ".join([seg["text"] for seg in merged_segments])
            full_text = self._clean_text(full_text)
            
            return {
                "text": full_text,
                "language": detected_language or "unknown",
                "segments": merged_segments
            }
            
        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logging.warning(f"Could not clean up temp directory {temp_dir}: {e}")

    def _merge_segments(self, segments: list) -> list:
        """Merge overlapping segments from chunked transcription."""
        if not segments:
            return []
        
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x["start"])
        
        merged = []
        current_segment = None
        
        for segment in segments:
            if current_segment is None:
                current_segment = segment.copy()
                continue
            
            # Check for overlap (within overlap window)
            overlap_threshold = self.chunk_overlap_seconds
            
            # If segments are close together or overlapping, merge them
            if segment["start"] <= current_segment["end"] + overlap_threshold:
                # Merge: extend end time and combine text
                current_segment["end"] = max(current_segment["end"], segment["end"])
                current_segment["text"] = current_segment["text"].rstrip() + " " + segment["text"].lstrip()
                
                # Merge words if available
                if "words" in current_segment and "words" in segment:
                    current_segment["words"].extend(segment["words"])
                    # Sort words by start time
                    current_segment["words"].sort(key=lambda x: x["start"])
                
                # Update confidence (average)
                if "confidence" in current_segment and "confidence" in segment:
                    current_segment["confidence"] = (current_segment["confidence"] + segment["confidence"]) / 2
            else:
                # No overlap, save current and start new
                merged.append(current_segment)
                current_segment = segment.copy()
        
        # Add the last segment
        if current_segment is not None:
            merged.append(current_segment)
        
        return merged

    def _transcribe_single(self, audio_path: Path) -> dict:
        """Transcribe a single audio file (non-chunked)."""
        # Optimize settings based on device
        # CUDA can use fp16 for faster processing, CPU should use fp32
        # Note: MPS is not used due to compatibility issues
        use_fp16 = self.device == "cuda"
        
        # Use multiple temperatures for better accuracy (greedy decoding with fallback)
        # This helps with difficult audio while maintaining consistency
        temperatures = [0.0, 0.2, 0.4] if self.device == "cuda" else [0.0, 0.2]
        
        options = {
            "fp16": use_fp16,  # Use fp16 on GPU for speed, fp32 on CPU for accuracy
            "beam_size": 5,  # Increase beam size for better accuracy
            "best_of": 3,    # Reduced from 5 to prevent over-analysis
            "temperature": temperatures,  # Multiple temperatures for better accuracy
            "task": "transcribe",
            "initial_prompt": self.default_prompt,
            "condition_on_previous_text": False,  # Disabled to prevent context loop
            "compression_ratio_threshold": 1.8,  # More aggressive threshold to prevent repetition
            "no_speech_threshold": 0.6,  # More aggressive filtering of non-speech
            "word_timestamps": True,    # Enable word timestamps for better segmentation
        }
        
        logging.debug(f"Transcription options: fp16={use_fp16}, device={self.device}")
        
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

    def transcribe(self, audio_path: Path) -> dict:
        """
        Transcribe an audio file using Whisper.
        Automatically uses chunking for large files to prevent memory issues.
        
        Args:
            audio_path (Path): Path to the audio file
            
        Returns:
            dict: Dictionary containing transcribed text and metadata
        """
        # Ensure model is loaded before transcription
        self.ensure_loaded()
        
        try:
            # Check if file needs chunking
            duration = self._get_audio_duration(audio_path)
            
            if duration > self.chunk_threshold_seconds:
                logging.info(f"File duration ({duration:.1f}s) exceeds threshold ({self.chunk_threshold_seconds}s). Using chunked transcription.")
                return self._transcribe_chunked(audio_path)
            else:
                logging.debug(f"File duration ({duration:.1f}s) is within threshold. Using single-file transcription.")
                return self._transcribe_single(audio_path)
                
        except MemoryError as e:
            logging.error(f"Memory error during transcription. File may be too large. Consider reducing WHISPER_CHUNK_DURATION.")
            raise Exception(f"Transcription failed due to memory constraints: {str(e)}")
        except Exception as e:
            error_msg = str(e)
            # If single-file transcription fails, try chunking as fallback
            # Get duration again for fallback check
            try:
                duration = self._get_audio_duration(audio_path)
                if "chunked" not in error_msg.lower() and duration > 60:  # Only retry if file is reasonably long
                    logging.warning(f"Single-file transcription failed, attempting chunked transcription as fallback: {error_msg}")
                    try:
                        return self._transcribe_chunked(audio_path)
                    except Exception as chunk_error:
                        raise Exception(f"Transcription failed (both single and chunked attempts): {str(chunk_error)}")
            except:
                pass  # If we can't get duration, just raise the original error
            raise Exception(f"Transcription failed: {error_msg}")

    def _clean_text(self, text: str) -> str:
        """Clean up the transcribed text with improved formatting."""
        if not text or not text.strip():
            return text
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Fix common punctuation issues (remove spaces before punctuation)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.,!?;:])([^\s\.,!?;:])', r'\1 \2', text)
        
        # Fix capitalization at sentence starts
        # Capitalize first letter of text
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Capitalize after sentence-ending punctuation
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Fix common transcription errors
        text = re.sub(r'\bi\s+', 'I ', text)  # Fix lowercase 'i' at word boundaries
        text = re.sub(r'\s+i\s+', ' I ', text)  # Fix standalone lowercase 'i'
        
        # Remove any repeated phrases (3 or more words that repeat)
        text = self._remove_repeated_phrases(text)
        
        # Final cleanup: normalize whitespace
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on new lines
        
        return text.strip()

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
