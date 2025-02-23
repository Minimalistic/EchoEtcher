import os
import json
import requests
import logging
import re
import time
from typing import Dict, Optional
from pathlib import Path
from .tag_manager import TagManager

class OllamaProcessor:
    def __init__(self):
        self.api_url = os.getenv('OLLAMA_API_URL')
        self.model = os.getenv('OLLAMA_MODEL')
        if not self.model:
            raise ValueError("OLLAMA_MODEL must be set in environment variables")
        self.temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.3'))
        vault_path = Path(os.getenv('OBSIDIAN_VAULT_PATH'))
        self.tag_manager = TagManager(vault_path)
        self.max_retries = 3
        self.base_delay = 1  # Base delay for exponential backoff in seconds
        
        # Timeout settings
        self.connect_timeout = 10  # Timeout for initial connection
        self.read_timeout = int(os.getenv('OLLAMA_TIMEOUT', '120'))  # Timeout for reading response
        self.backoff_factor = 2  # Factor to increase timeout with each retry

    def clean_text(self, text: str) -> str:
        """Clean text by removing problematic characters while preserving meaningful whitespace."""
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        # Normalize whitespace while preserving intentional line breaks
        text = re.sub(r'[\r\f\v]+', '\n', text)
        # Remove zero-width characters and other invisible unicode
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        # Normalize unicode quotes and dashes
        text = text.replace('"', '"').replace('"', '"').replace('—', '-')
        return text

    def clean_json_string(self, text: str) -> str:
        """Clean a string that should be valid JSON."""
        # Log the original input for debugging
        logging.debug(f"Original JSON string: {text[:200]}...")  # First 200 chars
        
        try:
            # First try: direct JSON parsing
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            logging.info(f"Initial JSON parse failed: {str(e)}, attempting repairs...")
            
            # Replace single quotes with double quotes, but only for property names and string values
            text = re.sub(r"'([^']*)':", r'"\1":', text)  # Fix property names
            text = re.sub(r":\s*'([^']*)'", r':"\1"', text)  # Fix string values
            
            # Remove any trailing commas in objects and arrays
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            
            # Ensure property names are quoted
            text = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
            
            logging.debug(f"Cleaned JSON string: {text[:200]}...")  # First 200 chars
            
            # Verify the cleaning worked
            try:
                json.loads(text)
                logging.info("JSON repair successful")
                return text
            except json.JSONDecodeError as e:
                logging.error(f"JSON repair failed: {str(e)}")
                logging.error(f"Failed JSON: {text}")
                return text  # Return the cleaned version anyway, let the caller handle any remaining issues

    def attempt_json_repair(self, text: str) -> Optional[Dict]:
        """Attempt to repair malformed JSON response."""
        try:
            # First try: Basic string cleanup
            cleaned = text.strip()
            # Remove any trailing commas before closing braces/brackets
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            # Ensure proper quote usage
            cleaned = re.sub(r'(?<!\\)"', '"', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # Second try: Find the largest valid JSON substring
                # This helps when the response has extra text before/after the JSON
                json_pattern = r'\{(?:[^{}]|(?R))*\}'
                matches = re.finditer(json_pattern, text)
                largest_valid_json = None
                max_length = 0
                
                for match in matches:
                    potential_json = match.group(0)
                    try:
                        parsed = json.loads(potential_json)
                        if len(potential_json) > max_length:
                            largest_valid_json = parsed
                            max_length = len(potential_json)
                    except json.JSONDecodeError:
                        continue
                
                return largest_valid_json
            except Exception:
                return None

    def call_ollama_with_retry(self, prompt: str) -> Optional[Dict]:
        """Make Ollama API call with exponential backoff retry and adaptive timeouts."""
        for attempt in range(self.max_retries):
            current_read_timeout = self.read_timeout * (self.backoff_factor ** attempt)
            try:
                logging.info(f"Attempt {attempt + 1}/{self.max_retries} with {current_read_timeout}s timeout")
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": self.temperature
                    },
                    timeout=(self.connect_timeout, current_read_timeout)  # (connect timeout, read timeout)
                )
                response.raise_for_status()
                
                try:
                    return response.json()
                except json.JSONDecodeError:
                    logging.warning("Received malformed JSON from Ollama, attempting repair...")
                    repaired_json = self.attempt_json_repair(response.text)
                    if repaired_json:
                        logging.info("Successfully repaired malformed JSON response")
                        return repaired_json
                    else:
                        raise json.JSONDecodeError("Failed to repair JSON", response.text, 0)
                        
            except requests.Timeout as e:
                delay = self.base_delay * (2 ** attempt)
                logging.warning(
                    f"Timeout during attempt {attempt + 1} (timeout={current_read_timeout}s): {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    logging.info(f"Retrying in {delay} seconds with increased timeout...")
                    time.sleep(delay)
                else:
                    logging.error(
                        f"All Ollama API attempts failed after {self.max_retries} retries "
                        f"with max timeout of {current_read_timeout}s"
                    )
                    raise
            except (requests.RequestException, json.JSONDecodeError) as e:
                delay = self.base_delay * (2 ** attempt)
                logging.warning(f"Ollama API attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error("All Ollama API attempts failed")
                    raise

    def clean_formatted_content(self, content: str) -> str:
        """Clean the formatted content by removing prompt artifacts and transcription markers."""
        # Remove any lines about allowed tags
        content = re.sub(r'You must only use tags from the ALLOWED TAGS list above\..*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = re.sub(r'If no tags from the allowed list are relevant.*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove transcription markers and metadata
        content = re.sub(r'\[uncertain\]', '', content)
        content = re.sub(r'\[Pause: [^\]]+\]', '', content)  # Remove pause markers
        content = re.sub(r'\[Non-speech section: [^\]]+\]', '', content)  # Remove non-speech markers
        content = re.sub(r'\[Low confidence section: [^\]]+\]', '', content)  # Remove confidence markers
        
        # Remove any empty lines that might have been created
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()

    def process_transcription(self, transcription_data: dict, audio_filename: str) -> Dict:
        """Process the transcription to generate tags and a title."""
        try:
            logging.info("Starting Ollama processing...")
            
            # Clean the transcription before processing
            cleaned_transcription = self.clean_text(transcription_data["text"])
            
            # Get the list of allowed tags
            allowed_tags = list(self.tag_manager._allowed_tags)
            allowed_tags_str = '\n'.join(allowed_tags)
            
            # Build metadata information
            metadata = []
            
            # Add language information
            if transcription_data.get("language"):
                metadata.append(f"Language: {transcription_data['language']}")
            
            # Process segments to identify notable features
            segments_info = []
            for segment in transcription_data.get("segments", []):
                # Calculate pause before this segment
                if segments_info:  # Not the first segment
                    pause_duration = segment["start"] - segments_info[-1]["end"]
                    if pause_duration > 1.0:  # Only note significant pauses
                        segments_info.append({
                            "type": "pause",
                            "duration": round(pause_duration, 1),
                            "position": segment["start"]
                        })
                
                # Add segment with confidence info
                segments_info.append({
                    "type": "segment",
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": segment["confidence"],
                    "no_speech_prob": segment["no_speech_prob"]
                })
            
            # Convert segments info to readable format
            for info in segments_info:
                if info["type"] == "pause":
                    metadata.append(f"[Pause: {info['duration']}s at {info['position']}s]")
                elif info["type"] == "segment":
                    if info["no_speech_prob"] > 0.5:  # Likely background noise or non-speech
                        metadata.append(f"[Non-speech section: {info['start']}-{info['end']}s]")
                    elif info["confidence"] < -1.0:  # Low confidence section
                        metadata.append(f"[Low confidence section: {info['start']}-{info['end']}s]")
            
            metadata_str = "\n".join(metadata)
            
            prompt = f"""Given this transcription from an audio note with metadata and the list of ALLOWED TAGS below:

METADATA:
{metadata_str}

1. Format the transcription by:
   - Using the metadata to inform natural breaks and section divisions
   - Adding appropriate line breaks where pauses or topic changes occur
   - Using markdown headers (# or ##) for major topic changes or sections
   - Adding [uncertain] tags for low confidence sections
   - Preserving the original meaning and content
   - Using single line breaks between paragraphs
2. Generate a clear, concise filename (without extension)
3. Select the most relevant tags from ONLY the allowed tags list that apply to this content

IMPORTANT: Your response must be valid JSON with double quotes around property names and string values.

ALLOWED TAGS:
{allowed_tags_str}

Original audio filename: {audio_filename}
Transcription: {cleaned_transcription}

You must ONLY use tags from the ALLOWED TAGS list above. Do not create new tags.
If no tags from the allowed list are relevant, return an empty list.

Respond in this exact format:
{{
    "title": "clear-descriptive-filename",
    "tags": ["#tag1", "#tag2"],
    "formatted_content": "The formatted transcription with single line breaks"
}}"""

            response = self.call_ollama_with_retry(prompt)
            if not response or "response" not in response:
                raise ValueError("Invalid response from Ollama")

            # Log the raw response for debugging
            logging.debug(f"Raw Ollama response: {response['response'][:500]}...")

            try:
                # Clean the response string before attempting to parse JSON
                cleaned_response = self.clean_json_string(response["response"])
                result = json.loads(cleaned_response)
                
                # Validate the result has required fields
                if not all(k in result for k in ["title", "tags", "formatted_content"]):
                    raise ValueError("Missing required fields in Ollama response")
                
                # Clean the formatted content
                result["formatted_content"] = self.clean_formatted_content(result["formatted_content"])
                
                # Validate tags are from allowed list
                result["tags"] = [tag for tag in result["tags"] if tag in allowed_tags]
                
                return result
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {str(e)}")
                logging.error(f"Problematic JSON: {cleaned_response[:500]}...")  # Log first 500 chars
                raise ValueError(f"Invalid JSON in Ollama response: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error processing transcription: {str(e)}")
            raise
