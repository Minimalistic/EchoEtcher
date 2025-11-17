import os
import json
import requests
import logging
import re
import time
import yaml
from typing import Dict, Optional, Tuple
from pathlib import Path
from .tag_manager import TagManager

class OllamaProcessor:
    def __init__(self):
        self.api_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
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

    def parse_yaml_frontmatter(self, text: str) -> Dict:
        """
        Parse YAML frontmatter from markdown text.
        Expected format:
        ---
        title: ...
        tags: [...]
        ---
        
        Content here...
        """
        text = text.strip()
        
        # Check if it starts with frontmatter delimiter
        if not text.startswith('---'):
            # Try to find frontmatter if there's leading whitespace or text
            # Some models might add explanatory text before the frontmatter
            frontmatter_start = text.find('\n---')
            if frontmatter_start != -1:
                # Found frontmatter after some text, extract from there
                text = text[frontmatter_start + 1:].strip()
            else:
                raise ValueError("Response does not contain YAML frontmatter delimiter (---)")
        
        # Find the end of frontmatter (second ---)
        lines = text.split('\n')
        if len(lines) < 2:
            raise ValueError("Invalid YAML frontmatter format")
        
        # Find the closing --- (must be on its own line)
        frontmatter_end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                frontmatter_end = i
                break
        
        if frontmatter_end is None:
            raise ValueError("YAML frontmatter not properly closed (missing closing ---)")
        
        # Extract frontmatter (skip first line with ---)
        frontmatter_lines = lines[1:frontmatter_end]
        frontmatter_text = '\n'.join(frontmatter_lines)
        
        # Parse YAML
        try:
            frontmatter = yaml.safe_load(frontmatter_text)
            if frontmatter is None:
                frontmatter = {}  # Empty frontmatter
            if not isinstance(frontmatter, dict):
                raise ValueError("YAML frontmatter must be a dictionary")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in frontmatter: {str(e)}")
        
        # Extract content (everything after the closing ---)
        content_lines = lines[frontmatter_end + 1:]
        content = '\n'.join(content_lines).strip()
        
        # Build result dictionary
        result = {
            "title": frontmatter.get("title", ""),
            "tags": frontmatter.get("tags", []),
            "formatted_content": content,
            "ai_summary": frontmatter.get("summary", "").strip()
        }
        
        # Validate required fields
        if not result["title"]:
            raise ValueError("YAML frontmatter missing required field: title")
        
        return result

    def extract_summary_if_present(self, content: str) -> Tuple[str, str]:
        """
        Extract AI-generated summary paragraphs if present, returning (summary, main_content).
        Only extracts clearly meta-commentary paragraphs, not legitimate spoken content.
        
        Returns:
            tuple: (summary_text, main_content) - summary will be empty string if none found
        """
        if not content:
            return "", content
        
        # Look for meta-commentary paragraphs that are clearly AI-generated summaries
        # These typically appear at the end and reference "this transcription/audio note" in third person
        # Pattern: Paragraphs that start with "This transcription/audio note/recording discusses/is about..."
        # We're being conservative - only matching clear meta-commentary, not legitimate speech
        
        summary_pattern = r'\n\n\s*(This (?:transcription|audio note|recording|note|audio)|The (?:transcription|audio note|recording|note)).*?(?:discusses|is about|describes|covers|mentions|refers to|is intended for|is for).*?(?=\n\n|\n#|$)'
        
        match = re.search(summary_pattern, content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            # Found a summary paragraph - extract it
            summary_text = match.group(0).strip()
            # Remove it from the main content
            main_content = content[:match.start()] + content[match.end():]
            return summary_text, main_content.strip()
        
        return "", content

    def clean_formatted_content(self, content: str) -> str:
        """Clean and validate the formatted content with improved markdown processing."""
        if not content:
            return content
        
        # Remove any code block markers (```) that might be at the start or end
        content = re.sub(r'^```+\s*\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n?```+\s*$', '', content, flags=re.MULTILINE)
        
        # Remove any lines about allowed tags or instructions
        content = re.sub(r'You must only use tags from the ALLOWED TAGS list above\..*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = re.sub(r'If no tags from the allowed list are relevant.*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove transcription markers and metadata (but preserve [uncertain] if it's meaningful)
        # Only remove pause/non-speech markers that are in brackets
        content = re.sub(r'\[Pause: [^\]]+\]', '', content)  # Remove pause markers
        content = re.sub(r'\[Non-speech section: [^\]]+\]', '', content)  # Remove non-speech markers
        content = re.sub(r'\[Low confidence section: [^\]]+\]', '', content)  # Remove confidence markers
        
        # Fix common markdown issues
        # Ensure headers have proper spacing
        content = re.sub(r'^(#{1,6})\s*([^\n]+)', r'\1 \2', content, flags=re.MULTILINE)
        # Fix headers without space after #
        content = re.sub(r'^#+([^\s#])', r'# \1', content, flags=re.MULTILINE)
        
        # Normalize bullet points (ensure space after - or *)
        # Note: - must be escaped or at end of character class to be literal
        content = re.sub(r'^([-*])([^\s*\-])', r'\1 \2', content, flags=re.MULTILINE)
        
        # Remove trailing whitespace from lines
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Ensure content starts and ends cleanly
        content = content.strip()
        
        # Normalize paragraph breaks: preserve double newlines (paragraph breaks)
        # but collapse excessive blank lines (more than 2 consecutive)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Ensure proper spacing around headers (add blank line before if missing)
        content = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', content)
        # Ensure blank line after headers
        content = re.sub(r'(#{1,6} [^\n]+)\n([^\n#])', r'\1\n\n\2', content)
        
        # Ensure proper spacing around lists
        content = re.sub(r'([^\n])\n([-*] )', r'\1\n\n\2', content)  # Before list
        content = re.sub(r'([-*] [^\n]+)\n([^\n\-*#])', r'\1\n\n\2', content)  # After list
        
        # If the content is all one paragraph (no double newlines), try to add breaks
        # after sentences that end with periods followed by capital letters (likely topic changes)
        # But only if there are no existing paragraph breaks
        if '\n\n' not in content and len(content) > 200:
            # Add paragraph breaks after sentences ending with . ! or ? followed by a capital letter
            # This helps break up long paragraphs into more readable chunks
            content = re.sub(r'([.!?])\s+([A-Z][a-z])', r'\1\n\n\2', content)
            # Clean up any triple newlines this might create
            content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Final cleanup: remove any trailing code blocks or backticks
        content = content.rstrip()
        while content.endswith('`'):
            content = content.rstrip('`').rstrip()
        
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
            low_confidence_segments = []
            
            for segment in transcription_data.get("segments", []):
                # Calculate pause before this segment
                if segments_info:  # Not the first segment
                    last_item = segments_info[-1]
                    if last_item["type"] == "segment":
                        pause_duration = segment["start"] - last_item["end"]
                        if pause_duration > 1.0:  # Only note significant pauses
                            segments_info.append({
                                "type": "pause",
                                "duration": round(pause_duration, 1),
                                "position": segment["start"]
                            })
                
                # Add segment with confidence info
                segment_info = {
                    "type": "segment",
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": segment["confidence"],
                    "no_speech_prob": segment["no_speech_prob"],
                    "text": segment.get("text", "")
                }
                segments_info.append(segment_info)
                
                # Track low confidence segments for better handling
                if segment_info["confidence"] < -1.0 and segment_info["no_speech_prob"] < 0.5:
                    low_confidence_segments.append(segment_info)
            
            # Convert segments info to readable format for metadata
            for info in segments_info:
                if info["type"] == "pause":
                    metadata.append(f"[Pause: {info['duration']}s at {info['position']}s]")
                elif info["type"] == "segment":
                    if info["no_speech_prob"] > 0.5:  # Likely background noise or non-speech
                        metadata.append(f"[Non-speech section: {info['start']}-{info['end']}s]")
                    elif info["confidence"] < -1.0:  # Low confidence section
                        metadata.append(f"[Low confidence section: {info['start']}-{info['end']}s, confidence: {info['confidence']:.2f}]")
            
            # Add summary of low confidence segments if any
            if low_confidence_segments:
                metadata.append(f"[Note: {len(low_confidence_segments)} low-confidence segments detected - review for accuracy]")
            
            metadata_str = "\n".join(metadata)
            
            prompt = f"""Given this transcription from an audio note with metadata and the list of ALLOWED TAGS below:

METADATA:
{metadata_str}

Your task is to transform this raw transcription into a well-formatted markdown note:

1. FORMATTING GUIDELINES:
   - CRITICAL: Use double line breaks (blank line) between paragraphs and distinct thoughts
   - Each new topic, idea, or major thought should start a new paragraph with a blank line before it
   - Use markdown headers (# for main sections, ## for subsections) when there are clear topic changes
   - Use bullet points (- or *) for lists or key points
   - Use **bold** for emphasis on important concepts
   - Preserve natural flow and readability
   - Add [uncertain] markers only for genuinely low-confidence sections (from metadata)
   - Use single line breaks within paragraphs, double line breaks (blank line) between paragraphs
   - Ensure proper capitalization and punctuation throughout
   - DO NOT use code blocks (```) - just write plain markdown text

2. CONTENT ORGANIZATION:
   - Identify natural topic transitions based on pauses and content changes
   - Group related thoughts into coherent paragraphs
   - Create sections with headers when topics shift significantly
   - Maintain the speaker's original meaning and intent

3. TITLE GENERATION:
   - Create a clear, descriptive filename (without extension)
   - Use lowercase with hyphens (e.g., "meeting-notes-project-discussion")
   - Keep it concise but informative (30-50 characters ideal)

4. TAG SELECTION:
   - Select ONLY from the ALLOWED TAGS list below
   - Choose 2-5 most relevant tags that best describe the content
   - If no tags are relevant, return an empty list

5. SUMMARY GENERATION:
   - Create a brief 1-2 sentence summary of the transcription content
   - This summary will be displayed at the top of the note
   - Focus on the main topic or key points discussed
   - Keep it concise and informative

IMPORTANT: Your response must use YAML frontmatter format (standard markdown with YAML metadata block).

ALLOWED TAGS:
{allowed_tags_str}

Original audio filename: {audio_filename}
Transcription: {cleaned_transcription}

You must ONLY use tags from the ALLOWED TAGS list above. Do not create new tags.
If no tags from the allowed list are relevant, use an empty list: []

Respond in this exact format (YAML frontmatter followed by markdown content):
---
title: clear-descriptive-filename
tags: ["#tag1", "#tag2"]
summary: "Brief 1-2 sentence summary of the transcription content"
---

The formatted transcription with proper markdown formatting goes here.
ONLY include the formatted transcription content - do NOT add summaries or explanations in the body."""

            response = self.call_ollama_with_retry(prompt)
            if not response or "response" not in response:
                raise ValueError("Invalid response from Ollama")

            # Log the raw response for debugging
            raw_response = response["response"]
            logging.debug(f"Raw Ollama response: {raw_response[:500]}...")

            try:
                # Parse YAML frontmatter from the response
                result = self.parse_yaml_frontmatter(raw_response)
                
                # Validate the result has required fields
                if not all(k in result for k in ["title", "tags", "formatted_content"]):
                    raise ValueError("Missing required fields in Ollama response (need: title, tags, formatted_content)")
                
                # Clean the formatted content
                result["formatted_content"] = self.clean_formatted_content(result["formatted_content"])
                
                # If summary wasn't in frontmatter, try to extract it from content
                if not result.get("ai_summary"):
                    summary, main_content = self.extract_summary_if_present(result["formatted_content"])
                    if summary:
                        result["ai_summary"] = self.clean_formatted_content(summary)
                        result["formatted_content"] = self.clean_formatted_content(main_content)
                
                # Validate tags are from allowed list
                result["tags"] = [tag for tag in result["tags"] if tag in allowed_tags]
                
                return result
                
            except (yaml.YAMLError, ValueError) as e:
                logging.error(f"YAML frontmatter parse error: {str(e)}")
                logging.error(f"Problematic response: {raw_response[:1000]}...")  # Log first 1000 chars
                raise ValueError(f"Invalid YAML frontmatter in Ollama response: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error processing transcription: {str(e)}")
            raise
