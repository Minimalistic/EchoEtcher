import os
import json
import requests
import logging
import re
import time
import yaml
import base64
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from .tag_manager import TagManager

class OllamaProcessor:
    def __init__(self):
        self.api_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
        self.model = os.getenv('OLLAMA_MODEL')
        if not self.model:
            raise ValueError("OLLAMA_MODEL must be set in environment variables")
        # Vision model for image analysis (optional, falls back to main model if not set)
        self.vision_model = os.getenv('OLLAMA_VISION_MODEL', self.model)
        self.temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.3'))
        vault_path = Path(os.getenv('OBSIDIAN_VAULT_PATH'))
        self.tag_manager = TagManager(vault_path)
        self.max_retries = 3
        self.base_delay = 1  # Base delay for exponential backoff in seconds
        
        # Timeout settings
        self.connect_timeout = 10  # Timeout for initial connection
        self.read_timeout = int(os.getenv('OLLAMA_TIMEOUT', '120'))  # Timeout for reading response
        self.backoff_factor = 2  # Factor to increase timeout with each retry
        
        # Log model configuration
        if self.vision_model != self.model:
            logging.info(f"Using text model: {self.model}, vision model: {self.vision_model}")
        else:
            logging.info(f"Using model: {self.model} (for both text and vision)")

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

    def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Encode an image file to base64 string for Ollama vision API."""
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                return base64_image
        except Exception as e:
            logging.error(f"Error encoding image to base64: {e}")
            return None
    
    def call_ollama_with_retry(self, prompt: str, images: Optional[List[Path]] = None) -> Optional[Dict]:
        """
        Make Ollama API call with exponential backoff retry and adaptive timeouts.
        
        Args:
            prompt: Text prompt to send to Ollama
            images: Optional list of image paths to include for vision analysis
        """
        # Select the appropriate model based on whether we're sending images
        model_to_use = self.vision_model if images else self.model
        if images and model_to_use != self.model:
            logging.info(f"Using vision model '{model_to_use}' for image analysis")
        
        for attempt in range(self.max_retries):
            current_read_timeout = self.read_timeout * (self.backoff_factor ** attempt)
            try:
                logging.info(f"Attempt {attempt + 1}/{self.max_retries} with {current_read_timeout}s timeout")
                
                # Build request payload
                payload = {
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature
                }
                
                # Add images if provided (for vision models)
                if images:
                    encoded_images = []
                    for img_path in images:
                        if img_path.exists():
                            base64_img = self._encode_image_to_base64(img_path)
                            if base64_img:
                                # Determine image format from extension
                                img_format = img_path.suffix.lower().lstrip('.')
                                if img_format in ['jpg', 'jpeg']:
                                    mime_type = 'image/jpeg'
                                elif img_format == 'png':
                                    mime_type = 'image/png'
                                elif img_format in ['heic', 'heif']:
                                    mime_type = 'image/heic'
                                else:
                                    mime_type = f'image/{img_format}'
                                
                                # Ollama vision API expects base64 strings directly
                                encoded_images.append(base64_img)
                            else:
                                logging.warning(f"Could not encode image: {img_path}")
                    
                    if encoded_images:
                        # Ollama vision API uses "images" field with base64 strings
                        payload["images"] = encoded_images
                        logging.info(f"Including {len(encoded_images)} image(s) for vision analysis")
                
                response = requests.post(
                    self.api_url,
                    json=payload,
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

    def process_content(self, extracted_content: dict, source_filename: str, source_type: str = "audio", source_file_path: Optional[Path] = None) -> Dict:
        """
        Process extracted content (from any source type) to generate tags and a title.
        
        Args:
            extracted_content: Dictionary with 'text', 'metadata', 'source_type', etc.
            source_filename: Original filename
            source_type: Type of content ('audio', 'text', 'image', etc.)
            
        Returns:
            Dictionary with processed content ready for note creation
        """
        try:
            logging.info(f"Starting Ollama processing for {source_type} content...")
            
            # Clean the text before processing
            cleaned_text = self.clean_text(extracted_content.get("text", ""))
            
            # Get the list of allowed tags
            allowed_tags = list(self.tag_manager._allowed_tags)
            allowed_tags_str = '\n'.join(allowed_tags)
            
            # Build metadata information based on source type
            metadata = []
            metadata_dict = extracted_content.get("metadata", {})
            
            # Add language information if available
            if extracted_content.get("language"):
                metadata.append(f"Language: {extracted_content['language']}")
            
            # Source-type specific metadata handling
            if source_type == "audio":
                # Process audio-specific metadata (segments, pauses, etc.)
                raw_data = extracted_content.get("raw_transcription_data", {})
                segments_info = []
                low_confidence_segments = []
                
                for segment in raw_data.get("segments", []):
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
                    
                    # Track low confidence segments
                    if segment_info["confidence"] < -1.0 and segment_info["no_speech_prob"] < 0.5:
                        low_confidence_segments.append(segment_info)
                
                # Convert segments info to readable format
                for info in segments_info:
                    if info["type"] == "pause":
                        metadata.append(f"[Pause: {info['duration']}s at {info['position']}s]")
                    elif info["type"] == "segment":
                        if info["no_speech_prob"] > 0.5:
                            metadata.append(f"[Non-speech section: {info['start']}-{info['end']}s]")
                        elif info["confidence"] < -1.0:
                            metadata.append(f"[Low confidence section: {info['start']}-{info['end']}s, confidence: {info['confidence']:.2f}]")
                
                if low_confidence_segments:
                    metadata.append(f"[Note: {len(low_confidence_segments)} low-confidence segments detected - review for accuracy]")
            
            elif source_type == "text":
                # Add text-specific metadata
                if metadata_dict.get("format"):
                    metadata.append(f"Format: {metadata_dict['format']}")
                if metadata_dict.get("word_count"):
                    metadata.append(f"Word count: {metadata_dict['word_count']}")
                if metadata_dict.get("line_count"):
                    metadata.append(f"Line count: {metadata_dict['line_count']}")
            
            elif source_type == "image":
                # Add image-specific metadata
                if metadata_dict.get("width") and metadata_dict.get("height"):
                    metadata.append(f"Image dimensions: {metadata_dict['width']}x{metadata_dict['height']}")
                if metadata_dict.get("format"):
                    metadata.append(f"Image format: {metadata_dict['format']}")
                if metadata_dict.get("ocr_success") is not None:
                    if metadata_dict.get("ocr_success"):
                        metadata.append("OCR: Text successfully extracted")
                    else:
                        metadata.append("OCR: No text found or OCR unavailable")
                if metadata_dict.get("text_length"):
                    metadata.append(f"Extracted text length: {metadata_dict['text_length']} characters")
            
            elif source_type == "folder":
                # Add folder-specific metadata
                if metadata_dict.get("file_count"):
                    metadata.append(f"Files in folder: {metadata_dict['file_count']}")
                if metadata_dict.get("folder_name"):
                    metadata.append(f"Folder name: {metadata_dict['folder_name']}")
                if metadata_dict.get("files"):
                    file_types = {}
                    for file_info in metadata_dict.get("files", []):
                        file_type = file_info.get("type", "unknown")
                        file_types[file_type] = file_types.get(file_type, 0) + 1
                    if file_types:
                        type_summary = ", ".join([f"{count} {ftype}" for ftype, count in file_types.items()])
                        metadata.append(f"File types: {type_summary}")
            
            metadata_str = "\n".join(metadata) if metadata else "None"
            
            # Build source-type specific prompt
            if source_type == "audio":
                content_description = "transcription from an audio note"
                formatting_guidelines = """1. FORMATTING GUIDELINES:
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
   - Maintain the speaker's original meaning and intent"""
            elif source_type == "text":
                content_description = "text content from a file"
                formatting_guidelines = """1. FORMATTING GUIDELINES:
   - Preserve the original structure and formatting if it's well-formatted
   - If the text is unformatted, add proper markdown formatting:
     * Use headers (# for main sections, ## for subsections) for topic changes
     * Use bullet points (- or *) for lists
     * Use **bold** for emphasis
     * Add paragraph breaks (double line breaks) between distinct thoughts
   - Maintain the original meaning and intent
   - DO NOT use code blocks (```) unless the content is actual code"""
            elif source_type == "image":
                # Check if OCR was successful or if we need to use vision to describe the image
                ocr_success = metadata_dict.get("ocr_success", False)
                has_text = bool(cleaned_text and len(cleaned_text.strip()) > 50 and "no text" not in cleaned_text.lower() and "image file" not in cleaned_text.lower() and "please describe" not in cleaned_text.lower())
                
                if has_text and ocr_success:
                    content_description = "text extracted from an image via OCR"
                    formatting_guidelines = """1. FORMATTING GUIDELINES:
   - Format the OCR-extracted text as well-structured markdown
   - If the text appears to be structured (lists, headings, etc.), preserve that structure
   - Use headers (# for main sections, ## for subsections) if the text has clear sections
   - Use bullet points (- or *) for lists
   - Add paragraph breaks (double line breaks) between distinct thoughts
   - If the OCR text is unclear or fragmented, organize it into coherent paragraphs
   - Note any OCR errors or unclear text with [unclear] markers if needed"""
                else:
                    # No OCR text - will use vision API to analyze the image
                    is_heic_unsupported = metadata_dict.get("heic_unsupported", False)
                    if is_heic_unsupported:
                        content_description = "a HEIC/HEIF image file (image will be analyzed with vision AI)"
                        formatting_guidelines = """1. FORMATTING GUIDELINES:
   - Analyze the provided image and describe what you see
   - Use headers (# for main sections, ## for subsections) to organize your description
   - Use bullet points (- or *) for key features or elements you observe
   - Add paragraph breaks (double line breaks) between distinct thoughts
   - Be descriptive about the content, objects, text, people, or scenes visible in the image
   - If there's text in the image, transcribe it accurately"""
                    else:
                        content_description = "an image file (image will be analyzed with vision AI)"
                        formatting_guidelines = """1. FORMATTING GUIDELINES:
   - Analyze the provided image and describe what you see in detail
   - Use headers (# for main sections, ## for subsections) to organize your description
   - Use bullet points (- or *) for key features, objects, or elements you observe
   - Add paragraph breaks (double line breaks) between distinct thoughts
   - Be descriptive about the content, objects, text, people, scenes, or any notable details visible in the image
   - If there's text in the image, transcribe it accurately
   - Describe the overall context and any important information conveyed by the image"""
            elif source_type == "folder":
                content_description = "combined content from multiple files in a folder"
                formatting_guidelines = """1. FORMATTING GUIDELINES:
   - The content is from multiple files (audio transcriptions, text files, image OCR, etc.)
   - Organize the content logically, grouping related information together
   - Use headers (# for main sections, ## for subsections) to separate content from different files or topics
   - Preserve the structure from individual files where appropriate
   - Use bullet points (- or *) for lists
   - Add paragraph breaks (double line breaks) between distinct thoughts
   - Create a cohesive narrative that ties together content from all files in the folder"""
            else:
                content_description = f"content from a {source_type} file"
                formatting_guidelines = """1. FORMATTING GUIDELINES:
   - Format the content as well-structured markdown
   - Use headers (# for main sections, ## for subsections) for topic changes
   - Use bullet points (- or *) for lists
   - Use **bold** for emphasis
   - Add paragraph breaks (double line breaks) between distinct thoughts
   - Ensure proper capitalization and punctuation"""
            
            prompt = f"""Given this {content_description} with metadata and the list of ALLOWED TAGS below:

METADATA:
{metadata_str}

Your task is to transform this content into a well-formatted markdown note:

{formatting_guidelines}

3. TITLE GENERATION:
   - Create a clear, descriptive filename (without extension)
   - Use lowercase with hyphens (e.g., "meeting-notes-project-discussion")
   - Keep it concise but informative (30-50 characters ideal)

4. TAG SELECTION:
   - Select ONLY from the ALLOWED TAGS list below
   - Choose 2-5 most relevant tags that best describe the content
   - If no tags are relevant, return an empty list

5. SUMMARY GENERATION:
   - Create a brief 1-2 sentence summary of the content
   - This summary will be displayed at the top of the note
   - Focus on the main topic or key points
   - Keep it concise and informative

IMPORTANT: Your response must use YAML frontmatter format (standard markdown with YAML metadata block).

ALLOWED TAGS:
{allowed_tags_str}

Original filename: {source_filename}
Content: {cleaned_text}

You must ONLY use tags from the ALLOWED TAGS list above. Do not create new tags.
If no tags from the allowed list are relevant, use an empty list: []

Respond in this exact format (YAML frontmatter followed by markdown content):
---
title: clear-descriptive-filename
tags: ["#tag1", "#tag2"]
summary: "Brief 1-2 sentence summary of the content"
---

The formatted content with proper markdown formatting goes here.
ONLY include the formatted content - do NOT add summaries or explanations in the body."""

            # Determine if we need to include images for vision analysis
            images_to_send = None
            if source_type == "image" and source_file_path:
                # Check if OCR failed or found no text - if so, use vision
                ocr_success = metadata_dict.get("ocr_success", False)
                has_text = bool(cleaned_text and len(cleaned_text.strip()) > 50 and "no text" not in cleaned_text.lower() and "image file" not in cleaned_text.lower() and "please describe" not in cleaned_text.lower())
                
                if not (has_text and ocr_success):
                    # OCR failed or found no text - use vision to analyze the image
                    if source_file_path.exists():
                        images_to_send = [source_file_path]
                        logging.info(f"Using vision API to analyze image: {source_file_path.name}")
            
            elif source_type == "folder" and source_file_path:
                # For folders, collect all image files that need vision analysis
                images_to_send = []
                folder_path = source_file_path
                if folder_path.exists() and folder_path.is_dir():
                    for file_info in metadata_dict.get("files", []):
                        if file_info.get("type") == "image":
                            file_meta = file_info.get("metadata", {})
                            # Check if this image needs vision (no OCR text)
                            if not file_meta.get("ocr_success", False):
                                # Find the image file in the folder
                                img_filename = file_info.get("filename")
                                if img_filename:
                                    img_path = folder_path / img_filename
                                    if img_path.exists():
                                        images_to_send.append(img_path)
                                        logging.info(f"Including image for vision analysis: {img_filename}")
                
                if not images_to_send:
                    images_to_send = None

            response = self.call_ollama_with_retry(prompt, images=images_to_send)
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
                
                # Add source type and original metadata
                result["source_type"] = source_type
                result["original_metadata"] = metadata_dict
                
                return result
                
            except (yaml.YAMLError, ValueError) as e:
                logging.error(f"YAML frontmatter parse error: {str(e)}")
                logging.error(f"Problematic response: {raw_response[:1000]}...")  # Log first 1000 chars
                raise ValueError(f"Invalid YAML frontmatter in Ollama response: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error processing content: {str(e)}")
            raise

    def process_transcription(self, transcription_data: dict, audio_filename: str) -> Dict:
        """Process the transcription to generate tags and a title. (Legacy method for backward compatibility)"""
        # Convert old format to new format
        extracted_content = {
            'text': transcription_data.get('text', ''),
            'metadata': {
                'language': transcription_data.get('language'),
                'segments': transcription_data.get('segments', []),
            },
            'source_type': 'audio',
            'language': transcription_data.get('language'),
            'raw_transcription_data': transcription_data,
        }
        return self.process_content(extracted_content, audio_filename, source_type='audio')
    
