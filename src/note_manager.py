import os
from datetime import datetime
from pathlib import Path
from typing import Dict
import re
import logging
import shutil

class NoteManager:
    def __init__(self):
        self.vault_path = Path(os.getenv('OBSIDIAN_VAULT_PATH'))
        self.notes_folder = self.vault_path / os.getenv('NOTES_FOLDER', 'notes')
        self.audio_folder = self.vault_path / os.getenv('NOTES_FOLDER', 'notes') / "audio"
        self.notes_folder.mkdir(parents=True, exist_ok=True)
        self.audio_folder.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove invalid Windows characters
        """
        # Remove invalid Windows filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # Trim spaces from ends
        return sanitized.strip()

    def _format_title_for_filename(self, title: str) -> str:
        """
        Format title for use in filename: replace hyphens with spaces and capitalize words
        Example: "frustration-at-work" -> "Frustration at Work"
        """
        # First sanitize to remove invalid characters
        sanitized = self._sanitize_filename(title)
        # Replace hyphens and underscores with spaces
        sanitized = re.sub(r'[-_]', ' ', sanitized)
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # Capitalize each word (Title Case)
        words = sanitized.split()
        capitalized_words = [word.capitalize() for word in words]
        return ' '.join(capitalized_words).strip()

    def _extract_datetime_from_filename(self, filename: Path) -> tuple:
        """
        Extract date and time from filename if it starts with a date pattern YYYY-MM-DD
        Returns tuple of (date_str, time_str) or (None, None) if not found
        """
        try:
            # Extract date and time pattern from filename
            datetime_match = re.match(r'(\d{4}-\d{2}-\d{2})[-_](\d{2}[-_]\d{2}(?:AM|PM)?)', filename.stem, re.IGNORECASE)
            if datetime_match:
                return datetime_match.group(1), datetime_match.group(2)
            
            # If only date is found
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', filename.stem)
            if date_match:
                return date_match.group(1), None
                
            return None, None
        except Exception as e:
            logging.error(f"Error extracting datetime from filename: {str(e)}")
            return None, None

    def _get_audio_folder(self, date_str: str = None) -> Path:
        """
        Get or create a dated audio folder for storing audio files
        Args:
            date_str (str): Optional date string in YYYY-MM-DD format. If None, uses current date.
        Returns:
            Path: Path to the audio folder
        """
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
        audio_folder = self.audio_folder / date_str
        audio_folder.mkdir(parents=True, exist_ok=True)
        return audio_folder

    def create_note(self, processed_content: Dict, source_file: Path) -> Dict:
        """
        Create a markdown note in the Obsidian vault
        
        Args:
            processed_content (Dict): Processed content from Ollama
            source_file (Path): Path to the original source file (audio, text, image, etc.)
            
        Returns:
            Dict with 'note_path' and attachment paths
        """
        # Get source type
        source_type = processed_content.get('source_type', 'audio')
        
        # Get source date from processed content or extract from filename
        source_date = processed_content.get('source_date')
        source_time = processed_content.get('source_time')
        
        if not source_date:
            source_date, source_time = self._extract_datetime_from_filename(source_file)
        
        # Create new filename with source date/time or current time
        if source_date:
            if source_time:
                date_time = f"{source_date}_{source_time}"
            else:
                now = datetime.now()
                date_time = f"{source_date}_{now.strftime('%I-%M%p')}"
        else:
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d_%I-%M%p")
            
        title = processed_content.get('title', 'Untitled Note')
        # Format title for filename: capitalize words and use spaces instead of hyphens
        formatted_title = self._format_title_for_filename(title[:30])
        
        # Handle attachments based on source type
        attachment_paths = []
        
        if source_type == 'folder':
            # For folders, move the entire folder to attachments
            attachments_folder = self.vault_path / os.getenv('NOTES_FOLDER', 'notes') / "attachments" / (source_date or datetime.now().strftime("%Y-%m-%d"))
            attachments_folder.mkdir(parents=True, exist_ok=True)
            
            new_folder_name = f"{date_time}_{formatted_title}"
            new_folder_path = attachments_folder / new_folder_name
            
            # Handle case where folder already exists
            if new_folder_path.exists():
                counter = 1
                while new_folder_path.exists():
                    new_folder_path = attachments_folder / f"{new_folder_name}_{counter}"
                    counter += 1
            
            # Move entire folder to attachments
            if source_file.parent != attachments_folder:
                logging.info(f"Moving folder from {source_file} to {new_folder_path}")
                try:
                    # Copy entire folder
                    shutil.copytree(str(source_file), str(new_folder_path))
                    
                    # Verify copy was successful (check if folder exists and has files)
                    if new_folder_path.exists() and any(new_folder_path.iterdir()):
                        # Remove original folder
                        shutil.rmtree(str(source_file))
                        logging.info("Folder moved successfully")
                    else:
                        raise Exception("Folder copy verification failed")
                except Exception as e:
                    logging.error(f"Failed to move folder: {str(e)}")
                    raise Exception(f"Failed to move folder: {str(e)}")
            
            # Add all files in the folder as attachments
            for file_path in new_folder_path.iterdir():
                if file_path.is_file():
                    attachment_paths.append(file_path)
        
        elif source_type == 'audio':
            # Audio files go to audio folder
            audio_folder = self._get_audio_folder(source_date)
            new_audio_name = f"{date_time}_{formatted_title}{source_file.suffix}"
            new_audio_path = audio_folder / new_audio_name
            
            # Move audio file to audio folder if it's not already there
            if source_file.parent != audio_folder:
                new_audio_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle case where file already exists
                if new_audio_path.exists():
                    base = new_audio_path.stem
                    suffix = new_audio_path.suffix
                    counter = 1
                    while new_audio_path.exists():
                        new_audio_path = audio_folder / f"{base}_{counter}{suffix}"
                        counter += 1
                
                logging.info(f"Moving audio file from {source_file} to {new_audio_path}")
                
                try:
                    # Copy file first
                    shutil.copy2(str(source_file), str(new_audio_path))
                    
                    # Verify copy was successful
                    if new_audio_path.exists() and new_audio_path.stat().st_size == source_file.stat().st_size:
                        # Only delete original after successful copy
                        source_file.unlink()
                        logging.info("Audio file moved successfully")
                    else:
                        raise Exception("File copy verification failed")
                    
                except Exception as e:
                    logging.error(f"Failed to move audio file: {str(e)}")
                    raise Exception(f"Failed to move audio file: {str(e)}")
            
            attachment_paths.append(new_audio_path)
        else:
            # For other file types, create an attachments folder
            attachments_folder = self.vault_path / os.getenv('NOTES_FOLDER', 'notes') / "attachments" / (source_date or datetime.now().strftime("%Y-%m-%d"))
            attachments_folder.mkdir(parents=True, exist_ok=True)
            
            new_file_name = f"{date_time}_{formatted_title}{source_file.suffix}"
            new_file_path = attachments_folder / new_file_name
            
            # Handle case where file already exists
            if new_file_path.exists():
                base = new_file_path.stem
                suffix = new_file_path.suffix
                counter = 1
                while new_file_path.exists():
                    new_file_path = attachments_folder / f"{base}_{counter}{suffix}"
                    counter += 1
            
            # Move file to attachments folder
            if source_file.parent != attachments_folder:
                logging.info(f"Moving {source_type} file from {source_file} to {new_file_path}")
                try:
                    shutil.copy2(str(source_file), str(new_file_path))
                    if new_file_path.exists() and new_file_path.stat().st_size == source_file.stat().st_size:
                        source_file.unlink()
                        logging.info(f"{source_type.capitalize()} file moved successfully")
                    else:
                        raise Exception("File copy verification failed")
                except Exception as e:
                    logging.error(f"Failed to move {source_type} file: {str(e)}")
                    raise Exception(f"Failed to move {source_type} file: {str(e)}")
            
            attachment_paths.append(new_file_path)
        
        # Create the note with matching naming convention
        note_filename = f"{date_time}_{formatted_title}.md"
        note_path = self.notes_folder / note_filename
        
        # Build note content with proper formatting
        note_parts = []
        
        # Add title
        note_parts.append(f"# {title}")
        note_parts.append("")  # Blank line after title
        
        # Add attachments based on source type
        if source_type == 'folder':
            # For folders, list all files in the folder
            note_parts.append("## Attachments")
            note_parts.append("")
            for attachment_path in attachment_paths:
                rel_path = os.path.relpath(attachment_path, self.vault_path)
                rel_path = rel_path.replace('\\', '/')  # Convert Windows path to forward slashes
                
                # Determine how to embed based on file type
                if attachment_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                    # Image embed
                    note_parts.append(f"![[{rel_path}]]")
                elif attachment_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']:
                    # Audio player
                    note_parts.append(f"![[{rel_path}]]")
                elif attachment_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                    # Video embed
                    note_parts.append(f"![[{rel_path}]]")
                else:
                    # Link to other files
                    note_parts.append(f"- [[{rel_path}|{attachment_path.name}]]")
            note_parts.append("")  # Blank line after attachments
        else:
            # For single files
            for attachment_path in attachment_paths:
                rel_path = os.path.relpath(attachment_path, self.vault_path)
                rel_path = rel_path.replace('\\', '/')  # Convert Windows path to forward slashes
                
                if source_type == 'audio':
                    # Audio player
                    note_parts.append(f"![[{rel_path}]]")
                elif source_type in ['image', 'video']:
                    # Image or video embed
                    note_parts.append(f"![[{rel_path}]]")
                else:
                    # For text files, just link to the file
                    note_parts.append(f"[[{rel_path}|Original {source_type} file]]")
                note_parts.append("")  # Blank line after attachment
        
        # Add tags if present
        if processed_content.get('tags'):
            # Strip any existing hashtags and add a single one
            formatted_tags = [tag.lstrip('#') for tag in processed_content.get('tags', [])]
            note_parts.append(' '.join(['#' + tag for tag in formatted_tags]))
            note_parts.append("")  # Blank line after tags
        
        # Always add AI summary section
        note_parts.append("## AI Summary")
        note_parts.append("")
        ai_summary = processed_content.get('ai_summary', '').strip()
        if ai_summary:
            note_parts.append(ai_summary)
        else:
            note_parts.append("*No summary available*")
        note_parts.append("")  # Blank line after summary
        
        # Add metadata section if there's interesting metadata
        metadata_dict = processed_content.get('original_metadata', {})
        if any(key in processed_content for key in ['language', 'confidence_issues', 'non_speech_sections']) or metadata_dict:
            note_parts.append("## Metadata")
            if processed_content.get('language'):
                note_parts.append(f"- Language: {processed_content['language']}")
            if processed_content.get('confidence_issues'):
                note_parts.append("- Low confidence sections noted in content with [uncertain] tags")
            if processed_content.get('non_speech_sections'):
                note_parts.append("- Contains non-speech sections (marked in content)")
            if metadata_dict.get('format'):
                note_parts.append(f"- Format: {metadata_dict['format']}")
            if metadata_dict.get('word_count'):
                note_parts.append(f"- Word count: {metadata_dict['word_count']}")
            note_parts.append("")  # Blank line after metadata
        
        # Add formatted content (should already have proper paragraph breaks from Ollama)
        formatted_content = processed_content.get('formatted_content', '').strip()
        if formatted_content:
            note_parts.append(formatted_content)
        
        # Join all parts with newlines (preserves paragraph breaks in formatted_content)
        final_content = '\n'.join(note_parts)
        
        # Write the note
        try:
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            logging.info(f"Note created successfully at {note_path}")
            
            # Return paths for state tracking
            result = {
                'note_path': note_path,
            }
            
            # Add attachment path(s) - use 'audio_path' for backward compatibility
            if source_type == 'audio' and attachment_paths:
                result['audio_path'] = attachment_paths[0]
            elif attachment_paths:
                result['attachment_path'] = attachment_paths[0]
            
            return result
        except Exception as e:
            logging.error(f"Failed to create note: {str(e)}")
            raise Exception(f"Failed to create note: {str(e)}")
