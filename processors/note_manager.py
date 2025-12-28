import os
import shutil
import logging
import re
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class NoteManager:
    """
    Manages creation of notes in the Obsidian vault.
    Now relies on environment variables set by the Agent before use, 
    or we could strictly start passing paths in methods. 
    For compatibility with original logic, we assume env vars or we adapt.
    Actually, let's adapt to use instance variables if possible, but for MVP speed mapping strictly to original is easier if we set env vars in Agent.
    """
    def __init__(self, vault_path: str = None, notes_folder: str = None):
        self.vault_path = Path(vault_path) if vault_path else Path('/tmp')
        self.notes_folder_name = notes_folder if notes_folder else 'notes'

    def _get_paths(self):
        # Use instance vars
        vault_path = self.vault_path
        notes_folder = vault_path / self.notes_folder_name
        audio_folder = notes_folder / "audio"
        attachments_folder = notes_folder / "attachments"
        
        try:
            notes_folder.mkdir(parents=True, exist_ok=True)
            audio_folder.mkdir(parents=True, exist_ok=True)
            attachments_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directories at {notes_folder}: {e}")
            # Fallback to tmp if permission error? 
            # For now just log and let it fail or it might crash
        
        return vault_path, notes_folder, audio_folder, attachments_folder

    def create_note(self, processed_content: Dict, source_file: Path, recording_date: Optional[datetime] = None) -> Dict:
        """
        Create a note from processed content.
        
        Args:
            processed_content: Dictionary containing processed content with title, tags, etc.
            source_file: Path to the source file being processed
            recording_date: Optional datetime object representing when the audio was recorded.
                          If provided, this date will be used instead of the current date.
        """
        vault_path, notes_folder, audio_folder, attachments_folder = self._get_paths()
        
        source_type = processed_content.get('source_type', 'unknown')
        title = processed_content.get('title', 'Untitled Note')
        
        # Use recording date if available, otherwise use current date
        if recording_date:
            date_str = recording_date.strftime("%Y-%m-%d")
            time_str = recording_date.strftime("%H-%M")
            logging.info(f"Using recording date for note: {date_str} {time_str}")
        else:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H-%M")
            logging.info(f"Using current date for note: {date_str} {time_str}")
        
        date_time = f"{date_str}_{time_str}"
        
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title).strip().replace(' ', '_')
        note_filename = f"{date_time}_{safe_title}.md"
        note_path = notes_folder / note_filename
        
        # Handle attachments (move source file)
        attachment_path = None
        if source_type == 'audio':
            dated_audio_folder = audio_folder / date_str
            dated_audio_folder.mkdir(exist_ok=True)
            new_name = f"{date_time}_{safe_title}{source_file.suffix}"
            dest_path = dated_audio_folder / new_name
            
            # User requested file to be "processed" (removed from watch folder)
            shutil.move(source_file, dest_path)
            attachment_path = dest_path
        elif source_type in ['image', 'text']:
            # Maybe move text files to attachments if we want to keep original?
            # Or just leave them.
            pass

        # Build Content
        content_lines = []
        content_lines.append(f"# {title}")
        content_lines.append(f"Tags: {', '.join(processed_content.get('tags', []))}")
        content_lines.append("")
        content_lines.append(f"**Summary**: {processed_content.get('ai_summary', 'No summary')}")
        content_lines.append("")
        
        if attachment_path:
            rel_path = attachment_path.relative_to(vault_path)
            content_lines.append(f"![[{rel_path}]]")
            content_lines.append("")
        
        # Include raw transcript if available (collapsed by default)
        raw_text = processed_content.get('text', '')
        if raw_text:
            content_lines.append("<details>")
            content_lines.append("<summary>Raw Transcript</summary>")
            content_lines.append("")
            content_lines.append(raw_text)
            content_lines.append("")
            content_lines.append("</details>")
            content_lines.append("")
        
        # Include LLM-formatted content
        formatted = processed_content.get('formatted_content', '')
        if formatted:
            content_lines.append("## Formatted Content")
            content_lines.append("")
            content_lines.append(formatted)
        
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
            
        logging.info(f"NoteManager saved note to: {note_path}")
        return {'note_path': str(note_path)}
