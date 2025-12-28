import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from core.agent_base import BaseAgent
from .processors.content_processor_manager import ContentProcessorManager
from .processors.note_manager import NoteManager
from core.event_bus import EventBus

class EchoEtcherAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.capabilities = [
            {
                "name": "configure", 
                "description": "Update watch_path and dest_path for file monitoring (audio/images).",
                "schema": {
                    "type": "object",
                    "properties": {
                        "watch_path": {"type": "string", "description": "Absolute path to folder to watch for new files."},
                        "dest_path": {"type": "string", "description": "Absolute path to Obsidian vault folder for notes."},
                        "polling_enabled": {"type": "boolean", "description": "Enable checking the system database for manual commands from the Orchestrator or Radar."},
                        "polling_interval": {"type": "integer", "description": "How often (in seconds) the agent check for new manual tasks assigned to it.", "minimum": 10},
                        "scan_enabled": {"type": "boolean", "description": "Enable proactive disk scanning to find new files automatically."},
                        "scan_interval": {"type": "integer", "description": "How often (in seconds) the agent sweeps the watch_path for new files on its own.", "minimum": 30}
                    },
                    "required": ["watch_path", "dest_path"]
                }
            },
            {
                "name": "scan", 
                "description": "Trigger a manual scan of the watched folder to process new files.",
                "schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        self.observer = None
        self.watch_path = None
        self.dest_path = None
        self.processor_manager = None
        self.note_manager = None
        self.logs = [] # In-memory logs for UI
        self._last_error = None
        
        # Scan configuration
        self.scan_enabled = False
        self.scan_interval = 120

    def _sanitize_path(self, path: str) -> str:
        """Strip whitespace and surrounding quotes from path string."""
        if not path:
            return None
        s = path.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        return s

    async def load_config(self):
        """
        Load configuration from memory on startup.
        Overrides base class to also load watch_path, dest_path, and scan settings.
        """
        # Load base polling config
        await super().load_config()
        
        # Load watch and destination paths
        raw_watch = await self.get_memory("watch_path")
        if raw_watch:
            self.watch_path = os.path.expanduser(raw_watch)
        
        raw_dest = await self.get_memory("dest_path")
        if raw_dest:
            self.dest_path = os.path.expanduser(raw_dest)
        
        # Load scan configuration
        raw_scan_enabled = await self.get_memory("scan_enabled")
        if raw_scan_enabled is not None:
            self.scan_enabled = str(raw_scan_enabled).lower() == 'true'
        
        raw_scan_interval = await self.get_memory("scan_interval")
        if raw_scan_interval is not None:
            self.scan_interval = int(raw_scan_interval)

    async def process_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming tasks from Orchestrator or Router.
        """
        action = payload.get("action")

        if action == "configure":
            watch_path = payload.get("watch_path")
            dest_path = payload.get("dest_path")
            
            if watch_path:
                watch_path = os.path.expanduser(watch_path)
                await self.save_memory("watch_path", watch_path)
                self.watch_path = watch_path
                
            if dest_path:
                dest_path = os.path.expanduser(dest_path)
                await self.save_memory("dest_path", dest_path)
                self.dest_path = dest_path
            
            # Polling configuration
            polling_enabled = payload.get("polling_enabled", self.polling_enabled)
            polling_interval = payload.get("polling_interval", self.polling_interval)
            await self.update_polling_config(polling_enabled, polling_interval)
            
            # Scan configuration
            self.scan_enabled = payload.get("scan_enabled", self.scan_enabled)
            self.scan_interval = payload.get("scan_interval", self.scan_interval)
            await self.save_memory("scan_enabled", str(self.scan_enabled))
            await self.save_memory("scan_interval", str(self.scan_interval))
            
            # Restart if config changed
            if watch_path or dest_path or "scan_enabled" in payload:
                if self.scan_enabled:
                    await self.start_watching()
                else:
                    await self.stop_watching()
                
            return {"status": "success", "message": "Configuration updated"}

        if action == "scan":
            await self.scan_folder()
            return {"status": "success", "message": "Scan triggered"}

        return {"status": "error", "message": "Unknown action"}
        
    async def run(self):
        """
        Main loop. Runs watchdog observer + periodic scheduled scans.
        """
        last_scan = 0
        
        # Ensure we have config
        if not self.watch_path:
            raw_watch = await self.get_memory("watch_path")
            self.watch_path = os.path.expanduser(raw_watch) if raw_watch else None
        if not self.dest_path:
            raw_dest = await self.get_memory("dest_path")
            self.dest_path = os.path.expanduser(raw_dest) if raw_dest else None
            
        # Load scan config
        raw_scan_enabled = await self.get_memory("scan_enabled")
        self.scan_enabled = str(raw_scan_enabled).lower() == 'true' if raw_scan_enabled is not None else False
        
        raw_scan_interval = await self.get_memory("scan_interval")
        self.scan_interval = int(raw_scan_interval) if raw_scan_interval is not None else 120
            
        # Initialize components if paths are set and scanning is enabled
        if self.watch_path and self.dest_path and self.scan_enabled and not self.observer:
            await self.start_watching()
            # Initial scan on startup
            await self.scan_folder()
            last_scan = time.time()
        
        # Periodic scan loop
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            if not self.scan_enabled:
                continue
                
            current_time = time.time()
            if current_time - last_scan >= self.scan_interval:
                if self.watch_path and self.dest_path:
                    await self.log_to_ui(f"Running scheduled scan...")
                    await self.scan_folder()
                    last_scan = current_time
            
    async def start_watching(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            
        raw_watch = await self.get_memory("watch_path")
        self.watch_path = self._sanitize_path(os.path.expanduser(raw_watch)) if raw_watch else None
        
        raw_dest = await self.get_memory("dest_path")
        self.dest_path = self._sanitize_path(os.path.expanduser(raw_dest)) if raw_dest else None
        
        if not self.watch_path or not os.path.exists(self.watch_path):
            await self.log_to_ui(f"Invalid watch path: {repr(self.watch_path)}")
            return

        # Initialize managers
        try:
            vault_path = str(Path(self.dest_path).parent)
            folder_name = Path(self.dest_path).name
            
            self.processor_manager = ContentProcessorManager(self) # Pass self to give access to LLM
            self.note_manager = NoteManager(vault_path=vault_path, notes_folder=folder_name)
            
            event_handler = Handler(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, self.watch_path, recursive=False)
            self.observer.start()
            
            await self.log_to_ui(f"Started watching: {self.watch_path}")
        except Exception as e:
            await self.log_to_ui(f"Error starting observer: {e}")

    async def stop_watching(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            await self.log_to_ui("Stopped watching.")
    
    async def cleanup(self):
        """
        Graceful shutdown: stop file watcher and clean up resources.
        """
        await self.stop_watching()
        self.processor_manager = None
        self.note_manager = None

    async def log_to_ui(self, message: str):
        """Log to local list for UI consumption"""
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        self.logs.append(log_msg)
        # Keep only last 100 logs
        if len(self.logs) > 100:
            self.logs.pop(0)
        # Also log to standard logger
        await self.log(message)

    async def scan_folder(self):
        """Manually scans the watched folder for content."""
        # Ensure paths are loaded from memory if not set
        if not self.watch_path:
            raw_watch = await self.get_memory("watch_path")
            self.watch_path = self._sanitize_path(os.path.expanduser(raw_watch)) if raw_watch else None
        if not self.dest_path:
            raw_dest = await self.get_memory("dest_path")
            self.dest_path = self._sanitize_path(os.path.expanduser(raw_dest)) if raw_dest else None
            
        if not self.watch_path:
             await self.log_to_ui(f"Cannot scan: Watch path not set")
             return
             
        # Debug diagnostics for path issues
        watch_path_obj = Path(self.watch_path)
        await self.log_to_ui(f"DEBUG: Checking path: {self.watch_path}")
        await self.log_to_ui(f"DEBUG: os.path.exists = {os.path.exists(self.watch_path)}")
        await self.log_to_ui(f"DEBUG: Path.exists() = {watch_path_obj.exists()}")
        await self.log_to_ui(f"DEBUG: Path.is_dir() = {watch_path_obj.is_dir()}")
        
        # Check parent directories to find where the path breaks
        parent = watch_path_obj
        while parent != parent.parent:
            if parent.exists():
                await self.log_to_ui(f"DEBUG: EXISTS: {parent}")
                break
            else:
                await self.log_to_ui(f"DEBUG: MISSING: {parent}")
            parent = parent.parent
             
        if not watch_path_obj.is_dir():
             await self.log_to_ui(f"Cannot scan: Watch path invalid: {repr(self.watch_path)}")
             return

        await self.log_to_ui(f"Scanning folder: {self.watch_path}")
        await EventBus.publish("echo_etcher:scan_started", {"sender": self.name, "path": self.watch_path})
        
        # We need to initialize components if not running
        if not self.processor_manager:
             self.processor_manager = ContentProcessorManager(self)
             
             if self.dest_path:
                 vault_path = str(Path(self.dest_path).parent)
                 folder_name = Path(self.dest_path).name
                 self.note_manager = NoteManager(vault_path=vault_path, notes_folder=folder_name)
             else:
                 self.note_manager = NoteManager()

        try:
            count = 0
            for entry in os.scandir(self.watch_path):
                if entry.is_file() and not entry.name.startswith('.'):
                    await self.process_file(Path(entry.path))
                    count += 1
            await self.log_to_ui(f"Scan complete. Processed {count} items.")
        except Exception as e:
            await self.log_to_ui(f"Error during scan: {e}")

    async def process_file(self, file_path: Path):
        await self.log_to_ui(f"Processing detected file: {file_path.name}")
        await EventBus.publish("echo_etcher:file_detected", {"sender": self.name, "file": file_path.name})
        try:
            if not self.processor_manager:
                await self.log_to_ui("Processor manager not initialized")
                return

            if not self.processor_manager.can_process(file_path):
                await self.log_to_ui(f"Skipping unsupported file: {file_path.name}")
                return

            # Get processor
            processor = self.processor_manager.get_processor(file_path)
            content = processor.extract_content(file_path)
            
            # DEBUG: Log raw extracted content
            raw_text_len = len(content.get('text', ''))
            await self.log_to_ui(f"DEBUG: Raw extracted text length: {raw_text_len} chars")
            if raw_text_len > 0:
                preview = content.get('text', '')[:200]
                await self.log_to_ui(f"DEBUG: Text preview: {preview}...")
            else:
                await self.log_to_ui(f"DEBUG: WARNING - No text extracted from file!")
            
            # Use LLM to format note (using the agent's LLM client)
            processed_content = await self.processor_manager.enrich_with_llm(content, file_path)
            
            # DEBUG: Log enriched content
            await self.log_to_ui(f"DEBUG: After LLM - title: {processed_content.get('title')}")
            await self.log_to_ui(f"DEBUG: After LLM - tags: {processed_content.get('tags')}")
            await self.log_to_ui(f"DEBUG: After LLM - summary length: {len(processed_content.get('ai_summary', ''))}")
            await self.log_to_ui(f"DEBUG: After LLM - formatted_content length: {len(processed_content.get('formatted_content', ''))}")
            
            # Extract recording date from metadata if available
            recording_date = None
            metadata = processed_content.get('metadata', {})
            if 'recording_date' in metadata:
                recording_date = metadata['recording_date']
                await self.log_to_ui(f"Found recording date: {recording_date}")
            else:
                await self.log_to_ui("No recording date found in metadata, will use processing date")
            
            # Create note
            result = self.note_manager.create_note(processed_content, file_path, recording_date=recording_date)
            await self.log_to_ui(f"Note created: {result.get('note_path')}")
            await EventBus.publish("echo_etcher:note_created", {
                "sender": self.name, 
                "file": file_path.name, 
                "note": result.get('note_path')
            })
            
        except Exception as e:
            import traceback
            await self.log_to_ui(f"Error processing {file_path.name}: {e}")
            await self.log_to_ui(f"DEBUG: Traceback: {traceback.format_exc()[:500]}")

class Handler(FileSystemEventHandler):
    """
    File system event handler for EchoEtcher file watching.
    Dispatches file creation events to the agent for processing.
    """
    def __init__(self, agent: 'EchoEtcherAgent'):
        """
        Initialize the handler with a reference to the agent.
        
        Args:
            agent: The EchoEtcherAgent instance to dispatch events to
        """
        self.agent = agent

    def on_created(self, event) -> None:
        """
        Handle file creation events from the file system watcher.
        
        Args:
            event: File system event from watchdog
        """
        if not event.is_directory:
            import asyncio
            # Build a safe way to run async from sync callback
            # For this MVP, we might just fire and forget or use a loop helper
            # Correct way in BaseAgent context isn't strictly defined for sync callbacks
            # We will try to get the running loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.agent.process_file(Path(event.src_path)))
            except RuntimeError:
                # No running loop, try to get event loop (for compatibility)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.agent.process_file(Path(event.src_path)))
                except Exception as e:
                    print(f"Error dispatching file event: {e}")
            except Exception as e:
                print(f"Error dispatching file event: {e}")

