import os
import time
from pathlib import Path
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from logging.handlers import RotatingFileHandler
import requests
import subprocess
import platform
import signal
import sys
import gc
import json
import re
import argparse
from datetime import datetime

# Version information
__version__ = "1.0.0"

# Set up logging configuration
def setup_logging(verbose=False):
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    # Configure the rotating file handler
    log_file = log_dir / "talknote.log"
    max_bytes = 10 * 1024 * 1024  # 10MB per file
    backup_count = 5  # Keep 5 backup files
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to the handlers
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove any existing handlers and add our configured handlers
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Logging system initialized with rotation enabled")

def check_ffmpeg():
    """Check if ffmpeg and ffprobe are available in PATH."""
    tools = ['ffmpeg', 'ffprobe']
    missing_tools = []
    
    for tool in tools:
        found = False
        try:
            result = subprocess.run([tool, '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                # Extract version info
                version_line = result.stdout.split('\n')[0] if result.stdout else "available"
                logging.info(f"{tool.capitalize()} found: {version_line}")
                found = True
        except FileNotFoundError:
            pass
        except Exception as e:
            logging.warning(f"Error checking {tool}: {e}")
        
        if not found:
            # Try to find tool in common locations
            common_paths = [
                '/opt/homebrew/bin',  # Homebrew on Apple Silicon
                '/usr/local/bin',     # Homebrew on Intel Mac
                '/usr/bin',           # System location
            ]
            
            for base_path in common_paths:
                tool_path = Path(base_path) / tool
                if tool_path.exists():
                    logging.warning(f"{tool.capitalize()} found at {tool_path} but not in PATH. Adding to PATH.")
                    os.environ['PATH'] = str(Path(base_path)) + os.pathsep + os.environ.get('PATH', '')
                    found = True
                    break
        
        if not found:
            missing_tools.append(tool)
    
    if missing_tools:
        error_msg = (
            f"FFmpeg tools not found! The following tools are required: {', '.join(missing_tools)}\n\n"
            "Install FFmpeg (which includes both ffmpeg and ffprobe) with:\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: sudo apt install ffmpeg (or sudo yum install ffmpeg)\n"
            "  Windows: Download from https://ffmpeg.org/download.html\n\n"
            "After installing, restart the program."
        )
        raise ValueError(error_msg)

def validate_config():
    """Validate that all required configuration is present."""
    required_vars = {
        'WATCH_FOLDER': 'Path to folder to watch for audio files',
        'OBSIDIAN_VAULT_PATH': 'Path to Obsidian vault root directory',
        'OLLAMA_MODEL': 'Ollama model name to use for text processing (e.g., mistral)'
    }
    
    # Optional: Check if vision model is set (will fall back to main model if not)
    vision_model = os.getenv('OLLAMA_VISION_MODEL')
    if vision_model:
        logging.info(f"Vision model configured: {vision_model} (for image analysis)")
    else:
        logging.info("No OLLAMA_VISION_MODEL set - will use OLLAMA_MODEL for both text and vision")
    
    missing = []
    invalid_paths = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing.append(f"{var} ({description})")
        elif var in ['WATCH_FOLDER', 'OBSIDIAN_VAULT_PATH']:
            path = Path(value)
            if not path.exists():
                invalid_paths.append(f"{var}: {value} (path does not exist)")
    
    if missing:
        error_msg = "Missing required environment variables:\n  " + "\n  ".join(missing)
        error_msg += "\n\nPlease copy .env.example to .env and configure it."
        raise ValueError(error_msg)
    
    if invalid_paths:
        error_msg = "Invalid paths in configuration:\n  " + "\n  ".join(invalid_paths)
        raise ValueError(error_msg)
    
    # Check for ffmpeg
    check_ffmpeg()
    
    # Validate optional settings
    whisper_size = os.getenv('WHISPER_MODEL_SIZE', 'medium')
    valid_whisper_sizes = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    if whisper_size not in valid_whisper_sizes:
        logging.warning(f"Invalid WHISPER_MODEL_SIZE '{whisper_size}', using 'medium'. Valid options: {', '.join(valid_whisper_sizes)}")
    
    try:
        temp = float(os.getenv('OLLAMA_TEMPERATURE', '0.3'))
        if not 0.0 <= temp <= 1.0:
            logging.warning(f"OLLAMA_TEMPERATURE should be between 0.0 and 1.0, got {temp}")
    except ValueError:
        logging.warning(f"Invalid OLLAMA_TEMPERATURE value, using default 0.3")
    
    logging.info("Configuration validation passed")

from src.processor import OllamaProcessor
from src.note_manager import NoteManager
from src.state_manager import StateManager
from src.processing_queue import ProcessingQueue, ProcessingStatus
from src.content_processor_manager import ContentProcessorManager

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        logging.info("Initializing AudioFileHandler...")
        
        # Initialize state manager for persistent tracking
        self.state_manager = StateManager()
        
        # Initialize processing queue
        max_workers = int(os.getenv('MAX_CONCURRENT_PROCESSING', '1'))
        self.processing_queue = ProcessingQueue(max_workers=max_workers)
        self.processing_queue.set_processor(self._process_audio_file)
        
        # Initialize components (needed even in dry_run for file type detection)
        if not self.dry_run:
            self.initialize_components()
            # Start processing queue workers
            self.processing_queue.start_workers()
        else:
            # In dry_run mode, still initialize the content processor manager for file type detection
            try:
                self.content_processor_manager = ContentProcessorManager()
                logging.info("Content processor manager initialized (dry-run mode)")
            except Exception as e:
                logging.warning(f"Could not initialize content processor manager in dry-run: {e}")
        
        # Track failed files and their attempt counts (in-memory for retry logic)
        self.failed_files = {}  # Track failed files and their attempt counts
        self.max_retry_attempts = 3  # Maximum number of retry attempts for failed files
        self.last_health_check = time.time()
        self.last_directory_scan = time.time()
        self.health_check_interval = 3600  # Run health check every hour
        self.directory_scan_interval = 300  # Scan directory every 5 minutes
        self.files_in_progress = {}  # Track files that are being monitored for stability
        self.folders_in_progress = {}  # Track folders that are being monitored for completion
        self.stability_check_interval = 1  # Check file stability every second
        self.required_stable_time = 3  # File must be stable for 3 seconds
        self.max_wait_time = 60  # Maximum time to wait for file stability (1 minute)
        self.folder_completion_timeout = 30  # Wait 30 seconds after last file addition before processing folder
        self.last_empty_notification = 0  # Track when we last notified about empty folder
        self.empty_notification_interval = 300  # How often to notify about empty folder (10 minutes)
        self.last_stats_log = time.time()
        self.stats_log_interval = 3600  # Log statistics every hour
        
        # Ensure error directory exists
        self.error_dir = Path(os.getenv('WATCH_FOLDER')) / 'errors'
        self.error_dir.mkdir(exist_ok=True)
        
        # Initial scan
        self.scan_directory()
        
    def initialize_components(self):
        """Initialize or reinitialize components with error handling"""
        try:
            self.content_processor_manager = ContentProcessorManager()
            logging.info("Content processor manager initialized")
            self.processor = OllamaProcessor()
            logging.info("Ollama processor initialized")
            self.note_manager = NoteManager()
            logging.info("Note manager initialized")
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise

    def check_health(self):
        """Perform periodic health checks and cleanup"""
        current_time = time.time()
        if current_time - self.last_health_check >= self.health_check_interval:
            logging.info("Performing periodic health check...")
            try:
                # Clean up old database entries (keep last 90 days)
                if not self.dry_run:
                    self.state_manager.cleanup_old_entries(days_to_keep=90)

                # Check Ollama connection
                if not self.dry_run and not self.check_ollama_health():
                    logging.warning("Ollama connection issue detected, reinitializing processor")
                    self.processor = OllamaProcessor()
                
                # Unload models to free memory
                if not self.dry_run and hasattr(self, 'content_processor_manager'):
                    self.content_processor_manager.unload_models()

                # Force garbage collection
                gc.collect()
                
                self.last_health_check = current_time
                logging.info("Health check completed successfully")
            except Exception as e:
                logging.error(f"Error during health check: {str(e)}")
        
        # Log statistics periodically
        if current_time - self.last_stats_log >= self.stats_log_interval:
            self._log_statistics()
            self.last_stats_log = current_time

        # Check if it's time to scan the directory
        if current_time - self.last_directory_scan >= self.directory_scan_interval:
            had_files = self.scan_directory()
            self.last_directory_scan = current_time
            
            # Only log empty status periodically to avoid spam
            if not had_files and not self.files_in_progress:
                if current_time - self.last_empty_notification >= self.empty_notification_interval:
                    next_scan = time.localtime(current_time + self.directory_scan_interval)
                    next_check = time.strftime('%I:%M %p', next_scan)
                    logging.info(f"Watch folder is empty, monitoring for new files (next check at {next_check})")
                    self.last_empty_notification = current_time

        # Check files in progress
        if self.files_in_progress:
            self.check_files_in_progress()
        
        # Check folders in progress
        if self.folders_in_progress:
            self.check_folders_in_progress()

    def check_ollama_health(self):
        """Check if Ollama is responsive"""
        try:
            response = requests.get(self.processor.api_url.replace('/api/generate', '/api/version'))
            return response.status_code == 200
        except:
            return False

    def check_files_in_progress(self):
        """Check the stability of files being monitored"""
        current_time = time.time()
        files_to_remove = []

        for file_path, file_info in self.files_in_progress.items():
            try:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    logging.debug(f"File no longer exists, removing from monitoring: {file_path}")
                    files_to_remove.append(file_path)
                    continue

                current_size = file_path_obj.stat().st_size
                last_check_time = file_info['last_check_time']
                last_size = file_info['last_size']
                first_seen_time = file_info['first_seen_time']
                last_stable_time = file_info.get('last_stable_time', current_time)
                
                time_since_stable = current_time - last_stable_time
                time_since_first_seen = current_time - first_seen_time

                # Update size information
                if current_size != last_size:
                    logging.debug(f"File size changed: {file_path} ({last_size} -> {current_size} bytes), resetting stability timer")
                    file_info['last_size'] = current_size
                    file_info['last_stable_time'] = current_time
                elif time_since_stable >= self.required_stable_time:
                    # File has been stable for required time
                    logging.info(f"File {file_path} is stable (stable for {time_since_stable:.1f}s), adding to processing queue...")
                    self._add_to_processing_queue(file_path_obj)
                    files_to_remove.append(file_path)
                elif time_since_first_seen >= self.max_wait_time:
                    # File has been waiting too long
                    logging.warning(f"File {file_path} exceeded maximum wait time ({time_since_first_seen:.1f}s), adding to queue anyway...")
                    self._add_to_processing_queue(file_path_obj)
                    files_to_remove.append(file_path)
                else:
                    # Log progress periodically (every 5 seconds)
                    if int(time_since_first_seen) % 5 == 0 and int(time_since_first_seen) > 0:
                        logging.debug(f"File still being monitored: {file_path} (stable for {time_since_stable:.1f}s, total wait: {time_since_first_seen:.1f}s)")

                file_info['last_check_time'] = current_time

            except Exception as e:
                logging.error(f"Error checking file {file_path}: {str(e)}")
                logging.exception("Full error trace:")
                files_to_remove.append(file_path)

        # Remove processed or errored files
        for file_path in files_to_remove:
            self.files_in_progress.pop(file_path, None)

    def on_created(self, event):
        try:
            path = Path(event.src_path)
            
            # Handle folder creation
            if event.is_directory:
                # Skip the errors folder and anything inside it
                if path == self.error_dir or self.error_dir in path.parents:
                    logging.debug(f"Skipping errors folder or folder inside errors: {path}")
                    return
                
                if not self.dry_run and hasattr(self, 'content_processor_manager') and self.content_processor_manager.can_process(path):
                    logging.info(f"New folder detected: {path}")
                    self.start_monitoring_folder(path)
                return
            
            # Handle file creation
            file_path = path
            
            # Skip files in the errors folder
            if self.error_dir in file_path.parents or file_path.parent == self.error_dir:
                logging.debug(f"Skipping file in errors folder: {file_path}")
                return
            
            # Skip iCloud placeholder files
            if file_path.name.endswith('.icloud'):
                logging.debug(f"Skipping iCloud placeholder file: {file_path}")
                return
            
            # Check if file is actually an iCloud placeholder (sometimes they don't have .icloud extension)
            if file_path.exists() and file_path.stat().st_size == 0:
                logging.debug(f"Skipping empty file (likely iCloud placeholder): {file_path}")
                return
            
            # Check if file is inside a folder we're monitoring
            parent_folder = file_path.parent
            if str(parent_folder) in self.folders_in_progress:
                # File added to a monitored folder - update folder monitoring
                logging.debug(f"File added to monitored folder: {file_path.name} in {parent_folder.name}")
                self.folders_in_progress[str(parent_folder)]['last_file_added'] = time.time()
                self.folders_in_progress[str(parent_folder)]['file_count'] = len([f for f in parent_folder.iterdir() if f.is_file() and not f.name.startswith('.')])
                return  # Don't process individual files in monitored folders
            
            # Check if we can process this file type
            if not self.dry_run and hasattr(self, 'content_processor_manager') and self.content_processor_manager.can_process(file_path):
                # Check if we've already processed this file using state manager
                if self.state_manager.is_processed(file_path):
                    logging.info(f"File already processed, skipping: {file_path}")
                    return
                
                file_type = self.content_processor_manager.get_processor(file_path).get_source_type()
                logging.info(f"New {file_type} file detected: {file_path}")
                logging.debug(f"Calling start_monitoring_file for {file_path}")
                self.start_monitoring_file(file_path)
                logging.debug(f"Returned from start_monitoring_file for {file_path}")
                
        except Exception as e:
            logging.error(f"Error in on_created handler for {event.src_path}: {str(e)}")
            logging.exception("Full error trace:")

    def start_monitoring_file(self, file_path):
        """Start monitoring a file for stability"""
        try:
            current_time = time.time()
            str_path = str(file_path)
            
            logging.debug(f"Attempting to start monitoring file: {file_path}")
            
            # Check if file exists and is accessible (with retry for iCloud sync)
            max_checks = 3
            file_exists = False
            file_size = 0
            for check in range(max_checks):
                try:
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        if file_size > 0:  # Skip empty files (iCloud placeholders)
                            file_exists = True
                            break
                        else:
                            logging.debug(f"File exists but is empty (iCloud placeholder?), attempt {check + 1}/{max_checks}: {file_path}")
                    else:
                        logging.debug(f"File does not exist yet, attempt {check + 1}/{max_checks}: {file_path}")
                except (OSError, PermissionError) as e:
                    logging.debug(f"Error accessing file, attempt {check + 1}/{max_checks}: {e}")
                
                if check < max_checks - 1:
                    time.sleep(1)  # Wait 1 second before retry
            
            if not file_exists:
                logging.warning(f"File not accessible or is empty after {max_checks} attempts (may still be syncing from iCloud): {file_path}")
                logging.info(f"Will retry on next directory scan. File path: {file_path}")
                return
            
            # Check if file has previously failed using state manager
            if not self.dry_run:
                try:
                    file_info = self.state_manager.get_file_info(file_path)
                    if file_info and file_info.get('status') == 'failed':
                        attempt_count = self.failed_files.get(str_path, 0)
                        if attempt_count >= self.max_retry_attempts:
                            logging.warning(f"File {file_path} has exceeded maximum retry attempts. Moving to error directory.")
                            self.move_to_error_dir(file_path)
                            return
                        logging.info(f"Retrying file {file_path} (attempt {attempt_count + 1}/{self.max_retry_attempts})")
                except Exception as e:
                    logging.warning(f"Error checking file info in state manager: {e}, continuing anyway")
            
            # Check in-memory failed files tracking
            if str_path in self.failed_files:
                attempts = self.failed_files[str_path]
                if attempts >= self.max_retry_attempts:
                    logging.warning(f"File {file_path} has exceeded maximum retry attempts. Moving to error directory.")
                    self.move_to_error_dir(file_path)
                    return
                logging.info(f"Retrying file {file_path} (attempt {attempts + 1}/{self.max_retry_attempts})")
            
            # Check if already being monitored
            if str_path in self.files_in_progress:
                logging.debug(f"File already being monitored: {file_path}")
                return
            
            # Check if already in processing queue
            if not self.dry_run and hasattr(self, 'content_processor_manager'):
                if not self.content_processor_manager.can_process(file_path):
                    logging.warning(f"No processor available for file type: {file_path.suffix}")
                    return
                
                queue_status = self.processing_queue.get_job_status(file_path)
                if queue_status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
                    logging.info(f"File already in processing queue (status: {queue_status}), skipping monitoring: {file_path}")
                    return
            
            # Start monitoring the file
            self.files_in_progress[str_path] = {
                'first_seen_time': current_time,
                'last_check_time': current_time,
                'last_size': file_size,
                'last_stable_time': current_time
            }
            logging.info(f"Started monitoring file: {file_path} (size: {file_size} bytes)")
            logging.debug(f"File will be processed after {self.required_stable_time}s of stability")
        except Exception as e:
            logging.error(f"Error starting to monitor file {file_path}: {str(e)}")
            logging.exception("Full error trace:")
    
    def start_monitoring_folder(self, folder_path: Path):
        """Start monitoring a folder for completion (no new files added for timeout period)."""
        try:
            current_time = time.time()
            str_path = str(folder_path)
            
            logging.info(f"Starting to monitor folder: {folder_path}")
            
            # Check if folder exists
            if not folder_path.exists() or not folder_path.is_dir():
                logging.warning(f"Folder does not exist or is not a directory: {folder_path}")
                return
            
            # Check if already being monitored
            if str_path in self.folders_in_progress:
                logging.debug(f"Folder already being monitored: {folder_path}")
                return
            
            # Check if already processed
            if not self.dry_run and self.state_manager.is_processed(folder_path):
                logging.info(f"Folder already processed, skipping: {folder_path}")
                return
            
            # Count files in folder
            files = [f for f in folder_path.iterdir() if f.is_file() and not f.name.startswith('.')]
            file_count = len(files)
            
            # Start monitoring the folder
            self.folders_in_progress[str_path] = {
                'first_seen_time': current_time,
                'last_file_added': current_time,
                'last_check_time': current_time,
                'file_count': file_count,
            }
            logging.info(f"Started monitoring folder: {folder_path} ({file_count} file(s))")
            logging.debug(f"Folder will be processed after {self.folder_completion_timeout}s with no new files")
            
        except Exception as e:
            logging.error(f"Error starting to monitor folder {folder_path}: {str(e)}")
            logging.exception("Full error trace:")
    
    def check_folders_in_progress(self):
        """Check folders being monitored and process them when complete."""
        current_time = time.time()
        folders_to_remove = []
        
        for folder_path_str, folder_info in self.folders_in_progress.items():
            try:
                folder_path = Path(folder_path_str)
                
                if not folder_path.exists():
                    logging.debug(f"Folder no longer exists, removing from monitoring: {folder_path}")
                    folders_to_remove.append(folder_path_str)
                    continue
                
                last_file_added = folder_info['last_file_added']
                time_since_last_file = current_time - last_file_added
                first_seen_time = folder_info['first_seen_time']
                time_since_first_seen = current_time - first_seen_time
                
                # Count current files
                current_files = [f for f in folder_path.iterdir() if f.is_file() and not f.name.startswith('.')]
                current_file_count = len(current_files)
                
                # Update file count if it changed
                if current_file_count != folder_info['file_count']:
                    logging.debug(f"File count changed in folder {folder_path.name}: {folder_info['file_count']} -> {current_file_count}")
                    folder_info['file_count'] = current_file_count
                    folder_info['last_file_added'] = current_time
                    folder_info['last_check_time'] = current_time
                    continue  # Reset timer, continue monitoring
                
                # Check if folder is complete (no new files for timeout period)
                if time_since_last_file >= self.folder_completion_timeout:
                    # Folder is complete - process it
                    logging.info(f"Folder {folder_path.name} is complete (no new files for {time_since_last_file:.1f}s, {current_file_count} file(s)), adding to processing queue...")
                    self._add_to_processing_queue(folder_path)
                    folders_to_remove.append(folder_path_str)
                elif time_since_first_seen >= 300:  # Log progress every 5 minutes
                    logging.debug(f"Folder still being monitored: {folder_path.name} (last file added {time_since_last_file:.1f}s ago, {current_file_count} file(s))")
                
                folder_info['last_check_time'] = current_time
                
            except Exception as e:
                logging.error(f"Error checking folder {folder_path_str}: {str(e)}")
                logging.exception("Full error trace:")
                folders_to_remove.append(folder_path_str)
        
        # Remove processed or errored folders
        for folder_path_str in folders_to_remove:
            self.folders_in_progress.pop(folder_path_str, None)
    
    def _add_to_processing_queue(self, file_path: Path):
        """Add a file to the processing queue."""
        try:
            # Double-check it hasn't been processed
            if not self.dry_run and self.state_manager.is_processed(file_path):
                logging.info(f"File already processed, skipping: {file_path}")
                return
            
            # Check if we can process this file type
            if not self.dry_run and hasattr(self, 'content_processor_manager'):
                if not self.content_processor_manager.can_process(file_path):
                    logging.debug(f"No processor available for file type: {file_path.suffix}")
                    return
            
            # Check if already in queue before attempting to add
            queue_status = self.processing_queue.get_job_status(file_path)
            if queue_status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
                logging.info(f"File already in processing queue (status: {queue_status}), skipping: {file_path}")
                return
            
            # Add to queue
            if self.processing_queue.add_file(file_path):
                logging.info(f"Added file to processing queue: {file_path}")
            else:
                logging.warning(f"Could not add file to queue (may already be queued): {file_path}")
        except Exception as e:
            logging.error(f"Error adding file to processing queue {file_path}: {e}", exc_info=True)

    def move_to_error_dir(self, file_path):
        """Move a failed file to the error directory with metadata"""
        try:
            # Create a unique name in case of conflicts
            error_path = self.error_dir / f"{file_path.name}"
            if error_path.exists():
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                error_path = self.error_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            # Move the file
            file_path.rename(error_path)
            
            # Create metadata file
            metadata_path = error_path.with_suffix(error_path.suffix + '.error')
            with open(metadata_path, 'w') as f:
                metadata = {
                    'original_path': str(file_path),
                    'first_error_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'attempts': self.failed_files.get(str(file_path), 0),
                    'last_error': self.failed_files.get(str(file_path) + '_last_error', 'Unknown error')
                }
                json.dump(metadata, f, indent=2)
            
            logging.info(f"Moved failed file to {error_path} with metadata")
            
            # Remove from failed files tracking
            self.failed_files.pop(str(file_path), None)
            self.failed_files.pop(str(file_path) + '_last_error', None)
            
        except Exception as e:
            logging.error(f"Failed to move file {file_path} to error directory: {str(e)}")

    def _extract_source_datetime(self, file_path: Path) -> tuple:
        """Extract the source date and time from the audio filename"""
        try:
            # Extract date and optionally time from filename
            datetime_match = re.match(r'(\d{4}-\d{2}-\d{2})(?:[-_](\d{2}[-_]\d{2}(?:AM|PM)?))?\b', file_path.stem, re.IGNORECASE)
            if datetime_match:
                date = datetime_match.group(1)
                time = datetime_match.group(2) if datetime_match.group(2) else None
                return date, time
            return None, None
        except Exception as e:
            logging.error(f"Error extracting datetime from filename: {str(e)}")
            return None, None

    def _process_audio_file(self, file_path):
        """Process a file (audio, text, etc.). Called by the processing queue."""
        start_time = time.time()
        str_path = str(file_path)
        file_hash = None
        note_path = None
        attachment_path = None
        transcription_language = None
        
        try:
            logging.info(f"Processing file: {file_path}")
            
            if self.dry_run:
                logging.info(f"[DRY RUN] Would process: {file_path}")
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / 1024 / 1024
                    logging.info(f"[DRY RUN] File size: {file_size_mb:.2f} MB")
                return
            
            # Get the appropriate content processor
            content_processor = self.content_processor_manager.get_processor(file_path)
            if not content_processor:
                raise ValueError(f"No processor available for file type: {file_path.suffix}")
            
            source_type = content_processor.get_source_type()
            logging.info(f"Using {source_type} processor for {file_path}")
            
            # Check if file exists and is accessible (with retry for iCloud sync)
            max_checks = 3
            for check in range(max_checks):
                if file_path.exists():
                    break
                if check < max_checks - 1:
                    logging.debug(f"File not found (attempt {check + 1}/{max_checks}), waiting for iCloud sync...")
                    time.sleep(2)
                else:
                    logging.debug(f"File not found after {max_checks} attempts (may have been moved): {file_path}")
                    self.state_manager.mark_failed(file_path, "File not found after multiple attempts", 
                                                  attempt_count=self.failed_files.get(str_path, 0) + 1)
                    return
            
            # Compute file hash BEFORE processing (file will be moved later)
            file_hash = self.state_manager.get_file_hash(file_path)
            
            # Mark as processing in state manager
            self.state_manager.mark_processing(file_path)
            
            # Extract source date and time
            source_date, source_time = self._extract_source_datetime(file_path)
            
            # Extract content using the appropriate processor
            logging.info(f"Extracting content from {source_type} file...")
            extracted_content = content_processor.extract_content(file_path)
            logging.info(f"Content extraction completed successfully")
            
            # Log metadata information
            transcription_language = extracted_content.get("language")
            if transcription_language:
                logging.info(f"Detected language: {transcription_language}")
            
            # Unload models to free memory before Ollama processing (for audio)
            # This enables sequential processing for better memory efficiency
            sequential_mode = os.getenv('SEQUENTIAL_PROCESSING', 'true').lower() == 'true'
            if sequential_mode and source_type == 'audio' and hasattr(content_processor, 'unload_model'):
                logging.info("Unloading model to free memory for Ollama processing...")
                content_processor.unload_model()
            
            # Process with Ollama (pass file path for vision support)
            processed_content = self.processor.process_content(
                extracted_content, 
                file_path.name,
                source_type=source_type,
                source_file_path=file_path
            )
            
            # Add source date/time to processed content
            if source_date:
                processed_content['source_date'] = source_date
                if source_time:
                    processed_content['source_time'] = source_time
            
            logging.info("Ollama processing completed")
            
            # Create note (this also moves the source file)
            note_result = self.note_manager.create_note(processed_content, file_path)
            note_path = note_result.get('note_path')
            attachment_path = note_result.get('audio_path') or note_result.get('attachment_path')
            
            processing_time = time.time() - start_time
            logging.info(f"Note created successfully for: {file_path} (took {processing_time:.1f}s)")
            
            # Mark as successful in state manager (use pre-computed hash since file was moved)
            self.state_manager.mark_success(
                file_path, 
                processing_time,
                note_path=note_path,
                audio_path=attachment_path,  # Works for both audio and other attachments
                transcription_language=transcription_language,
                file_hash=file_hash  # Use hash computed before file was moved
            )
            
            # Remove from failed files if it was there
            if str_path in self.failed_files:
                self.failed_files.pop(str_path)
                self.failed_files.pop(str_path + '_last_error', None)
                
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Provide more user-friendly error messages
            if "Transcription failed" in error_msg or "extract_content" in error_msg.lower():
                user_msg = f"Failed to extract content from file. This might be due to:\n  - Corrupted file\n  - Unsupported file format\n  - Insufficient system resources\n  Original error: {error_msg}"
            elif "Ollama" in error_type or "connection" in error_msg.lower():
                user_msg = f"Failed to connect to Ollama. Please ensure:\n  - Ollama is running (run 'ollama serve')\n  - The model '{os.getenv('OLLAMA_MODEL')}' is available (run 'ollama pull {os.getenv('OLLAMA_MODEL')}')\n  - Original error: {error_msg}"
            elif "YAML" in error_type or "YAML" in error_msg or "frontmatter" in error_msg.lower():
                user_msg = f"Failed to parse AI response. This might be due to:\n  - Model response format issues (expected YAML frontmatter)\n  - Network timeout\n  - Original error: {error_msg}"
            else:
                user_msg = f"Unexpected error: {error_msg}"
            
            logging.error(f"Error processing {file_path}: {user_msg}")
            logging.exception("Full error trace:")
            
            # Track the failure
            attempt_count = self.failed_files.get(str_path, 0) + 1
            self.failed_files[str_path] = attempt_count
            self.failed_files[str_path + '_last_error'] = error_msg
            
            # Mark as failed in state manager
            if not self.dry_run:
                self.state_manager.mark_failed(file_path, error_msg, attempt_count)
            
            # If we've exceeded max retries, move to error directory
            if attempt_count >= self.max_retry_attempts:
                logging.warning(f"File {file_path} has exceeded maximum retry attempts ({self.max_retry_attempts}). Moving to error directory.")
                if not self.dry_run:
                    self.move_to_error_dir(file_path)
    
    def _log_statistics(self):
        """Log processing statistics."""
        try:
            stats = self.state_manager.get_statistics()
            queue_stats = self.processing_queue.get_queue_stats()
            
            logging.info("=" * 60)
            logging.info("Processing Statistics:")
            logging.info(f"  Total Success: {stats.get('total_success', 0)}")
            logging.info(f"  Total Failed: {stats.get('total_failed', 0)}")
            logging.info(f"  Success Rate: {stats.get('success_rate', 0)}%")
            logging.info(f"  Files Processed Today: {stats.get('files_processed_today', 0)}")
            logging.info(f"  Avg Processing Time: {stats.get('avg_processing_duration', 0):.1f}s")
            logging.info(f"  Total Processing Time: {stats.get('total_processing_duration', 0):.1f}s")
            logging.info("Queue Statistics:")
            logging.info(f"  Queue Size: {queue_stats.get('queue_size', 0)}")
            logging.info(f"  Pending: {queue_stats.get('pending', 0)}")
            logging.info(f"  Processing: {queue_stats.get('processing', 0)}")
            logging.info(f"  Completed: {queue_stats.get('completed', 0)}")
            logging.info(f"  Failed: {queue_stats.get('failed', 0)}")
            logging.info(f"  Workers: {queue_stats.get('workers_alive', 0)}/{queue_stats.get('workers_total', 0)} alive")
            logging.info(f"  Queue Running: {queue_stats.get('is_running', False)}")
            logging.info("=" * 60)
            
            # Warn if workers are dead
            if queue_stats.get('workers_alive', 0) < queue_stats.get('workers_total', 0):
                logging.warning(f"WARNING: {queue_stats.get('workers_total', 0) - queue_stats.get('workers_alive', 0)} worker thread(s) are dead!")
        except Exception as e:
            logging.error(f"Error logging statistics: {e}")

    def scan_directory(self):
        """Scan the watch directory for any unprocessed audio files"""
        try:
            watch_path = os.getenv('WATCH_FOLDER')
            if not watch_path or not os.path.exists(watch_path):
                logging.error(f"Watch folder not found or not set: {watch_path}")
                return False

            # Get all files and folders except those in the errors directory
            # Skip iCloud placeholder files and empty files
            all_items = []
            for item in Path(watch_path).iterdir():
                # Skip the errors folder itself and anything inside it
                if item == self.error_dir or self.error_dir in item.parents or item.parent == self.error_dir:
                    continue
                
                # Check folders
                if item.is_dir():
                    if self.content_processor_manager.can_process(item):
                        all_items.append(item)
                    continue
                
                # Check files
                if (item.is_file() 
                    and self.content_processor_manager.can_process(item)
                    and not item.name.endswith('.icloud')):
                    # Skip empty files (likely iCloud placeholders)
                    try:
                        if item.exists() and item.stat().st_size > 0:
                            all_items.append(item)
                    except (OSError, PermissionError) as e:
                        logging.debug(f"Could not check file {item}: {e}")
                        continue
            
            if all_items:
                logging.info(f"Found {len(all_items)} item(s) in watch directory")
                for item_path in all_items:
                    str_path = str(item_path)
                    # Check if already processed using state manager
                    if not self.dry_run and self.state_manager.is_processed(item_path):
                        continue
                    
                    # Handle folders
                    if item_path.is_dir():
                        if str_path not in self.folders_in_progress:
                            if not self.dry_run and hasattr(self, 'content_processor_manager'):
                                if self.content_processor_manager.can_process(item_path):
                                    logging.info(f"Found new folder: {item_path.name}")
                                    self.start_monitoring_folder(item_path)
                        continue
                    
                    # Handle files
                    if str_path not in self.files_in_progress:
                        # Check if we can process this file type
                        if not self.dry_run and hasattr(self, 'content_processor_manager'):
                            if not self.content_processor_manager.can_process(item_path):
                                continue
                        
                        queue_status = self.processing_queue.get_job_status(item_path)
                        if queue_status is None:  # Not in queue
                            logging.info(f"Found new file: {item_path.name}")
                            self.start_monitoring_file(item_path)
                return True
            return False
            
        except Exception as e:
            logging.error(f"Error during directory scan: {str(e)}")
            logging.exception("Full error trace:")
            return False

def ensure_ollama_running():
    """Check if Ollama is running and start it if not."""
    try:
        # Try to connect to Ollama API
        response = requests.get('http://localhost:11434/api/version')
        if response.status_code == 200:
            logging.info("Ollama is already running")
            return True
    except requests.exceptions.ConnectionError:
        logging.info("Ollama is not running. Attempting to start...")
        try:
            if platform.system() == 'Windows':
                # Start Ollama in a new process window
                subprocess.Popen('ollama serve', 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(['ollama', 'serve'])
            
            # Wait for Ollama to start (up to 30 seconds)
            max_attempts = 30
            for i in range(max_attempts):
                try:
                    response = requests.get('http://localhost:11434/api/version')
                    if response.status_code == 200:
                        logging.info("Ollama started successfully")
                        return True
                except requests.exceptions.ConnectionError:
                    if i < max_attempts - 1:
                        time.sleep(1)
                        continue
                    logging.error("Failed to start Ollama after 30 seconds")
                    return False
        except FileNotFoundError:
            logging.error("Ollama executable not found. Please ensure Ollama is installed")
            return False
        except Exception as e:
            logging.error(f"Error starting Ollama: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description='EchoEtcher - Automated audio transcription and note generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run normally
  python main.py --dry-run          # Test without processing files
  python main.py --verbose          # Enable debug logging
  python main.py --version          # Show version information
        """
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Test mode: scan and log but do not process files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Set up logging first (before loading env to see any issues)
    setup_logging(verbose=args.verbose)
    
    load_dotenv(override=True)
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)
    
    # Log configuration (without sensitive data)
    logging.info(f"EchoEtcher v{__version__} starting...")
    logging.info(f"WATCH_FOLDER = {os.getenv('WATCH_FOLDER')}")
    logging.info(f"OBSIDIAN_VAULT_PATH = {os.getenv('OBSIDIAN_VAULT_PATH')}")
    logging.info(f"NOTES_FOLDER = {os.getenv('NOTES_FOLDER', 'notes')}")
    logging.info(f"OLLAMA_MODEL = {os.getenv('OLLAMA_MODEL')}")
    vision_model = os.getenv('OLLAMA_VISION_MODEL')
    if vision_model:
        logging.info(f"OLLAMA_VISION_MODEL = {vision_model}")
    logging.info(f"WHISPER_MODEL_SIZE = {os.getenv('WHISPER_MODEL_SIZE', 'medium')}")
    
    if args.dry_run:
        logging.info("DRY RUN MODE: Files will be scanned but not processed")
    
    # Set up signal handlers for graceful shutdown
    observer = None
    event_handler = None
    
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
        if observer:
            observer.stop()
            observer.join()
        if event_handler and hasattr(event_handler, 'processing_queue'):
            event_handler.processing_queue.stop_workers(wait=True, timeout=10)
        logging.info("Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if not args.dry_run:
            # Ensure Ollama is running
            ensure_ollama_running()
        
        # Initialize the event handler (this also starts the processing queue)
        event_handler = AudioFileHandler(dry_run=args.dry_run)
        
        # Set up the observer with error handling
        observer = Observer()
        watch_path = os.getenv('WATCH_FOLDER')
        
        observer.schedule(event_handler, watch_path, recursive=False)
        observer.start()
        
        # Log initial startup message
        logging.info(f"Started watching folder: {watch_path}")
        logging.info(f"Checking for new files every {event_handler.directory_scan_interval} seconds")
        
        # Main loop with health monitoring
        last_check = time.time()
        check_interval = event_handler.directory_scan_interval  # Match the handler's interval
        
        while True:
            try:
                time.sleep(1)  # Sleep for 1 second, no need for rapid checks
                
                current_time = time.time()
                
                # Check files in progress every second (for stability checking)
                # This ensures files are processed as soon as they're stable
                if event_handler.files_in_progress:
                    event_handler.check_files_in_progress()
                
                # Check folders in progress every second (for completion checking)
                if event_handler.folders_in_progress:
                    event_handler.check_folders_in_progress()
                
                if current_time - last_check >= check_interval:
                    # Force a health check and directory scan
                    event_handler.check_health()
                    last_check = current_time
                
                if not observer.is_alive():
                    logging.error("Observer thread died, restarting...")
                    observer.stop()
                    observer.join()
                    observer = Observer()
                    observer.schedule(event_handler, watch_path, recursive=False)
                    observer.start()
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                logging.exception("Full error trace:")
                time.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
