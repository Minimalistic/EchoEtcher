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
    """Check if ffmpeg is available in PATH."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            # Extract version info
            version_line = result.stdout.split('\n')[0] if result.stdout else "available"
            logging.info(f"FFmpeg found: {version_line}")
            return True
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.warning(f"Error checking ffmpeg: {e}")
    
    # Try to find ffmpeg in common locations
    common_paths = [
        '/opt/homebrew/bin/ffmpeg',  # Homebrew on Apple Silicon
        '/usr/local/bin/ffmpeg',     # Homebrew on Intel Mac
        '/usr/bin/ffmpeg',           # System location
    ]
    
    for path in common_paths:
        if Path(path).exists():
            logging.warning(f"FFmpeg found at {path} but not in PATH. Adding to PATH.")
            os.environ['PATH'] = str(Path(path).parent) + os.pathsep + os.environ.get('PATH', '')
            return True
    
    error_msg = (
        "FFmpeg not found! FFmpeg is required for audio processing.\n\n"
        "Install it with:\n"
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
        'OLLAMA_MODEL': 'Ollama model name to use for processing'
    }
    
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

from src.transcriber import WhisperTranscriber
from src.processor import OllamaProcessor
from src.note_manager import NoteManager

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        logging.info("Initializing AudioFileHandler...")
        if not self.dry_run:
            self.initialize_components()
        self.processed_files = set()  # Track processed files
        self.failed_files = {}  # Track failed files and their attempt counts
        self.max_retry_attempts = 3  # Maximum number of retry attempts for failed files
        self.last_health_check = time.time()
        self.last_directory_scan = time.time()
        self.health_check_interval = 3600  # Run health check every hour
        self.directory_scan_interval = 300  # Scan directory every 5 minutes
        self.max_processed_files = 1000  # Maximum number of processed files to track
        self.files_in_progress = {}  # Track files that are being monitored for stability
        self.stability_check_interval = 1  # Check file stability every second
        self.required_stable_time = 3  # File must be stable for 3 seconds
        self.max_wait_time = 60  # Maximum time to wait for file stability (1 minute)
        self.last_empty_notification = 0  # Track when we last notified about empty folder
        self.empty_notification_interval = 300  # How often to notify about empty folder (10 minutes)
        
        # Ensure error directory exists
        self.error_dir = Path(os.getenv('WATCH_FOLDER')) / 'errors'
        self.error_dir.mkdir(exist_ok=True)
        
        # Initial scan
        self.scan_directory()
        
    def initialize_components(self):
        """Initialize or reinitialize components with error handling"""
        try:
            self.transcriber = WhisperTranscriber()
            logging.info("Whisper model loaded successfully")
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
                # Clean up processed files set to prevent memory growth
                if len(self.processed_files) > self.max_processed_files:
                    logging.info("Cleaning up processed files tracking set")
                    self.processed_files.clear()

                # Check Ollama connection
                if not self.check_ollama_health():
                    logging.warning("Ollama connection issue detected, reinitializing processor")
                    self.processor = OllamaProcessor()

                # Force garbage collection
                gc.collect()
                
                self.last_health_check = current_time
                logging.info("Health check completed successfully")
            except Exception as e:
                logging.error(f"Error during health check: {str(e)}")

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
                if not Path(file_path).exists():
                    files_to_remove.append(file_path)
                    continue

                current_size = Path(file_path).stat().st_size
                last_check_time = file_info['last_check_time']
                last_size = file_info['last_size']
                first_seen_time = file_info['first_seen_time']
                last_stable_time = file_info.get('last_stable_time', current_time)

                # Update size information
                if current_size != last_size:
                    file_info['last_size'] = current_size
                    file_info['last_stable_time'] = current_time
                elif current_time - last_stable_time >= self.required_stable_time:
                    # File has been stable for required time
                    logging.info(f"File {file_path} is stable, processing...")
                    self._process_audio_file(Path(file_path))
                    files_to_remove.append(file_path)
                elif current_time - first_seen_time >= self.max_wait_time:
                    # File has been waiting too long
                    logging.warning(f"File {file_path} exceeded maximum wait time, processing anyway...")
                    self._process_audio_file(Path(file_path))
                    files_to_remove.append(file_path)

                file_info['last_check_time'] = current_time

            except Exception as e:
                logging.error(f"Error checking file {file_path}: {str(e)}")
                files_to_remove.append(file_path)

        # Remove processed or errored files
        for file_path in files_to_remove:
            self.files_in_progress.pop(file_path, None)

    def on_created(self, event):
        if event.is_directory:
            return
        
        try:
            file_path = Path(event.src_path)
            
            # Skip iCloud placeholder files
            if file_path.name.endswith('.icloud'):
                logging.debug(f"Skipping iCloud placeholder file: {file_path}")
                return
            
            # Check if file is actually an iCloud placeholder (sometimes they don't have .icloud extension)
            if file_path.exists() and file_path.stat().st_size == 0:
                logging.debug(f"Skipping empty file (likely iCloud placeholder): {file_path}")
                return
            
            if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a']:
                # Check if we've already processed this file
                if file_path.name in self.processed_files:
                    logging.info(f"File already processed, skipping: {file_path}")
                    return
                
                logging.info(f"New audio file detected: {file_path}")
                self.start_monitoring_file(file_path)
                
        except Exception as e:
            logging.error(f"Error in on_created handler for {event.src_path}: {str(e)}")
            logging.exception("Full error trace:")

    def start_monitoring_file(self, file_path):
        """Start monitoring a file for stability"""
        try:
            current_time = time.time()
            str_path = str(file_path)
            
            # Check if file has previously failed
            if str_path in self.failed_files:
                attempts = self.failed_files[str_path]
                if attempts >= self.max_retry_attempts:
                    logging.warning(f"File {file_path} has exceeded maximum retry attempts. Moving to error directory.")
                    self.move_to_error_dir(file_path)
                    return
                logging.info(f"Retrying file {file_path} (attempt {attempts + 1}/{self.max_retry_attempts})")
            
            if str_path not in self.files_in_progress:
                size = file_path.stat().st_size
                self.files_in_progress[str_path] = {
                    'first_seen_time': current_time,
                    'last_check_time': current_time,
                    'last_size': size,
                    'last_stable_time': current_time
                }
                logging.info(f"Started monitoring file: {file_path}")
        except Exception as e:
            logging.error(f"Error starting to monitor file {file_path}: {str(e)}")

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
        try:
            start_time = time.time()
            logging.info(f"Processing file: {file_path}")
            
            if self.dry_run:
                logging.info(f"[DRY RUN] Would process: {file_path}")
                logging.info(f"[DRY RUN] File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
                self.processed_files.add(file_path.name)
                return
            
            str_path = str(file_path)
            
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
                    return
            
            # Extract source date and time
            source_date, source_time = self._extract_source_datetime(file_path)
            
            logging.info("Starting transcription...")
            transcription_data = self.transcriber.transcribe(file_path)
            logging.info("Transcription completed successfully")
            
            # Log metadata information
            if transcription_data.get("language"):
                logging.info(f"Detected language: {transcription_data['language']}")
            
            # Log confidence information
            low_confidence_segments = [s for s in transcription_data.get("segments", []) 
                                    if s.get("confidence", 0) < -1.0]
            if low_confidence_segments:
                logging.warning(f"Found {len(low_confidence_segments)} low confidence segments")
            
            # Process with Ollama
            processed_content = self.processor.process_transcription(transcription_data, file_path.name)
            
            # Add source date/time to processed content
            if source_date:
                processed_content['source_date'] = source_date
                if source_time:
                    processed_content['source_time'] = source_time
            
            logging.info("Ollama processing completed")
            
            # Create note
            self.note_manager.create_note(processed_content, file_path)
            processing_time = time.time() - start_time
            logging.info(f"Note created successfully for: {file_path} (took {processing_time:.1f}s)")
            
            # Add to processed files
            self.processed_files.add(file_path.name)
            
            # Remove from failed files if it was there
            if str_path in self.failed_files:
                self.failed_files.pop(str_path)
                self.failed_files.pop(str_path + '_last_error', None)
                
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Provide more user-friendly error messages
            if "Transcription failed" in error_msg:
                user_msg = f"Failed to transcribe audio file. This might be due to:\n  - Corrupted audio file\n  - Unsupported audio format\n  - Insufficient system resources\n  Original error: {error_msg}"
            elif "Ollama" in error_type or "connection" in error_msg.lower():
                user_msg = f"Failed to connect to Ollama. Please ensure:\n  - Ollama is running (run 'ollama serve')\n  - The model '{os.getenv('OLLAMA_MODEL')}' is available (run 'ollama pull {os.getenv('OLLAMA_MODEL')}')\n  - Original error: {error_msg}"
            elif "JSON" in error_type or "JSON" in error_msg:
                user_msg = f"Failed to parse AI response. This might be due to:\n  - Model response format issues\n  - Network timeout\n  - Original error: {error_msg}"
            else:
                user_msg = f"Unexpected error: {error_msg}"
            
            logging.error(f"Error processing {file_path}: {user_msg}")
            logging.exception("Full error trace:")
            
            str_path = str(file_path)
            # Track the failure
            self.failed_files[str_path] = self.failed_files.get(str_path, 0) + 1
            self.failed_files[str_path + '_last_error'] = error_msg
            
            # If we've exceeded max retries, move to error directory
            if self.failed_files[str_path] >= self.max_retry_attempts:
                logging.warning(f"File {file_path} has exceeded maximum retry attempts ({self.max_retry_attempts}). Moving to error directory.")
                if not self.dry_run:
                    self.move_to_error_dir(file_path)

    def scan_directory(self):
        """Scan the watch directory for any unprocessed audio files"""
        try:
            watch_path = os.getenv('WATCH_FOLDER')
            if not watch_path or not os.path.exists(watch_path):
                logging.error(f"Watch folder not found or not set: {watch_path}")
                return False

            # Get all files except those in the errors directory
            # Skip iCloud placeholder files and empty files
            all_files = []
            for f in Path(watch_path).glob('*'):
                if (f.parent != self.error_dir and f.is_file() 
                    and f.suffix.lower() in ['.mp3', '.wav', '.m4a']
                    and not f.name.endswith('.icloud')):
                    # Skip empty files (likely iCloud placeholders)
                    try:
                        if f.exists() and f.stat().st_size > 0:
                            all_files.append(f)
                    except (OSError, PermissionError) as e:
                        logging.debug(f"Could not check file {f}: {e}")
                        continue
            
            if all_files:
                logging.info(f"Found {len(all_files)} audio files in watch directory")
                for file_path in all_files:
                    str_path = str(file_path)
                    if file_path.name not in self.processed_files and str_path not in self.files_in_progress:
                        logging.info(f"Found new file: {file_path.name}")
                        self.start_monitoring_file(file_path)
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
    logging.info(f"WHISPER_MODEL_SIZE = {os.getenv('WHISPER_MODEL_SIZE', 'medium')}")
    
    if args.dry_run:
        logging.info("DRY RUN MODE: Files will be scanned but not processed")
    
    # Set up signal handlers for graceful shutdown
    observer = None
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
        if observer:
            observer.stop()
            observer.join()
        logging.info("Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if not args.dry_run:
            # Ensure Ollama is running
            ensure_ollama_running()
        
        # Initialize the event handler
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
