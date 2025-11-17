import os
import sqlite3
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import json

class StateManager:
    """
    Manages persistent state for processed files using SQLite.
    Tracks file hashes, processing status, and metadata.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the state manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Default to logs directory for database
            db_dir = Path("logs")
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "echotcher_state.db"
        
        self.db_path = Path(db_path)
        self._init_database()
        logging.info(f"State manager initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create processed_files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                processed_at TIMESTAMP NOT NULL,
                processing_duration REAL,
                status TEXT NOT NULL,
                error_message TEXT,
                transcription_language TEXT,
                note_path TEXT,
                audio_path TEXT,
                UNIQUE(file_hash)
            )
        """)
        
        # Create index on file_path for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path 
            ON processed_files(file_path)
        """)
        
        # Create index on processed_at for time-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_at 
            ON processed_files(processed_at)
        """)
        
        # Create index on status for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status 
            ON processed_files(status)
        """)
        
        conn.commit()
        conn.close()
    
    def get_file_hash(self, file_path: Path) -> str:
        """
        Generate SHA256 hash of file content or directory path.
        
        Args:
            file_path: Path to the file or directory
            
        Returns:
            SHA256 hash as hexadecimal string
        """
        # For directories, use path-based hash
        if file_path.is_dir():
            return hashlib.sha256(str(file_path).encode()).hexdigest()
        
        # For files, hash the content
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logging.error(f"Error computing hash for {file_path}: {e}")
            # Fallback to path-based hash if file can't be read
            return hashlib.sha256(str(file_path).encode()).hexdigest()
    
    def is_processed(self, file_path: Path) -> bool:
        """
        Check if a file has been processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file has been successfully processed, False otherwise
        """
        try:
            file_hash = self.get_file_hash(file_path)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT status FROM processed_files 
                WHERE file_hash = ? AND status = 'success'
            """, (file_hash,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result is not None
        except Exception as e:
            logging.error(f"Error checking if file is processed: {e}")
            return False
    
    def get_file_info(self, file_path: Path) -> Optional[Dict]:
        """
        Get processing information for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file info or None if not found
        """
        try:
            file_hash = self.get_file_hash(file_path)
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM processed_files 
                WHERE file_hash = ?
            """, (file_hash,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            logging.error(f"Error getting file info: {e}")
            return None
    
    def mark_processing(self, file_path: Path) -> str:
        """
        Mark a file as being processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File hash
        """
        try:
            file_hash = self.get_file_hash(file_path)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or update with 'processing' status
            cursor.execute("""
                INSERT OR REPLACE INTO processed_files 
                (file_hash, file_path, file_name, file_size, processed_at, status)
                VALUES (?, ?, ?, ?, ?, 'processing')
            """, (
                file_hash,
                str(file_path),
                file_path.name,
                file_size,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return file_hash
        except Exception as e:
            logging.error(f"Error marking file as processing: {e}")
            return self.get_file_hash(file_path)
    
    def mark_success(self, file_path: Path, processing_duration: float, 
                     note_path: Optional[Path] = None, 
                     audio_path: Optional[Path] = None,
                     transcription_language: Optional[str] = None,
                     file_hash: Optional[str] = None):
        """
        Mark a file as successfully processed.
        
        Args:
            file_path: Path to the original file (may not exist if already moved)
            processing_duration: Time taken to process in seconds
            note_path: Path to the created note file
            audio_path: Path to the moved audio file
            transcription_language: Detected language from transcription
            file_hash: Optional pre-computed file hash (recommended if file has been moved)
        """
        try:
            # Use provided hash or compute it (file may have been moved)
            if file_hash is None:
                file_hash = self.get_file_hash(file_path)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE processed_files 
                SET status = 'success',
                    processing_duration = ?,
                    note_path = ?,
                    audio_path = ?,
                    transcription_language = ?,
                    processed_at = ?
                WHERE file_hash = ?
            """, (
                processing_duration,
                str(note_path) if note_path else None,
                str(audio_path) if audio_path else None,
                transcription_language,
                datetime.now().isoformat(),
                file_hash
            ))
            
            conn.commit()
            conn.close()
            
            logging.debug(f"Marked file as successful: {file_path}")
        except Exception as e:
            logging.error(f"Error marking file as success: {e}")
    
    def mark_failed(self, file_path: Path, error_message: str, 
                    attempt_count: int = 1):
        """
        Mark a file as failed processing.
        
        Args:
            file_path: Path to the file
            error_message: Error message
            attempt_count: Number of attempts made
        """
        try:
            file_hash = self.get_file_hash(file_path)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Determine status based on attempt count
            status = 'failed' if attempt_count >= 3 else 'failed_retry'
            
            cursor.execute("""
                UPDATE processed_files 
                SET status = ?,
                    error_message = ?,
                    processed_at = ?
                WHERE file_hash = ?
            """, (
                status,
                error_message[:500],  # Limit error message length
                datetime.now().isoformat(),
                file_hash
            ))
            
            conn.commit()
            conn.close()
            
            logging.debug(f"Marked file as failed: {file_path} (attempt {attempt_count})")
        except Exception as e:
            logging.error(f"Error marking file as failed: {e}")
    
    def get_failed_files(self, max_retries: int = 3) -> List[Dict]:
        """
        Get list of files that failed processing but haven't exceeded max retries.
        
        Args:
            max_retries: Maximum number of retries allowed
            
        Returns:
            List of file info dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM processed_files 
                WHERE status = 'failed_retry'
                ORDER BY processed_at ASC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"Error getting failed files: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total processed
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE status = 'success'")
            total_success = cursor.fetchone()[0]
            
            # Total failed
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE status IN ('failed', 'failed_retry')")
            total_failed = cursor.fetchone()[0]
            
            # Total processing
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE status = 'processing'")
            total_processing = cursor.fetchone()[0]
            
            # Average processing time
            cursor.execute("""
                SELECT AVG(processing_duration) 
                FROM processed_files 
                WHERE status = 'success' AND processing_duration IS NOT NULL
            """)
            avg_duration = cursor.fetchone()[0] or 0
            
            # Total processing time
            cursor.execute("""
                SELECT SUM(processing_duration) 
                FROM processed_files 
                WHERE status = 'success' AND processing_duration IS NOT NULL
            """)
            total_duration = cursor.fetchone()[0] or 0
            
            # Files processed today
            cursor.execute("""
                SELECT COUNT(*) FROM processed_files 
                WHERE status = 'success' 
                AND date(processed_at) = date('now')
            """)
            today_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_success': total_success,
                'total_failed': total_failed,
                'total_processing': total_processing,
                'avg_processing_duration': round(avg_duration, 2) if avg_duration else 0,
                'total_processing_duration': round(total_duration, 2) if total_duration else 0,
                'files_processed_today': today_count,
                'success_rate': round(total_success / (total_success + total_failed) * 100, 2) 
                    if (total_success + total_failed) > 0 else 0
            }
        except Exception as e:
            logging.error(f"Error getting statistics: {e}")
            return {}
    
    def cleanup_old_entries(self, days_to_keep: int = 90):
        """
        Remove old entries from the database.
        
        Args:
            days_to_keep: Number of days of history to keep
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM processed_files 
                WHERE processed_at < datetime('now', '-' || ? || ' days')
                AND status IN ('success', 'failed')
            """, (days_to_keep,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logging.info(f"Cleaned up {deleted_count} old entries from database")
        except Exception as e:
            logging.error(f"Error cleaning up old entries: {e}")

