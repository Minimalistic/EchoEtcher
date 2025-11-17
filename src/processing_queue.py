import logging
import threading
import time
from queue import Queue, Empty
from pathlib import Path
from typing import Callable, Optional, Dict
from dataclasses import dataclass
from enum import Enum

class ProcessingStatus(Enum):
    """Status of a file in the processing queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingJob:
    """Represents a file processing job."""
    file_path: Path
    status: ProcessingStatus = ProcessingStatus.PENDING
    added_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.added_at is None:
            self.added_at = time.time()

class ProcessingQueue:
    """
    Thread-safe queue for processing audio files.
    Ensures only one file is processed at a time to prevent resource contention.
    """
    
    def __init__(self, max_workers: int = 1, max_queue_size: int = 100):
        """
        Initialize the processing queue.
        
        Args:
            max_workers: Maximum number of concurrent workers (default: 1)
            max_queue_size: Maximum number of jobs in queue
        """
        self.queue = Queue(maxsize=max_queue_size)
        self.max_workers = max_workers
        self.workers = []
        self.jobs: Dict[str, ProcessingJob] = {}  # Track jobs by file path
        self.jobs_lock = threading.Lock()
        self.is_running = False
        self.processor_callback: Optional[Callable] = None
        
        logging.info(f"Processing queue initialized (max_workers={max_workers}, max_queue_size={max_queue_size})")
    
    def set_processor(self, processor_callback: Callable):
        """
        Set the callback function to process files.
        
        Args:
            processor_callback: Function that takes a Path and processes it
        """
        self.processor_callback = processor_callback
    
    def add_file(self, file_path: Path) -> bool:
        """
        Add a file to the processing queue.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            True if added successfully, False if queue is full or file already queued
        """
        str_path = str(file_path)
        
        with self.jobs_lock:
            # Check if file is already in queue or being processed
            if str_path in self.jobs:
                job = self.jobs[str_path]
                if job.status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
                    logging.info(f"File already in queue (status: {job.status}): {file_path}")
                    return False
            
            # Create new job
            job = ProcessingJob(file_path=file_path)
            self.jobs[str_path] = job
        
        try:
            self.queue.put_nowait(job)
            logging.info(f"Added file to processing queue: {file_path}")
            return True
        except:
            with self.jobs_lock:
                self.jobs.pop(str_path, None)
            logging.warning(f"Queue is full, could not add file: {file_path}")
            return False
    
    def get_queue_size(self) -> int:
        """Get the current number of items in the queue."""
        return self.queue.qsize()
    
    def get_active_jobs(self) -> int:
        """Get the number of jobs currently being processed."""
        with self.jobs_lock:
            return sum(1 for job in self.jobs.values() 
                      if job.status == ProcessingStatus.PROCESSING)
    
    def get_job_status(self, file_path: Path) -> Optional[ProcessingStatus]:
        """
        Get the status of a job.
        
        Args:
            file_path: Path to the file
            
        Returns:
            ProcessingStatus or None if job not found
        """
        with self.jobs_lock:
            job = self.jobs.get(str(file_path))
            return job.status if job else None
    
    def start_workers(self):
        """Start worker threads to process the queue."""
        if self.is_running:
            logging.warning("Workers are already running")
            return
        
        if not self.processor_callback:
            raise ValueError("Processor callback must be set before starting workers")
        
        self.is_running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, name=f"ProcessingWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"Started {self.max_workers} processing worker(s)")
    
    def stop_workers(self, wait: bool = True, timeout: Optional[float] = None):
        """
        Stop worker threads.
        
        Args:
            wait: Whether to wait for workers to finish current jobs
            timeout: Maximum time to wait in seconds
        """
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Add None sentinels to wake up workers
        for _ in self.workers:
            try:
                self.queue.put_nowait(None)
            except:
                pass
        
        if wait:
            for worker in self.workers:
                worker.join(timeout=timeout)
        
        self.workers.clear()
        logging.info("Stopped processing workers")
    
    def _worker(self):
        """Worker thread that processes jobs from the queue."""
        while self.is_running:
            try:
                # Get job from queue with timeout
                try:
                    job = self.queue.get(timeout=1)
                except Empty:
                    continue
                
                # None is a sentinel to stop
                if job is None:
                    break
                
                # Process the job
                self._process_job(job)
                
                # Mark task as done
                self.queue.task_done()
                
            except Exception as e:
                logging.error(f"Error in worker thread: {e}", exc_info=True)
    
    def _process_job(self, job: ProcessingJob):
        """
        Process a single job.
        
        Args:
            job: The job to process
        """
        str_path = str(job.file_path)
        
        try:
            # Update job status
            with self.jobs_lock:
                job.status = ProcessingStatus.PROCESSING
                job.started_at = time.time()
            
            logging.info(f"Processing file: {job.file_path}")
            
            # Call the processor callback
            if self.processor_callback:
                self.processor_callback(job.file_path)
            
            # Mark as completed
            with self.jobs_lock:
                job.status = ProcessingStatus.COMPLETED
                job.completed_at = time.time()
                
                # Clean up old completed jobs (keep last 100)
                self._cleanup_old_jobs()
            
            processing_time = job.completed_at - job.started_at
            logging.info(f"Completed processing: {job.file_path} (took {processing_time:.1f}s)")
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing {job.file_path}: {error_msg}")
            
            with self.jobs_lock:
                job.status = ProcessingStatus.FAILED
                job.error = error_msg
                job.completed_at = time.time()
    
    def _cleanup_old_jobs(self, keep_count: int = 100):
        """
        Clean up old completed/failed jobs to prevent memory growth.
        
        Args:
            keep_count: Number of recent jobs to keep
        """
        with self.jobs_lock:
            # Get all completed/failed jobs sorted by completion time
            old_jobs = [
                (str_path, job) 
                for str_path, job in self.jobs.items()
                if job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
                and job.completed_at is not None
            ]
            
            # Sort by completion time (newest first)
            old_jobs.sort(key=lambda x: x[1].completed_at, reverse=True)
            
            # Remove jobs beyond keep_count
            if len(old_jobs) > keep_count:
                for str_path, _ in old_jobs[keep_count:]:
                    self.jobs.pop(str_path, None)
    
    def get_queue_stats(self) -> Dict:
        """
        Get statistics about the queue.
        
        Returns:
            Dictionary with queue statistics
        """
        with self.jobs_lock:
            pending = sum(1 for job in self.jobs.values() 
                         if job.status == ProcessingStatus.PENDING)
            processing = sum(1 for job in self.jobs.values() 
                           if job.status == ProcessingStatus.PROCESSING)
            completed = sum(1 for job in self.jobs.values() 
                          if job.status == ProcessingStatus.COMPLETED)
            failed = sum(1 for job in self.jobs.values() 
                        if job.status == ProcessingStatus.FAILED)
        
        return {
            'queue_size': self.queue.qsize(),
            'pending': pending,
            'processing': processing,
            'completed': completed,
            'failed': failed,
            'total_tracked': len(self.jobs)
        }

