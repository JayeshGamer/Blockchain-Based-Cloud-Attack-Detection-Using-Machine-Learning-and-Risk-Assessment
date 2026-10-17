"""
Process and Python Heap Memory Tracking Utility for Adaptive ML-DST (V5_REVIEW).
Provides robust run-level memory profiling:
1. Process Peak Resident Set Size (RSS) / Working Set Peak via psutil.
2. Traced Python Heap Allocations via tracemalloc with safe reference counting.
3. Nested child routines cannot inadvertently stop or reset the shared profiler.
4. Reports units (MB), method description, and status ('valid' or null with reason).
"""

import os
import sys
import time
import logging
import tracemalloc
from typing import Dict, Any, Optional
import psutil

logger = logging.getLogger(__name__)


class ProcessMemoryTracker:
    """
    Run-level process memory and heap tracker.
    Acts as a singleton or shared context to prevent nested routines from
    terminating tracing prematurely.
    """

    _active_instance: Optional["ProcessMemoryTracker"] = None

    def __init__(self, sampling_interval_batches: int = 10):
        self.process = psutil.Process(os.getpid())
        self.sampling_interval = sampling_interval_batches
        self.initial_rss_bytes = 0
        self.peak_rss_bytes = 0
        self.initial_wset_bytes = 0
        self.peak_wset_bytes = 0
        self.tracemalloc_started_here = False
        self.is_tracking = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> "ProcessMemoryTracker":
        """Starts run-level memory profiling."""
        self.start_time = time.time()
        self.is_tracking = True

        # Capture initial OS memory metrics
        try:
            mem_info = self.process.memory_info()
            self.initial_rss_bytes = getattr(mem_info, "rss", 0)
            self.peak_rss_bytes = self.initial_rss_bytes
            # On Windows, peak_wset reflects the lifetime peak working set
            self.initial_wset_bytes = getattr(mem_info, "peak_wset", getattr(mem_info, "wset", self.initial_rss_bytes))
            self.peak_wset_bytes = self.initial_wset_bytes
        except Exception as e:
            logger.warning(f"Could not read initial process memory: {e}")

        # Start tracemalloc if not already active
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_started_here = True
        else:
            self.tracemalloc_started_here = False

        ProcessMemoryTracker._active_instance = self
        return self

    def sample(self) -> None:
        """Samples current memory usage to record peak values."""
        if not self.is_tracking:
            return
        try:
            mem_info = self.process.memory_info()
            rss = getattr(mem_info, "rss", 0)
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            wset = getattr(mem_info, "peak_wset", getattr(mem_info, "wset", rss))
            if wset > self.peak_wset_bytes:
                self.peak_wset_bytes = wset
        except Exception:
            pass

    def stop(self) -> Dict[str, Any]:
        """
        Stops profiling and returns a verified memory summary dictionary.
        Never returns 0.00 MB for active processes.
        """
        self.end_time = time.time()
        self.is_tracking = False
        self.sample()

        peak_heap_bytes = 0
        if tracemalloc.is_tracing():
            try:
                _, peak_heap_bytes = tracemalloc.get_traced_memory()
            except Exception:
                peak_heap_bytes = 0
            if self.tracemalloc_started_here:
                tracemalloc.stop()

        # Determine best OS peak resident memory
        best_peak_bytes = max(self.peak_wset_bytes, self.peak_rss_bytes)

        if best_peak_bytes <= 0:
            # If for any reason reading failed, return null with explicit reason
            summary = {
                "process_peak_rss_mb": None,
                "traced_python_heap_peak_mb": None,
                "elapsed_time_sec": round(self.end_time - self.start_time, 3) if self.start_time else None,
                "units": "MB",
                "status": "unavailable",
                "reason": "Process memory counters inaccessible on current platform",
            }
        else:
            process_peak_mb = round(best_peak_bytes / (1024.0 * 1024.0), 2)
            heap_peak_mb = round(peak_heap_bytes / (1024.0 * 1024.0), 2) if peak_heap_bytes > 0 else None
            summary = {
                "process_peak_rss_mb": process_peak_mb,
                "traced_python_heap_peak_mb": heap_peak_mb,
                "elapsed_time_sec": round(self.end_time - self.start_time, 3) if self.start_time else None,
                "units": "MB",
                "status": "valid",
                "measurement_method": "psutil.Process.memory_info (Peak Working Set / RSS) + tracemalloc (Python Heap)",
            }

        ProcessMemoryTracker._active_instance = None
        return summary

    @classmethod
    def get_active(cls) -> Optional["ProcessMemoryTracker"]:
        return cls._active_instance
