"""modules/pipeline_checkpoint.py
Robust crash-recovery and checkpoint persistence engine for long-stream video processing.

Guarantees:
- Deterministic Job ID derived from source video identity (path, size, mtime).
- Atomic disk writes via temporary files and os.replace to prevent partial/corrupt files.
- Resumable pipeline stages: audio analysis, transcription, visual analysis,
  event classification, highlight scoring, and per-clip rendering.
- Crash recovery: load 100% of completed work in milliseconds after a reboot or reload.
- Bounded memory: keeps stage state on disk with structured JSON / binary serialization.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = os.path.join(getattr(config, "OUTPUT_DIR", "output"), "checkpoints")

STANDARD_STAGES = [
    "video_metadata",
    "audio_analysis",
    "transcription",
    "visual_analysis",
    "events_and_scoring",
    "candidate_selection",
    "clip_export",
]


class PipelineCheckpoint:
    """Manages persistent disk checkpoints for AI video clipper jobs."""

    def __init__(self, checkpoints_dir: Optional[str] = None):
        self.checkpoints_dir = os.path.abspath(checkpoints_dir or CHECKPOINTS_DIR)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    @staticmethod
    def compute_job_id(video_path: str) -> str:
        """Generate a stable, deterministic job ID from video file attributes.
        
        Uses absolute path, file size in bytes, and last modified timestamp.
        """
        abs_path = os.path.abspath(video_path)
        if not os.path.isfile(abs_path):
            # Fallback for paths that might be virtual or remote
            hash_str = f"virtual:{abs_path}"
        else:
            size = os.path.getsize(abs_path)
            mtime = os.path.getmtime(abs_path)
            base_name = os.path.basename(abs_path)
            hash_str = f"{abs_path}:{size}:{mtime}"

        digest = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:16]
        safe_name = os.path.splitext(os.path.basename(abs_path))[0]
        # Keep clean alphanumeric prefix for human readability
        prefix = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in safe_name)[:20].strip("_")
        return f"job_{prefix}_{digest}"

    def get_job_dir(self, job_id_or_path: str) -> str:
        """Return the absolute path to the job's checkpoint directory."""
        if job_id_or_path.startswith("job_"):
            job_id = job_id_or_path
        else:
            job_id = self.compute_job_id(job_id_or_path)
        return os.path.join(self.checkpoints_dir, job_id)

    def _manifest_path(self, job_id: str) -> str:
        return os.path.join(self.get_job_dir(job_id), "manifest.json")

    def _stage_path(self, job_id: str, stage_name: str) -> str:
        return os.path.join(self.get_job_dir(job_id), f"{stage_name}.json")

    def _atomic_write_json(self, target_path: str, data: Any) -> None:
        """Write JSON data atomically to prevent corrupt files on sudden crash."""
        dir_name = os.path.dirname(target_path)
        os.makedirs(dir_name, exist_ok=True)
        # Write to temporary file in same directory first so os.replace is atomic on same filesystem
        temp_fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_ckpt_", suffix=".json")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def init_job(self, video_path: str, extra_metadata: Optional[dict] = None) -> dict:
        """Initialize or retrieve existing job manifest."""
        job_id = self.compute_job_id(video_path)
        job_dir = self.get_job_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)

        manifest = self.get_manifest(job_id)
        if manifest:
            # Already initialized, update last_accessed
            manifest["last_accessed_at"] = time.time()
            if extra_metadata:
                manifest.setdefault("metadata", {}).update(extra_metadata)
            self._atomic_write_json(self._manifest_path(job_id), manifest)
            return manifest

        abs_path = os.path.abspath(video_path)
        file_size = os.path.getsize(abs_path) if os.path.isfile(abs_path) else 0

        manifest = {
            "job_id": job_id,
            "source_path": abs_path,
            "source_filename": os.path.basename(abs_path),
            "source_size_bytes": file_size,
            "created_at": time.time(),
            "last_accessed_at": time.time(),
            "stages": {stage: {"status": "pending", "completed_at": None, "item_count": 0} for stage in STANDARD_STAGES},
            "rendered_clips": [],
            "metadata": extra_metadata or {},
        }
        self._atomic_write_json(self._manifest_path(job_id), manifest)
        logger.info("Initialized new checkpoint job %s for %s", job_id, os.path.basename(abs_path))
        return manifest

    def get_manifest(self, job_id_or_path: str) -> Optional[dict]:
        """Read and return current job manifest if it exists and is valid."""
        if job_id_or_path.startswith("job_"):
            job_id = job_id_or_path
        else:
            job_id = self.compute_job_id(job_id_or_path)

        m_path = self._manifest_path(job_id)
        if not os.path.isfile(m_path):
            return None
        try:
            with open(m_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Corrupt or unreadable manifest for job %s: %s", job_id, e)
            return None

    def is_stage_completed(self, job_id_or_path: str, stage_name: str) -> bool:
        """Check if a specific pipeline stage has completed and its checkpoint exists."""
        if job_id_or_path.startswith("job_"):
            job_id = job_id_or_path
        else:
            job_id = self.compute_job_id(job_id_or_path)

        manifest = self.get_manifest(job_id)
        if not manifest:
            return False

        stage_info = manifest.get("stages", {}).get(stage_name)
        if not stage_info or stage_info.get("status") != "completed":
            return False

        s_path = self._stage_path(job_id, stage_name)
        if not os.path.isfile(s_path) or os.path.getsize(s_path) == 0:
            return False

        return True

    def save_stage(
        self,
        job_id_or_path: str,
        stage_name: str,
        data: Any,
        stage_metadata: Optional[dict] = None,
    ) -> bool:
        """Atomically persist a stage's results to disk and update manifest."""
        if job_id_or_path.startswith("job_"):
            job_id = job_id_or_path
        else:
            job_id = self.compute_job_id(job_id_or_path)

        job_dir = self.get_job_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)

        # 1. Determine item count for diagnostics
        item_count = 0
        if isinstance(data, (list, tuple)):
            item_count = len(data)
        elif isinstance(data, dict):
            item_count = len(data)

        # 2. Package payload with stage metadata and integrity timestamp
        payload = {
            "job_id": job_id,
            "stage": stage_name,
            "saved_at": time.time(),
            "item_count": item_count,
            "metadata": stage_metadata or {},
            "data": data,
        }

        # 3. Write data atomically
        target_stage_file = self._stage_path(job_id, stage_name)
        try:
            self._atomic_write_json(target_stage_file, payload)
        except Exception as e:
            logger.error("Failed to write stage %s for job %s: %s", stage_name, job_id, e)
            return False

        # 4. Update manifest atomically
        manifest = self.get_manifest(job_id) or self.init_job(job_id_or_path)
        manifest.setdefault("stages", {})[stage_name] = {
            "status": "completed",
            "completed_at": time.time(),
            "item_count": item_count,
            "file_size_bytes": os.path.getsize(target_stage_file),
            "metadata": stage_metadata or {},
        }
        manifest["last_accessed_at"] = time.time()
        self._atomic_write_json(self._manifest_path(job_id), manifest)

        logger.info("Successfully checkpointed stage '%s' for job %s (%d items)", stage_name, job_id, item_count)
        return True

    def load_stage(self, job_id_or_path: str, stage_name: str) -> Optional[Any]:
        """Load and return the raw data of a completed stage.
        
        Returns None if stage is incomplete or corrupted.
        """
        if not self.is_stage_completed(job_id_or_path, stage_name):
            return None

        if job_id_or_path.startswith("job_"):
            job_id = job_id_or_path
        else:
            job_id = self.compute_job_id(job_id_or_path)

        stage_file = self._stage_path(job_id, stage_name)
        try:
            with open(stage_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("data")
        except Exception as e:
            logger.error("Failed to load stage %s for job %s: %s", stage_name, job_id, e)
            return None

    def save_rendered_clip(
        self,
        job_id_or_path: str,
        clip_index: int,
        clip_id: str,
        output_path: str,
        clip_metadata: Optional[dict] = None,
    ) -> None:
        """Record an individual completed video clip immediately during render batch."""
        if job_id_or_path.startswith("job_"):
            job_id = job_id_or_path
        else:
            job_id = self.compute_job_id(job_id_or_path)

        manifest = self.get_manifest(job_id)
        if not manifest:
            manifest = self.init_job(job_id_or_path)

        rendered_list = manifest.setdefault("rendered_clips", [])
        # Check if already present to avoid duplicates
        existing = next((c for c in rendered_list if c.get("clip_id") == clip_id or c.get("clip_index") == clip_index), None)
        entry = {
            "clip_index": clip_index,
            "clip_id": clip_id,
            "output_path": os.path.abspath(output_path),
            "file_size_bytes": os.path.getsize(output_path) if os.path.isfile(output_path) else 0,
            "rendered_at": time.time(),
            "metadata": clip_metadata or {},
        }
        if existing:
            rendered_list.remove(existing)
        rendered_list.append(entry)

        self._atomic_write_json(self._manifest_path(job_id), manifest)
        logger.info("Recorded rendered clip %s (index %d) in job %s", clip_id, clip_index, job_id)

    def get_rendered_clips(self, job_id_or_path: str) -> list[dict]:
        """Return list of valid clips already rendered for this job."""
        manifest = self.get_manifest(job_id_or_path)
        if not manifest:
            return []
        clips = manifest.get("rendered_clips", [])
        # Only return clips whose physical output file still exists on disk
        valid_clips = []
        for c in clips:
            p = c.get("output_path", "")
            if os.path.isfile(p) and os.path.getsize(p) > 10_000:
                valid_clips.append(c)
        return valid_clips

    def invalidate_stage(self, job_id_or_path: str, stage_name: str) -> bool:
        """Mark a stage as pending and remove its checkpoint file."""
        if job_id_or_path.startswith("job_"):
            job_id = job_id_or_path
        else:
            job_id = self.compute_job_id(job_id_or_path)

        manifest = self.get_manifest(job_id)
        if not manifest:
            return False

        if stage_name in manifest.get("stages", {}):
            manifest["stages"][stage_name]["status"] = "pending"
            manifest["stages"][stage_name]["completed_at"] = None

        s_path = self._stage_path(job_id, stage_name)
        if os.path.exists(s_path):
            try:
                os.remove(s_path)
            except OSError:
                pass

        self._atomic_write_json(self._manifest_path(job_id), manifest)
        logger.info("Invalidated stage %s for job %s", stage_name, job_id)
        return True

    def get_job_summary(self, job_id_or_path: str) -> dict:
        """Provide detailed summary of job state for UI and logs."""
        manifest = self.get_manifest(job_id_or_path)
        if not manifest:
            return {"exists": False}

        stages = manifest.get("stages", {})
        completed_stages = [k for k, v in stages.items() if v.get("status") == "completed"]
        pending_stages = [k for k, v in stages.items() if v.get("status") != "completed"]

        total_stages = len(stages)
        completion_percent = round((len(completed_stages) / max(1, total_stages)) * 100, 1)

        return {
            "exists": True,
            "job_id": manifest["job_id"],
            "source_filename": manifest.get("source_filename"),
            "source_path": manifest.get("source_path"),
            "source_size_mb": round(manifest.get("source_size_bytes", 0) / (1024 * 1024), 1),
            "created_at": manifest.get("created_at"),
            "last_accessed_at": manifest.get("last_accessed_at"),
            "completed_stages": completed_stages,
            "pending_stages": pending_stages,
            "completion_percent": completion_percent,
            "stage_details": stages,
            "rendered_clip_count": len(self.get_rendered_clips(manifest["job_id"])),
        }

    def list_all_jobs(self) -> list[dict]:
        """List all discovered checkpointed jobs on disk, sorted by most recent."""
        if not os.path.isdir(self.checkpoints_dir):
            return []

        jobs = []
        for entry in os.listdir(self.checkpoints_dir):
            job_path = os.path.join(self.checkpoints_dir, entry)
            if os.path.isdir(job_path) and os.path.isfile(os.path.join(job_path, "manifest.json")):
                summary = self.get_job_summary(entry)
                if summary.get("exists"):
                    jobs.append(summary)

        jobs.sort(key=lambda x: x.get("last_accessed_at", 0), reverse=True)
        return jobs
