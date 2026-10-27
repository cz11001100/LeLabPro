# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import contextlib
import glob
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

# Import our custom recording functionality
from . import datasets as dataset_browser, record as _record
from .datasets import analyze_dataset

# Import our custom calibration functionality
from .calibrate import CalibrationRequest, calibration_manager
from .jobs import (
    JobAlreadyRunningError,
    JobNotFoundError,
    JobNotRunningError,
    JobTarget,
    job_registry,
)
from .record import (
    DatasetInfoRequest,
    DeleteEpisodeRequest,
    RecordingRequest,
    UploadRequest,
    get_upload_progress,
    handle_check_dataset_health,
    handle_delete_dataset,
    handle_delete_episode,
    handle_exit_early,
    handle_get_dataset_info,
    handle_recording_status,
    handle_repair_dataset,
    handle_rerecord_episode,
    handle_start_recording,
    handle_stop_recording,
    handle_upload_dataset,
)
from .rollout import (
    InferenceRequest,
    TakeoverRequest,
    handle_inference_end_round,
    handle_inference_snapshot,
    handle_inference_status,
    handle_inference_takeover,
    handle_start_inference,
    handle_stop_inference,
    inference_camera_frames,
)
from .rollout_eval import (
    RolloutScoreRequest,
    apply_intervention_exemption,
    handle_compare_progress,
    handle_rollout_score,
    handle_rollout_score_status,
    handle_rollout_summary,
    list_capture_datasets,
    promote_progress,
)
from .dataset_merge import MergeRequest, handle_merge, handle_merge_status, inspect_merge_sources
from .ssh_remote_env import sarm_training_status as _sarm_training_status
from .ssh_remote_env import start_sarm_training_remote as _start_sarm_training

# Import our custom teleoperation functionality
from .teleoperate import (
    TeleoperateRequest,
    UpdateEmaRequest,
    handle_get_joint_positions,
    handle_start_teleoperation,
    handle_stop_teleoperation,
    handle_teleoperation_status,
    handle_update_ema,
)

# Training is now job-based; see app/jobs.py.
from .train import TrainingRequest
from .update import handle_run_update, handle_update_check
from .utils import config
from .utils.config import (
    FOLLOWER_CONFIG_PATH,
    LEADER_CONFIG_PATH,
    delete_robot_record,
    detect_port_after_disconnect,
    find_available_ports,
    find_robot_port,
    get_default_robot_port,
    get_robot_record,
    get_saved_robot_port,
    is_robot_record_clean,
    is_valid_robot_name,
    list_robot_records,
    save_robot_port,
    save_robot_record,
)
from .utils.hf_auth import cached_whoami, handle_hf_auth_status, handle_hf_login, shared_hf_api
from .utils.system import (
    handle_get_cuda_status,
    handle_get_policy_extra,
    handle_get_training_extra,
    handle_get_wandb_extra,
    handle_install_policy_extra,
    handle_install_policy_extra_status,
    handle_install_training_extra,
    handle_install_training_extra_status,
    handle_install_wandb_extra,
    handle_install_wandb_extra_status,
    warn_if_cuda_mismatch,
)

# SSH runner config management
from .ssh_config import (
    SSHConfigPublic,
    delete_ssh_config,
    list_ssh_configs,
    save_ssh_config,
    test_ssh_connection,
)
from .ssh_remote_env import check_remote_environment as _check_remote_env
from .ssh_remote_env import install_lerobot_remote as _install_lerobot
from .ssh_remote_env import install_lerobot_remote_streaming as _install_lerobot_stream
from .ssh_remote_env import reward_model_status as _reward_model_status
from .ssh_remote_env import start_reward_model_download as _start_reward_model_download
from .task_tracker import task_tracker
from .training_monitor import analyze_training_health

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StartTrainingBody(BaseModel):
    """Wrapping body for POST /jobs/training. Adds optional target spec."""

    config: TrainingRequest
    target: JobTarget | None = None

    @classmethod
    def from_legacy(cls, raw: dict) -> "StartTrainingBody":
        """Accept the old request shape (TrainingRequest fields at top level)
        as well as the new shape ({config: ..., target: ...}).
        """
        if "config" in raw and isinstance(raw["config"], dict):
            return cls.model_validate(raw)
        # Legacy: top-level training fields, no target.
        return cls(config=TrainingRequest.model_validate(raw))


# Cache for HF Jobs hardware flavors (5-minute TTL)
_flavors_cache: dict = {"data": None, "fetched_at": 0.0}
_FLAVOR_CACHE_TTL_SECONDS = 300.0


app = FastAPI()

# In dev mode the React app runs on :8080 while the API runs on :8000; in
# prod they share an origin and CORS is unnecessary. allow_credentials with
# a wildcard origin is rejected by browsers, so we drop it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

# Get the path to the lerobot root directory (3 levels up from this script)
LEROBOT_PATH = str(Path(__file__).parent.parent.parent.parent)
logger.info(f"LeRobot path: {LEROBOT_PATH}")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.broadcast_queue = queue.Queue()
        self.broadcast_thread = None
        self.is_running = False
        # Guards `active_connections` since the broadcast worker thread also
        # mutates it on send failure.
        self._connections_lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._connections_lock:
            self.active_connections.append(websocket)
            count = len(self.active_connections)
        logger.info(f"WebSocket connected. Total connections: {count}")

        if not self.is_running:
            self.start_broadcast_thread()

    def disconnect(self, websocket: WebSocket):
        with self._connections_lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                count = len(self.active_connections)
                logger.info(f"WebSocket disconnected. Total connections: {count}")
            else:
                count = len(self.active_connections)

        if count == 0 and self.is_running:
            self.stop_broadcast_thread()

    def start_broadcast_thread(self):
        """Start the background thread for broadcasting data"""
        if self.is_running:
            return

        self.is_running = True
        self.broadcast_thread = threading.Thread(target=self._broadcast_worker, daemon=True)
        self.broadcast_thread.start()
        logger.info("📡 Broadcast thread started")

    def stop_broadcast_thread(self):
        """Stop the background thread"""
        self.is_running = False
        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=1.0)
            logger.info("📡 Broadcast thread stopped")

    def _broadcast_worker(self):
        """Background worker thread for broadcasting WebSocket data"""
        import asyncio

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.is_running:
                try:
                    # Get data from queue with timeout
                    data = self.broadcast_queue.get(timeout=0.1)
                    if data is None:  # Poison pill to stop
                        break

                    # Broadcast to all connections
                    if self.active_connections:
                        loop.run_until_complete(self._send_to_all_connections(data))

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in broadcast worker: {e}")

        finally:
            loop.close()

    async def _send_to_all_connections(self, data: dict[str, Any]):
        """Send data to all active WebSocket connections"""
        with self._connections_lock:
            connections = list(self.active_connections)
        if not connections:
            return

        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error sending data to WebSocket: {e}")
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

    def broadcast_joint_data_sync(self, data: dict[str, Any]):
        """Thread-safe method to queue data for broadcasting"""
        if self.is_running and self.active_connections:
            try:
                self.broadcast_queue.put_nowait(data)
            except queue.Full:
                logger.warning("Broadcast queue is full, dropping data")

    def notify_jobs_changed(self) -> None:
        """Push a 'jobs_changed' event to all WS clients so they refetch.

        Called from JobRegistry on submit / watchdog finalisation / delete.
        Skipped silently if no clients are connected — the frontend does an
        initial fetch on mount, so a missed broadcast is self-healing.
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait({"type": "jobs_changed", "timestamp": time.time()})

    def notify_job_progress(self, snapshots: list[dict]) -> None:
        """Push a 'job_progress' event with per-running-job snapshots.

        Fired from the JobRegistry watchdog (~1Hz) while jobs are running so
        the dashboard's progress bar updates live without refetching /jobs
        (let alone /jobs/hub, which hits the HF API on every call).
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait(
                    {"type": "job_progress", "jobs": snapshots, "timestamp": time.time()}
                )


manager = ConnectionManager()
job_registry.set_on_change(manager.notify_jobs_changed)
job_registry.set_on_progress(manager.notify_job_progress)


@app.get("/get-configs")
def get_configs():
    # Get all available calibration configs
    leader_configs = [os.path.basename(f) for f in glob.glob(os.path.join(LEADER_CONFIG_PATH, "*.json"))]
    follower_configs = [os.path.basename(f) for f in glob.glob(os.path.join(FOLLOWER_CONFIG_PATH, "*.json"))]

    return {"leader_configs": leader_configs, "follower_configs": follower_configs}


@app.post("/move-arm")
def teleoperate_arm(request: TeleoperateRequest):
    """Start teleoperation of the robot arm"""
    return handle_start_teleoperation(request, manager)


@app.post("/stop-teleoperation")
def stop_teleoperation():
    """Stop the current teleoperation session"""
    return handle_stop_teleoperation()


@app.get("/teleoperation-status")
def teleoperation_status():
    """Get the current teleoperation status"""
    return handle_teleoperation_status()


@app.post("/update-ema")
def update_ema(request: UpdateEmaRequest):
    """Update EMA alpha for the active teleoperation session"""
    return handle_update_ema(request)


@app.get("/joint-positions")
def get_joint_positions():
    """Get current robot joint positions"""
    return handle_get_joint_positions()


@app.post("/start-inference")
def start_inference(request: InferenceRequest):
    result = handle_start_inference(request)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start inference"),
        )
    return result


@app.post("/stop-inference")
def stop_inference():
    result = handle_stop_inference()
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to stop inference"),
        )
    return result


@app.get("/inference-status")
def inference_status():
    return handle_inference_status()


@app.post("/inference-snapshot")
def inference_snapshot():
    """Capture the last ~30 ticks of observations + actions for failure diagnosis."""
    result = handle_inference_snapshot()
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Snapshot failed"),
        )
    return result


class EndRoundBody(BaseModel):
    advance: bool = True


@app.post("/inference-end-round")
def inference_end_round(body: EndRoundBody):
    """End this trial (graceful: finalise data, arm returns home) and start the
    next one — the rollout twin of the recording page's exit-early."""
    result = handle_inference_end_round(advance=body.advance)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "End round failed"),
        )
    return result


@app.post("/inference-takeover")
def inference_takeover(request: TakeoverRequest):
    """Drive the running data-collection strategy (pause / correct / stop)."""
    result = handle_inference_takeover(request.action)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Takeover failed"),
        )
    return result


@app.get("/rollout-datasets")
def rollout_datasets():
    """Local datasets holding policy rollouts, with scoring status."""
    return {"datasets": list_capture_datasets()}


@app.get("/rollout-summary")
def rollout_summary(repo_id: str):
    """Closed-loop success rate and per-episode progress for a rollout dataset."""
    return handle_rollout_summary(repo_id)


@app.post("/rollout-score")
def rollout_score(request: RolloutScoreRequest):
    """Score a rollout dataset with the progress model (writes sarm_progress.parquet)."""
    result = handle_rollout_score(request)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Scoring failed"),
        )
    return result


@app.get("/rollout-score-status")
def rollout_score_status():
    return handle_rollout_score_status()


class RolloutRepoBody(BaseModel):
    repo_id: str


class RolloutBackendBody(BaseModel):
    repo_id: str
    backend: str = "robometer"


class SSHConfigIdBody(BaseModel):
    config_id: str


class SarmTrainBody(BaseModel):
    config_id: str
    dataset_repo_id: str
    image_key: str = "observation.images.up"
    steps: int = 5000
    batch_size: int = 32
    push_repo_id: str = ""
    use_cn_mirror: bool = True


@app.get("/rollout-compare")
def rollout_compare(repo_id: str):
    """Agreement between the two scoring backends on the same episodes."""
    return handle_compare_progress(repo_id)


@app.post("/rollout-promote-progress")
def rollout_promote_progress(body: RolloutBackendBody):
    """Pick which backend's progress file training and the summary use."""
    result = promote_progress(body.repo_id, body.backend)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Promote failed"),
        )
    return result


@app.post("/ssh-configs/sarm-train")
def ssh_sarm_train(body: SarmTrainBody):
    """Train a task-specific SARM scorer on the pod (the second scoring backend)."""
    return _start_sarm_training(
        body.config_id,
        body.dataset_repo_id,
        image_key=body.image_key,
        steps=body.steps,
        batch_size=body.batch_size,
        push_repo_id=body.push_repo_id,
        use_cn_mirror=body.use_cn_mirror,
    )


@app.post("/ssh-configs/sarm-train-status")
def ssh_sarm_train_status(body: SSHConfigIdBody):
    return _sarm_training_status(body.config_id)


@app.post("/rollout-exempt-interventions")
def rollout_exempt_interventions(body: RolloutRepoBody):
    """Lift correction windows in the progress file so RA-BC doesn't zero them."""
    result = apply_intervention_exemption(body.repo_id)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Exemption failed"),
        )
    return result


@app.get("/inference-camera-feed/{cam_key}")
def inference_camera_feed(cam_key: str):
    """Live MJPEG preview of one camera during an active inference run.

    Frames come from the preview files the rollout subprocess publishes; the
    server only opens the cameras itself if those files go silent, since a
    second open steals the device from the rollout on this machine.
    """
    from . import rollout as _rollout

    if not _rollout.inference_active:
        raise HTTPException(status_code=409, detail="No active inference session")
    return StreamingResponse(
        inference_camera_frames(cam_key),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify server is running"""
    return {"status": "ok", "message": "FastAPI server is running"}


@app.get("/hf-auth-status")
def hf_auth_status():
    """Check whether the local HF CLI is authenticated and return user info."""
    return handle_hf_auth_status()


class HfLoginBody(BaseModel):
    token: str


@app.post("/hf-auth/login")
def hf_auth_login(body: HfLoginBody):
    """Persist a pasted HF token (validated against whoami) for this user."""
    try:
        return handle_hf_login(body.token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/datasets")
def datasets_list():
    """List datasets available to the user — Hub-owned + local cache.

    Each entry carries a `source` field: "local", "hub", or "both".
    """
    return dataset_browser.list_all_datasets()


@app.get("/ws-test")
def websocket_test():
    """Test endpoint to verify WebSocket support"""
    return {"websocket_endpoint": "/ws/joint-data", "status": "available"}


@app.websocket("/ws/joint-data")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("🔗 New WebSocket connection attempt")
    try:
        await manager.connect(websocket)
        logger.info("✅ WebSocket connection established")

        while True:
            # Keep the connection alive and wait for messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Handle any incoming messages if needed
                logger.debug(f"Received WebSocket message: {data}")
            except TimeoutError:
                # No message received, continue
                pass
            except WebSocketDisconnect:
                logger.info("🔌 WebSocket client disconnected")
                break

            # Small delay to prevent excessive CPU usage
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        logger.info("🧹 WebSocket connection cleaned up")


@app.post("/start-recording")
def start_recording(request: RecordingRequest):
    """Start a dataset recording session"""
    result = handle_start_recording(request)
    if result.get("success"):
        task_tracker.create(
            task_id="recording",
            task_type="recording",
            label="Recording in progress",
            metadata={"dataset_repo_id": request.dataset_repo_id},
        )
    return result


@app.post("/stop-recording")
def stop_recording():
    """Stop the current recording session"""
    result = handle_stop_recording()
    task_tracker.update("recording", status="done", message="Recording completed")
    return result


@app.get("/recording-status")
def recording_status():
    """Get the current recording status"""
    result = handle_recording_status()
    if result.get("recording_active"):
        task = task_tracker.get("recording")
        if task is None:
            task_tracker.create(
                task_id="recording",
                task_type="recording",
                label="Recording in progress",
                metadata={"dataset_repo_id": result.get("dataset_repo_id", "")},
            )
        else:
            task_tracker.update(
                "recording",
                label=f"Recording episode {result.get('current_episode', '?')}/{result.get('total_episodes', '?')}",
                metadata={
                    "current_episode": result.get("current_episode"),
                    "total_episodes": result.get("total_episodes"),
                    "saved_episodes": result.get("saved_episodes"),
                },
            )
    elif result.get("session_ended"):
        task = task_tracker.get("recording")
        if task and task.status == "running":
            status = "failed" if result.get("current_phase") == "error" else "done"
            task_tracker.update("recording", status=status, message=result.get("message", ""))
    return result


@app.get("/camera-feed/{cam_key}")
def camera_feed(cam_key: str):
    """Live MJPEG preview of one configured camera during an active recording.

    Browsers render `multipart/x-mixed-replace` directly in an <img>, so the
    frontend just points an <img> at this URL. Only valid while a session is
    active; the generator ends itself when recording stops.
    """
    if not _record.recording_active:
        raise HTTPException(status_code=409, detail="No active recording session")
    return StreamingResponse(
        _record.camera_feed_frames(cam_key),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# A stream URL carries `user:password@`. Registering it once and addressing the
# preview by token keeps those credentials out of `<img src>`, access logs and
# browser history.
_network_preview_sources: dict[str, tuple[str, float]] = {}
_NETWORK_PREVIEW_TTL_S = 1800.0


class NetworkCameraSource(BaseModel):
    url: str


def _remember_preview_source(url: str) -> str:
    key = os.urandom(8).hex()
    now = time.time()
    for stale, (_, seen) in list(_network_preview_sources.items()):
        if now - seen > _NETWORK_PREVIEW_TTL_S:
            _network_preview_sources.pop(stale, None)
    _network_preview_sources[key] = (url, now)
    return key


@app.post("/network-camera-source")
def register_network_camera_source(body: NetworkCameraSource):
    """Validate a stream URL and report what it sends, for the camera picker.

    The resolution and frame rate are the *stream's*, not whatever was typed into
    the form: they become the dataset's image features, and an IP camera won't
    switch to a resolution it wasn't configured to encode.
    """
    from .network_camera import NetworkCameraError, normalize_url, probe, redact

    try:
        url = normalize_url(body.url)
    except NetworkCameraError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        stream = probe(url)
    except NetworkCameraError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "key": _remember_preview_source(url),
        "url": redact(url),
        **stream,
    }


@app.get("/network-camera-feed")
def network_camera_feed(key: str):
    """Live MJPEG preview of a registered stream URL.

    Independent of any session: RTSP takes several client sessions, so the picker
    can watch a camera while a recording or inference run holds its own copy.
    """
    from .network_camera import preview_frames

    source = _network_preview_sources.get(key)
    if source is None:
        raise HTTPException(status_code=404, detail="Unknown preview key — re-register the camera URL")
    _network_preview_sources[key] = (source[0], time.time())
    return StreamingResponse(
        preview_frames(source[0]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/recording-exit-early")
def recording_exit_early():
    """Skip to next episode (replaces right arrow key)"""
    return handle_exit_early()


@app.post("/recording-rerecord-episode")
def recording_rerecord_episode():
    """Re-record current episode (replaces left arrow key)"""
    return handle_rerecord_episode()


@app.post("/upload-dataset")
def upload_dataset(request: UploadRequest):
    """Upload dataset to HuggingFace Hub"""
    return handle_upload_dataset(request)


@app.get("/upload-progress")
def upload_progress_sse():
    """SSE endpoint for upload progress updates."""
    import time

    async def event_generator():
        last_data = None
        while True:
            data = get_upload_progress()
            if data != last_data:
                yield f"data: {json.dumps(data)}\n\n"
                last_data = data
                if data.get("completed") or data.get("error"):
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/upload-progress/current")
def upload_progress_current():
    """Get current upload progress (non-streaming)."""
    return get_upload_progress()


@app.post("/dataset-info")
def get_dataset_info(request: DatasetInfoRequest):
    """Get information about a saved dataset"""
    return handle_get_dataset_info(request)


@app.post("/dataset-statistics")
def get_dataset_statistics(request: DatasetInfoRequest):
    """Get dataset statistics and anomaly detection results"""
    return analyze_dataset(request.dataset_repo_id)


@app.post("/delete-dataset")
def delete_dataset(request: DatasetInfoRequest):
    """Remove a recorded dataset directory from local disk."""
    return handle_delete_dataset(request)


@app.post("/delete-episode")
def delete_episode(request: DeleteEpisodeRequest):
    """Remove a single episode from a local dataset."""
    return handle_delete_episode(request)


class MergeCheckBody(BaseModel):
    sources: list[str]


@app.post("/dataset-merge-check")
def dataset_merge_check(body: MergeCheckBody):
    """Report each source's facts and what would block merging them."""
    return inspect_merge_sources(body.sources)


@app.post("/dataset-merge")
def dataset_merge(request: MergeRequest):
    """Merge local datasets into one training set (lerobot can't train on many)."""
    result = handle_merge(request)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Merge failed"),
        )
    return result


@app.get("/dataset-merge-status")
def dataset_merge_status():
    return handle_merge_status()


@app.post("/check-dataset-health")
def check_dataset_health(request: DatasetInfoRequest):
    """Check if a dataset is healthy or corrupted."""
    return handle_check_dataset_health(request)


@app.post("/repair-dataset")
def repair_dataset_endpoint(request: DatasetInfoRequest):
    """Start repairing a corrupted dataset in background."""
    return handle_repair_dataset(request)


@app.get("/repair-progress")
def repair_progress_sse():
    """SSE endpoint for repair progress updates."""

    async def event_generator():
        from .dataset_repair import get_repair_progress
        last_data = None
        while True:
            data = get_repair_progress()
            if data != last_data:
                yield f"data: {json.dumps(data)}\n\n"
                last_data = data
                if data.get("completed") or data.get("error"):
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/repair-progress/current")
def repair_progress_current():
    """Get current repair progress (non-streaming)."""
    from .dataset_repair import get_repair_progress
    return get_repair_progress()


# ============================================================================
# JOB ENDPOINTS
# ============================================================================


@app.post("/jobs/training", status_code=201)
async def create_training_job(req: Request):
    raw = await req.json()
    body = StartTrainingBody.from_legacy(raw)
    try:
        record = job_registry.start(body.config, body.target)
    except JobAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job already running: {exc}") from exc
    except ValueError as exc:
        # e.g. "flavor is required when runner is hf_cloud"
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record


class ImportModelRequest(BaseModel):
    source: str
    name: str | None = None


@app.post("/jobs/import", status_code=201)
def import_model(body: ImportModelRequest):
    """Register an external model (local dir or HF repo) as a pseudo-job."""
    try:
        return job_registry.register_imported(body.source, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs")
def list_jobs(limit: int = 10):
    return {"jobs": job_registry.list(limit=limit)}


@app.get("/jobs/hub")
def list_hub_jobs():
    """List the user's HF Cloud compute Jobs and their uploaded LeRobot model
    repos on huggingface.co.

    Returns 200 with empty lists when no token is configured so the frontend
    can render an unauthenticated empty state without surfacing an error.

    Declared before `/jobs/{job_id}` so FastAPI's first-match routing doesn't
    treat "hub" as a job id.
    """
    info = cached_whoami()
    if info is None:
        return {"authenticated": False, "jobs": [], "models": []}
    api = shared_hf_api()

    authors: list[str] = []
    if info.get("name"):
        authors.append(info["name"])
    for o in info.get("orgs", []) or []:
        if isinstance(o, dict) and o.get("name"):
            authors.append(o["name"])

    try:
        jobs = api.list_jobs()
    except Exception as exc:
        logger.warning("list_jobs failed: %s", exc)
        jobs = []

    seen_models: set[str] = set()
    models: list[dict] = []
    for author in authors:
        try:
            for m in api.list_models(author=author, filter="LeRobot", limit=200):
                if m.id in seen_models:
                    continue
                seen_models.add(m.id)
                models.append(
                    {
                        "repo_id": m.id,
                        "last_modified": m.last_modified.isoformat() if m.last_modified else None,
                        "private": bool(getattr(m, "private", False)),
                    }
                )
        except Exception as exc:
            logger.warning("list_models(%s) failed: %s", author, exc)
    models.sort(key=lambda m: m["last_modified"] or "", reverse=True)

    return {
        "authenticated": True,
        "jobs": [
            {
                "id": ji.id,
                "created_at": ji.created_at.isoformat() if ji.created_at else None,
                "docker_image": ji.docker_image,
                "space_id": ji.space_id,
                "flavor": ji.flavor,
                "status": ({"stage": ji.status.stage, "message": ji.status.message} if ji.status else None),
                "owner": ji.owner.name if ji.owner else None,
                "url": ji.url,
            }
            for ji in jobs
        ],
        "models": models,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return job_registry.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    try:
        logs = job_registry.drain_logs(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    return {"logs": logs}


@app.get("/jobs/{job_id}/log-file")
def get_job_log_file(job_id: str):
    """Return the entire on-disk log file for a job. Drains the live queue too
    so the next /logs poll returns only lines that arrived after this call."""
    try:
        logs = job_registry.read_persisted_logs(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    # Best-effort drain so the frontend doesn't double-display.
    with contextlib.suppress(JobNotFoundError):
        job_registry.drain_logs(job_id)
    return {"logs": logs}


@app.get("/jobs/{job_id}/metrics-history")
def get_job_metrics_history(job_id: str):
    """Return the per-step loss/lr/grad-norm series reconstructed from the
    job's log.jsonl. Used to seed the monitoring charts so curves persist
    across page reloads, navigation, and lelab restarts."""
    try:
        points = job_registry.read_metrics_history(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    return {"points": points}


@app.get("/jobs/{job_id}/training-health")
def get_job_training_health(job_id: str):
    """Analyze training metrics and return a health report with overfitting detection.

    Returns status (healthy/warning/critical), loss trend, gradient health,
    convergence progress, overfitting risk, and actionable recommendations.
    """
    try:
        points = job_registry.read_metrics_history(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc

    if not points:
        return {
            "status": "healthy",
            "loss_trend": "insufficient_data",
            "gradient_health": "insufficient_data",
            "convergence_progress": 0.0,
            "overfitting_risk": "low",
            "recommendations": ["Waiting for training data"],
            "details": {},
        }

    # Get total steps from job record
    try:
        record = job_registry.get(job_id)
        total_steps = record.config.steps
    except Exception:
        total_steps = points[-1].step * 2 if points else 10000

    steps = [p.step for p in points]
    losses = [p.loss for p in points]
    grad_norms = [p.grad_norm for p in points]
    lrs = [p.lr for p in points]

    report = analyze_training_health(steps, losses, grad_norms, lrs, total_steps)

    return {
        "status": report.status,
        "loss_trend": report.loss_trend,
        "gradient_health": report.gradient_health,
        "convergence_progress": report.convergence_progress,
        "overfitting_risk": report.overfitting_risk,
        "recommendations": report.recommendations,
        "details": report.details,
    }


@app.get("/jobs/{job_id}/checkpoints")
def get_job_checkpoints(job_id: str):
    """List the checkpoints saved for this job, ascending by step."""
    try:
        return {"checkpoints": job_registry.list_checkpoints(job_id)}
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/checkpoints/{step}/policy-config")
def get_checkpoint_policy_config(job_id: str, step: int):
    """Return the UX-relevant slice of a checkpoint's pretrained_model config:
    policy_type, image_features (per-camera height/width), and requires_task."""
    try:
        return job_registry.get_policy_config_summary(job_id, step)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/checkpoints/{step}/model-info")
def get_checkpoint_model_info(job_id: str, step: int):
    """Static architecture + training-hyperparameter metadata for the detail page."""
    try:
        return job_registry.get_model_info(job_id, step)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/export-checkpoint/{step}")
def export_checkpoint(job_id: str, step: int):
    """Download a checkpoint from the SSH remote to local via SFTP.

    Returns the newly created imported job record.
    """
    try:
        imported = job_registry.export_checkpoint(job_id, step)
        return imported.model_dump()
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/export-progress/{step}")
def get_export_progress(job_id: str, step: int):
    """Byte-level SFTP progress for a checkpoint export; polled while the
    blocking export POST is still in flight."""
    return job_registry.get_export_progress(job_id, step)


@app.post("/jobs/{job_id}/export-cancel/{step}")
def cancel_export_checkpoint(job_id: str, step: int):
    """Request cancellation of an in-flight checkpoint download."""
    return job_registry.cancel_export(job_id, step)


class ValidationOptions(BaseModel):
    """Sampling knobs for the offline evaluator. Omit the body for the defaults."""

    step: int | None = None
    num_batches: int = 20
    batch_size: int = 8
    seed: int = 0
    holdout_episodes: int = 0
    frames_per_episode: int = 0
    episodes: list[int] | None = None
    dataset_repo_id: str | None = None
    probe: bool = True
    saliency_frames: int = 2
    aim_frames: int = 4
    device: str = "cuda"


@app.post("/jobs/{job_id}/validate")
def start_job_validation(job_id: str, options: ValidationOptions | None = None):
    """Start the offline evaluation for a completed or imported job.

    Runs the open-loop action-error pass plus the vision/language sensitivity
    probe; results land in the job's ``validation_results``.
    """
    try:
        return job_registry.start_validation(job_id, options.model_dump() if options else None)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/validate/stop")
def stop_job_validation(job_id: str):
    """Cancel a running offline evaluation for a job.

    Sets the worker's cancel flag; the evaluation aborts at its next batch
    boundary (SSH runs also kill the remote process) and the job returns to the
    idle ``none`` validation state.
    """
    try:
        return job_registry.stop_validation(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/evaluation/artifact")
def get_evaluation_artifact(job_id: str, path: str):
    """Serve one evaluator chart PNG (paths are relative to the job directory)."""
    try:
        artifact = job_registry.get_evaluation_artifact(job_id, path)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(artifact, media_type="image/png")


@app.get("/jobs/{job_id}/validation")
def get_job_validation(job_id: str):
    """Return validation state and results for a job."""
    try:
        return job_registry.get_validation(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        return job_registry.stop(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is not running") from exc


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str):
    """Resume a training job with the same config, continuing from its latest checkpoint."""
    try:
        original = job_registry.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc

    if original.runner not in ("ssh", "local"):
        raise HTTPException(status_code=400, detail="Resume is only supported for SSH and local jobs")

    # Clone the original config and set resume_from_job_id
    new_config = original.config.model_copy()
    new_config.resume_from_job_id = job_id
    new_config.resume = True

    target = JobTarget(runner=original.runner, ssh_config_id=original.ssh_config_id)
    try:
        record = job_registry.start(new_config, target)
        return record.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        job_registry.delete(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is running; stop it first") from exc


@app.get("/jobs/runners/hardware")
def get_runners_hardware():
    """Return HF Jobs flavor catalog + auth state for the TargetCard.

    Both the flavors list and the whoami result are cached in-process to
    keep this endpoint cheap (it can be re-fetched whenever auth state
    changes). The whoami cache is invalidated on login.
    """
    info = cached_whoami()
    if info is None or not info.get("name"):
        return {"authenticated": False, "username": None, "flavors": []}
    username: str = info["name"]
    api = shared_hf_api()

    now = time.time()
    if _flavors_cache["data"] is None or now - _flavors_cache["fetched_at"] > _FLAVOR_CACHE_TTL_SECONDS:
        try:
            hw_list = api.list_jobs_hardware()
        except Exception as exc:
            logger.warning("list_jobs_hardware failed: %s", exc)
            return {"authenticated": True, "username": username, "flavors": []}
        _flavors_cache["data"] = [
            {
                "name": h.name,
                "pretty_name": h.pretty_name,
                "cpu": h.cpu,
                "ram": h.ram,
                "accelerator": h.accelerator,
                "unit_cost_usd": h.unit_cost_usd,
                "unit_label": h.unit_label,
            }
            for h in hw_list
        ]
        _flavors_cache["fetched_at"] = now

    return {
        "authenticated": True,
        "username": username,
        "flavors": _flavors_cache["data"],
    }


# ============================================================================
# SSH CONFIG ENDPOINTS
# ============================================================================


class SSHConfigBody(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str
    password: str | None = None
    key_path: str | None = None
    remote_python: str = "python"
    remote_workdir: str | None = None
    wandb_api_key: str | None = None
    use_cn_mirror: bool = False


class SSHConfigUpdateBody(SSHConfigBody):
    id: str


class SSHTestBody(BaseModel):
    config_id: str


class SSHRemoteEnvBody(BaseModel):
    config_id: str


class SSHInstallBody(BaseModel):
    config_id: str
    extra: str = ""
    use_cn_mirror: bool = False
    # Empty = install every supervised-policy extra (previous behavior).
    policies: list[str] = []


@app.get("/ssh-configs")
def ssh_configs_list():
    """List saved SSH connection configs (passwords redacted)."""
    return {"configs": [c.model_dump() for c in list_ssh_configs()]}


@app.post("/ssh-configs", status_code=201)
def ssh_configs_create(body: SSHConfigBody):
    """Save a new SSH connection config."""
    try:
        result = save_ssh_config(
            name=body.name,
            host=body.host,
            port=body.port,
            username=body.username,
            password=body.password,
            key_path=body.key_path,
            remote_python=body.remote_python,
            remote_workdir=body.remote_workdir,
            wandb_api_key=body.wandb_api_key,
            use_cn_mirror=body.use_cn_mirror,
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/ssh-configs/{config_id}")
def ssh_configs_update(config_id: str, body: SSHConfigBody):
    """Update an existing SSH config."""
    try:
        result = save_ssh_config(
            name=body.name,
            host=body.host,
            port=body.port,
            username=body.username,
            password=body.password,
            key_path=body.key_path,
            remote_python=body.remote_python,
            remote_workdir=body.remote_workdir,
            wandb_api_key=body.wandb_api_key,
            use_cn_mirror=body.use_cn_mirror,
            config_id=config_id,
        )
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/ssh-configs/{config_id}", status_code=204)
def ssh_configs_delete(config_id: str):
    """Delete a saved SSH config."""
    if not delete_ssh_config(config_id):
        raise HTTPException(status_code=404, detail=f"SSH config {config_id!r} not found")


@app.post("/ssh-configs/test")
def ssh_configs_test(body: SSHTestBody):
    """Test an SSH connection by connecting and running a trivial command."""
    return test_ssh_connection(body.config_id)


@app.post("/ssh-configs/check-env")
def ssh_check_env(body: SSHRemoteEnvBody):
    """Check remote environment: lerobot, CUDA, Python version, disk space."""
    return _check_remote_env(body.config_id)


class SSHRewardModelBody(BaseModel):
    config_id: str
    # The scoring model is ~18 GB; domestic links usually need the mirror.
    use_cn_mirror: bool = True


@app.post("/ssh-configs/reward-model-status")
def ssh_reward_model_status(body: SSHRewardModelBody):
    """Whether the rollout-scoring model is already on the pod."""
    return _reward_model_status(body.config_id)


@app.post("/ssh-configs/download-reward-model")
def ssh_download_reward_model(body: SSHRewardModelBody):
    """Start the scoring-model download on the pod (detached, polled via status)."""
    return _start_reward_model_download(body.config_id, use_cn_mirror=body.use_cn_mirror)


@app.post("/ssh-configs/install-lerobot")
def ssh_install_lerobot(body: SSHInstallBody):
    """Install lerobot on the remote machine via pip (SSE streaming)."""
    import json as _json

    task_id = f"ssh-install-{body.config_id}"
    # Guard against concurrent installs of the same target: two streams race
    # on the remote pip/dpkg locks and each restarts the other's downloads.
    existing = task_tracker.get(task_id)
    if existing is not None and existing.status == "running":
        raise HTTPException(
            status_code=409,
            detail="该目标已有安装任务在进行中，请等待其完成或失败后再重试",
        )
    task_tracker.create(
        task_id=task_id,
        task_type="ssh_install",
        label="Installing lerobot...",
        metadata={
            "config_id": body.config_id,
            "extra": body.extra,
            "policies": body.policies,
        },
    )

    def event_stream():
        for update in _install_lerobot_stream(
            body.config_id, body.extra, body.use_cn_mirror, body.policies or None
        ):
            # A step-level "done" means one step finished, not the whole task —
            # forwarding it would mark the task complete mid-install and the
            # banner would vanish with no feedback.
            task_status = "failed" if update.get("status") == "failed" else "running"
            task_tracker.update(
                task_id,
                status=task_status,
                label=update.get("label", ""),
                message=update.get("message", ""),
                metadata={"step": update.get("step"), "total_steps": update.get("total_steps")},
            )
            yield f"data: {_json.dumps(update)}\n\n"
        if task_tracker.get(task_id) and task_tracker.get(task_id).status == "running":
            task_tracker.update(task_id, status="done", message="Complete")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================================
# TASK TRACKING ENDPOINTS
# ============================================================================


@app.get("/tasks/active")
def get_active_tasks():
    """Return all currently running tasks (for frontend reconnection)."""
    return {"tasks": task_tracker.list_active()}


@app.get("/tasks/all")
def get_all_tasks():
    """Return all tasks including completed/failed."""
    return {"tasks": task_tracker.list_all()}


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================


@app.get("/system/cuda-status")
def get_cuda_status():
    """Report whether an NVIDIA GPU is present but PyTorch is CPU-only (issue #30)."""
    return handle_get_cuda_status()


@app.get("/system/training-extra")
def get_training_extra():
    """Return whether the LeRobot training extra (accelerate) is importable."""
    return handle_get_training_extra()


@app.post("/system/training-extra/install")
def install_training_extra():
    """Spawn `pip install accelerate` as a background subprocess. No-op if already running."""
    return handle_install_training_extra()


@app.get("/system/training-extra/install-status")
def install_training_extra_status():
    """Return current install state plus any pending log lines (drained on read)."""
    return handle_install_training_extra_status()


@app.get("/system/wandb-extra")
def get_wandb_extra():
    """Return whether the `wandb` package is importable in this lelab process."""
    return handle_get_wandb_extra()


@app.post("/system/wandb-extra/install")
def install_wandb_extra():
    """Spawn `pip install wandb` as a background subprocess. No-op if already running."""
    return handle_install_wandb_extra()


@app.get("/system/wandb-extra/install-status")
def install_wandb_extra_status():
    """Return current wandb install state plus any pending log lines (drained on read)."""
    return handle_install_wandb_extra_status()


@app.get("/system/policy-extra/{policy_type}")
def get_policy_extra(policy_type: str):
    """Whether the optional LeRobot extra a policy needs (e.g. transformers for
    smolvla/pi0, diffusers for diffusion) is importable. Core policies report available."""
    return handle_get_policy_extra(policy_type)


@app.post("/system/policy-extra/{policy_type}/install")
def install_policy_extra(policy_type: str):
    """Spawn `pip install lerobot[<extra>]` for the policy's extra in the background."""
    return handle_install_policy_extra(policy_type)


@app.get("/system/policy-extra/{policy_type}/install-status")
def install_policy_extra_status(policy_type: str):
    """Return the policy extra's install state plus any pending log lines (drained on read)."""
    return handle_install_policy_extra_status(policy_type)


@app.get("/system/update-check")
def update_check():
    """Report whether a newer LeLab commit exists on GitHub (cached, silent on failure)."""
    return handle_update_check()


@app.post("/system/update")
def run_update():
    """Run the pip upgrade in-process; the user must restart lelab afterwards."""
    return handle_run_update()


# Replay is rendered by the embedded lerobot/visualize_dataset Space; no backend routes needed.


# ============================================================================
# Calibration endpoints
@app.post("/start-calibration")
def start_calibration(request: CalibrationRequest):
    """Start calibration process"""
    return calibration_manager.start_calibration(request)


@app.post("/stop-calibration")
def stop_calibration():
    """Stop calibration process"""
    return calibration_manager.stop_calibration_process()


@app.get("/calibration-status")
def calibration_status():
    """Get current calibration status"""
    from dataclasses import asdict

    status = calibration_manager.get_status()
    return asdict(status)


@app.post("/complete-calibration-step")
def complete_calibration_step():
    """Complete the current calibration step"""
    return calibration_manager.complete_step()


@app.get("/calibration-configs/{device_type}")
def get_calibration_configs(device_type: str):
    """Get all calibration config files for a specific device type"""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        # Get all JSON files in the config directory
        configs = []
        if os.path.exists(config_path):
            for file in os.listdir(config_path):
                if file.endswith(".json"):
                    config_name = os.path.splitext(file)[0]
                    file_path = os.path.join(config_path, file)
                    file_size = os.path.getsize(file_path)
                    modified_time = os.path.getmtime(file_path)

                    configs.append(
                        {
                            "name": config_name,
                            "filename": file,
                            "size": file_size,
                            "modified": modified_time,
                        }
                    )

        return {"success": True, "configs": configs, "device_type": device_type}

    except Exception as e:
        logger.error(f"Error getting calibration configs: {e}")
        return {"success": False, "message": str(e)}


@app.delete("/calibration-configs/{device_type}/{config_name}")
def delete_calibration_config(device_type: str, config_name: str):
    """Delete a calibration config file"""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        # config_name is interpolated into a filename, so reject path-traversal
        # characters (/, \, ..) before touching the filesystem. Defense-in-depth:
        # FastAPI path params already block a literal "/", but not "\" or "..".
        # Reuses the same guard already applied to robot-record deletes.
        if not is_valid_robot_name(config_name):
            return {"success": False, "message": "Invalid configuration name"}

        # Construct the file path
        filename = f"{config_name}.json"
        file_path = os.path.join(config_path, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            return {"success": False, "message": "Configuration file not found"}

        # Delete the file
        os.remove(file_path)
        logger.info(f"Deleted calibration config: {file_path}")

        return {
            "success": True,
            "message": f"Configuration '{config_name}' deleted successfully",
        }

    except Exception as e:
        logger.error(f"Error deleting calibration config: {e}")
        return {"success": False, "message": str(e)}


# ============================================================================
# PORT DETECTION ENDPOINTS
# ============================================================================


@app.get("/available-ports")
def get_available_ports():
    """Get all available serial ports"""
    try:
        ports = find_available_ports()
        return {"status": "success", "ports": ports}
    except Exception as e:
        logger.error(f"Error getting available ports: {e}")
        return {"status": "error", "message": str(e)}


# Runs in a fresh Python — see _avfoundation_cameras_in_cv2_order for why.
# Mirrors OpenCV's macOS enumeration: video + muxed devices sorted by
# uniqueID (cap_avfoundation_mac.mm), so the returned index matches what
# cv2.VideoCapture will open.
_AVF_ENUM_SCRIPT = """
import json, objc
from Foundation import NSBundle
bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
bundle.load()
types = []
for name in (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",   # macOS < 14
    "AVCaptureDeviceTypeExternal",          # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",    # macOS >= 13
):
    loaded = {}
    try:
        objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
    except objc.error:
        continue
    if loaded.get(name) is not None:
        types.append(loaded[name])
cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
devs = []
for mt in ("vide", "muxx"):
    devs.extend(cls.discoverySessionWithDeviceTypes_mediaType_position_(types, mt, 0).devices() or [])
devs.sort(key=lambda d: d.uniqueID())
print(json.dumps([
    {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
    for i, d in enumerate(devs)
]))
"""


def _avfoundation_cameras_in_cv2_order() -> list[dict[str, Any]]:
    """Enumerate macOS cameras in a fresh Python subprocess.

    AVFoundation's in-process device cache doesn't refresh on USB
    hotplug. Both the deprecated ``+devicesWithMediaType:`` and a
    long-lived ``AVCaptureDeviceDiscoverySession`` go stale, because
    device-connection notifications are delivered via
    ``NSNotificationCenter`` on a thread that needs an active
    ``NSRunLoop`` — uvicorn workers don't run one. A fresh subprocess
    re-initializes AVFoundation, which reads IOKit's live device state
    at startup.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _AVF_ENUM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("AVFoundation enumeration subprocess failed: %s", e)
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("AVFoundation enumeration returned invalid JSON: %s", e)
        return []


def _generic_cv2_cameras(backend) -> list[dict[str, Any]]:
    """Last-resort enumeration: probe cv2 indices with placeholder names."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        cap.release()
        if opened:
            cameras.append({"index": i, "name": f"Camera {i}", "available": True})
    return cameras


def _windows_cameras() -> list[dict[str, Any]]:
    """Enumerate Windows cameras with their real DirectShow names.

    pygrabber lists DirectShow video devices in the same order cv2's DSHOW
    backend indexes them (which recording is pinned to), so the returned index
    matches what ``cv2.VideoCapture(i, CAP_DSHOW)`` opens. The real names let the
    frontend match each index to the browser's ``MediaDeviceInfo.label`` for the
    live preview. Falls back to generic names if pygrabber is unavailable.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        names = FilterGraph().get_input_devices()
    except Exception as e:  # ImportError, or a COM/DirectShow failure
        logger.warning("pygrabber unavailable; using generic camera names: %s", e)
        import cv2

        return _generic_cv2_cameras(cv2.CAP_DSHOW)
    return [{"index": i, "name": name, "available": True} for i, name in enumerate(names)]


def _v4l2_camera_name(index: int) -> str | None:
    """Real camera name for /dev/video{index} from sysfs (Linux, no deps)."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _linux_cameras() -> list[dict[str, Any]]:
    """Enumerate Linux cameras, naming each from sysfs (no extra deps)."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        opened = cap.isOpened()
        cap.release()
        if not opened:
            continue
        cameras.append({"index": i, "name": _v4l2_camera_name(i) or f"Camera {i}", "available": True})
    return cameras


@app.get("/available-cameras")
def get_available_cameras():
    """List cameras with the same index ordering cv2 will use to record.

    Each platform enumerates in the order its cv2 backend indexes devices, and
    pairs each index with the device's real name so the frontend can match it to
    the browser's ``MediaDeviceInfo.label`` for the live preview:
      - macOS: AVFoundation ``localizedName`` (via a PyObjC subprocess);
      - Windows: DirectShow FriendlyName (via pygrabber; recording pinned DSHOW);
      - Linux: the v4l2 device name from sysfs.
    Without real names the frontend can't match a camera and shows "No browser
    match" with an empty device_id (issues #12, #16).
    """
    try:
        import platform

        system = platform.system()

        if system == "Darwin":
            cameras = _avfoundation_cameras_in_cv2_order()
            for cam in cameras:
                cam["available"] = True
            return {"status": "success", "cameras": cameras}
        if system == "Windows":
            return {"status": "success", "cameras": _windows_cameras()}
        if system == "Linux":
            return {"status": "success", "cameras": _linux_cameras()}

        import cv2

        return {"status": "success", "cameras": _generic_cv2_cameras(cv2.CAP_ANY)}
    except ImportError:
        logger.warning("OpenCV not available for camera detection")
        return {"status": "success", "cameras": []}
    except Exception as e:
        logger.error(f"Error detecting cameras: {e}")
        return {"status": "error", "message": str(e), "cameras": []}


RobotSideLiteral = Literal["leader", "follower"]


class PortDetectionBody(BaseModel):
    robot_type: RobotSideLiteral = "follower"


class PortDisconnectBody(BaseModel):
    ports_before: list[str]


class SaveRobotPortBody(BaseModel):
    robot_type: RobotSideLiteral
    port: str


class SaveRobotConfigBody(BaseModel):
    robot_type: RobotSideLiteral
    config_name: str


@app.post("/start-port-detection")
def start_port_detection(body: PortDetectionBody):
    """Snapshot available ports so the follow-up /detect-port-after-disconnect
    call can diff them."""
    result = find_robot_port(body.robot_type)
    return {"status": "success", "data": result}


@app.post("/detect-port-after-disconnect")
def detect_port_after_disconnect_endpoint(body: PortDisconnectBody):
    """Block up to 15s waiting for one port from `ports_before` to disappear."""
    try:
        detected_port = detect_port_after_disconnect(body.ports_before)
    except OSError as exc:
        raise HTTPException(status_code=408, detail=str(exc)) from exc
    return {"status": "success", "port": detected_port}


@app.post("/save-robot-port")
def save_robot_port_endpoint(body: SaveRobotPortBody):
    """Save a robot port for future use"""
    save_robot_port(body.robot_type, body.port)
    return {"status": "success", "message": f"Port {body.port} saved for {body.robot_type}"}


@app.get("/robot-port/{robot_type}")
def get_robot_port(robot_type: RobotSideLiteral):
    """Get the saved port for a robot type"""
    saved_port = get_saved_robot_port(robot_type)
    default_port = get_default_robot_port(robot_type)
    return {"status": "success", "saved_port": saved_port, "default_port": default_port}


@app.post("/save-robot-config")
def save_robot_config_endpoint(body: SaveRobotConfigBody):
    """Save a robot configuration for future use"""
    if not config.save_robot_config(body.robot_type, body.config_name):
        raise HTTPException(status_code=500, detail="Failed to save configuration")
    return {"status": "success", "message": f"Configuration saved for {body.robot_type}"}


@app.get("/robot-config/{robot_type}")
def get_robot_config(robot_type: RobotSideLiteral, available_configs: str = ""):
    """Get the saved configuration for a robot type"""
    available_configs_list = [c.strip() for c in available_configs.split(",") if c.strip()]
    saved_config = config.get_saved_robot_config(robot_type)
    default_config = config.get_default_robot_config(robot_type, available_configs_list)
    return {"status": "success", "saved_config": saved_config, "default_config": default_config}


# ============================================================================
# Robot config records (named robots)


def _record_with_clean(record: dict) -> dict:
    """Attach `is_clean` to a record for API responses."""
    return {**record, "is_clean": is_robot_record_clean(record)}


@app.get("/robots")
def get_robots():
    """List all saved robot records."""
    try:
        records = [_record_with_clean(r) for r in list_robot_records()]
        return {"status": "success", "robots": records}
    except Exception as e:
        logger.error(f"Error listing robots: {e}")
        return {"status": "error", "message": str(e), "robots": []}


@app.get("/robots/{name}")
def get_robot(name: str):
    """Get a single robot record by name."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    record = get_robot_record(name)
    if record is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})
    return {"status": "success", "robot": _record_with_clean(record)}


@app.post("/robots/{name}")
def upsert_robot(name: str, data: dict, create: bool = False):
    """
    Upsert a robot record.

    - `?create=true` is the "Add Robot" path: returns 409 if a record with that
      name already exists; otherwise creates with empty fields then merges body.
    - Without `?create=true` is the "patch" path (e.g., calibration write-back):
      merges body into existing record. If no record exists, no-ops and returns
      success — see deletion-during-calibration edge case in the spec.
    """
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    try:
        if create:
            if get_robot_record(name) is not None:
                return JSONResponse(
                    status_code=409,
                    content={"status": "error", "message": "A robot with this name already exists"},
                )
            save_robot_record(name, data or {}, allow_create=True)
        else:
            save_robot_record(name, data or {}, allow_create=False)
        record = get_robot_record(name)
        if record is None:
            return {"status": "success", "robot": None}
        return {"status": "success", "robot": _record_with_clean(record)}
    except Exception as e:
        logger.error(f"Error upserting robot {name}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.delete("/robots/{name}")
def delete_robot(name: str):
    """Delete a robot record."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    if delete_robot_record(name):
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})


@app.on_event("startup")
def startup_event():
    """One-time startup diagnostics surfaced in the server terminal."""
    warn_if_cuda_mismatch()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when FastAPI shuts down"""
    logger.info("🔄 FastAPI shutting down, cleaning up...")

    # Stop any active recording - handled by recording module cleanup

    if manager:
        manager.stop_broadcast_thread()
    logger.info("✅ Cleanup completed")


def _accepts_html(accept: str) -> bool:
    """Whether an Accept header explicitly wants text/html (quality > 0).

    Browser navigations list `text/html` with a positive quality value, so
    they get the SPA shell. A `text/html;q=0` entry is an explicit refusal and
    must not count — a plain substring check would wrongly treat it as a yes.
    `*/*` (curl, XHR, API clients) is deliberately not treated as wanting HTML.
    """
    for part in accept.split(","):
        media_type, _, params = part.strip().partition(";")
        if media_type.strip().lower() != "text/html":
            continue
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        return quality > 0
    return False


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown client-side routes.

    The frontend is a single-page app: routes like /recording and /calibration
    exist only in the browser's router, not as files on disk. A hard reload or
    deep link to one of those URLs asks the server for a file that isn't there;
    plain StaticFiles answers 404 ({"detail":"Not Found"}), so the page breaks.

    Here we fall back to index.html on 404 so the SPA boots and its router
    renders the route. Only requests that accept HTML (i.e. browser navigations)
    get the fallback — API typos, XHR, and curl still receive a JSON 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and _accepts_html(Headers(scope=scope).get("accept", "")):
                response = await super().get_response("index.html", scope)
            else:
                raise
        # Disable caching for HTML files so users always get the latest version
        if path.endswith(".html") or path == "" or path == "/" or path == "index.html":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        # Also disable caching for JS/CSS to ensure frontend updates are picked up immediately
        elif path.endswith(".js") or path.endswith(".css"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# Middleware to disable caching for HTML responses
@app.middleware("http")
async def disable_html_cache(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/html") or content_type.startswith("application/javascript") or content_type.startswith("text/css"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Serve the built frontend at /. Must be mounted last so API routes win.
if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    logger.warning(
        f"frontend/dist not found at {FRONTEND_DIST}; run `npm run build` in frontend/ or use `lelab --dev`."
    )
