"""job_state.py - the resumable-job manifest, so an interrupted video run can pick up where it stopped.

Before any frame is extracted, run_job writes a small JSON manifest into the target's temp
directory. It records which target/source/fps-strategy/frame-format the run belongs to, which
stage the run is in, and which frames each frame processor has already finished. On the next
start with the same source/target, a matching manifest lets the run skip the extraction and the
frames that are already done; a missing, corrupt, outdated or non-matching manifest simply means
"start over" - it never raises.

The functions here are plain file/JSON helpers with no UI or processor dependencies, so they can
be tested on their own.
"""
import json
import os
from typing import Dict, List, Optional

from changeling.utilities import get_temp_directory_path

MANIFEST_VERSION = 1
MANIFEST_FILE = 'changeling-job.json'

# stages, in order; the manifest is deleted once the whole job succeeds
STAGE_EXTRACTING = 'extracting'
STAGE_PROCESSING = 'processing'
STAGE_CREATING = 'creating'
STAGES = (STAGE_EXTRACTING, STAGE_PROCESSING, STAGE_CREATING)


def get_manifest_path(target_path: str) -> str:
    return os.path.join(get_temp_directory_path(target_path), MANIFEST_FILE)


def new_manifest(source_path: Optional[str], target_path: str, keep_fps: bool, temp_frame_format: str) -> Dict:
    """A fresh manifest for a run, in the 'extracting' stage with nothing done yet."""
    return {
        'version': MANIFEST_VERSION,
        'source_path': os.path.abspath(source_path) if source_path else None,
        'target_path': os.path.abspath(target_path),
        'keep_fps': bool(keep_fps),
        'temp_frame_format': temp_frame_format,
        'stage': STAGE_EXTRACTING,
        # frame processor name -> sorted list of finished frame file names
        'done_frames': {}
    }


def load_manifest(target_path: str) -> Optional[Dict]:
    """The manifest for target_path, or None when it is missing, unreadable, corrupt,
    from another manifest version, or structurally wrong. Never raises."""
    try:
        with open(get_manifest_path(target_path), 'r', encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get('version') != MANIFEST_VERSION:
        return None
    if manifest.get('stage') not in STAGES:
        return None
    if not isinstance(manifest.get('done_frames'), dict):
        return None
    return manifest


def save_manifest(target_path: str, manifest: Dict) -> None:
    """Write the manifest atomically (temp file + replace), so a kill mid-write cannot
    leave a half-written manifest behind."""
    manifest_path = get_manifest_path(target_path)
    temp_path = manifest_path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
    os.replace(temp_path, manifest_path)


def discard_manifest(target_path: str) -> None:
    try:
        os.remove(get_manifest_path(target_path))
    except OSError:
        pass


def manifest_matches(manifest: Optional[Dict], source_path: Optional[str], target_path: str,
                     keep_fps: bool, temp_frame_format: str) -> bool:
    """True only when the manifest describes exactly this run: same target, same source,
    same fps strategy (keep_fps) and same temp frame format. Any difference - or a
    manifest that cannot answer these questions - means no match, i.e. full rerun."""
    if not isinstance(manifest, dict):
        return False
    if manifest.get('version') != MANIFEST_VERSION:
        return False
    try:
        expected_target = os.path.abspath(target_path)
        expected_source = os.path.abspath(source_path) if source_path else None
    except (TypeError, ValueError):
        return False
    return (
        manifest.get('target_path') == expected_target
        and manifest.get('source_path') == expected_source
        and bool(manifest.get('keep_fps')) == bool(keep_fps)
        and manifest.get('temp_frame_format') == temp_frame_format
    )


def remaining_frames(manifest: Dict, processor_name: str, frame_paths: List[str]) -> List[str]:
    """The frames this processor still has to do (all of them, minus the ones the manifest
    already lists as done for this processor)."""
    done = set(manifest.get('done_frames', {}).get(processor_name, []))
    return [frame_path for frame_path in frame_paths if os.path.basename(frame_path) not in done]


def mark_frames_done(manifest: Dict, processor_name: str, frame_paths: List[str]) -> None:
    """Record frames as done for a processor. Pure in-memory update - pair with
    save_manifest to persist (that is what the per-batch callback in core.run_job does)."""
    done_frames = manifest.setdefault('done_frames', {})
    done = set(done_frames.get(processor_name, []))
    done.update(os.path.basename(frame_path) for frame_path in frame_paths)
    done_frames[processor_name] = sorted(done)


def set_stage(manifest: Dict, stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    manifest['stage'] = stage


def reset_temp(target_path: str) -> None:
    """Empty the temp directory (stale frames, half-written temp video, old manifest) so a
    full rerun starts from a clean slate. The directory itself stays."""
    temp_directory_path = get_temp_directory_path(target_path)
    if not os.path.isdir(temp_directory_path):
        return
    for name in os.listdir(temp_directory_path):
        path = os.path.join(temp_directory_path, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
