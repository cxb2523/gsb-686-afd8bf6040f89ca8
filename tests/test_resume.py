"""Resumable video jobs: manifest matching, crash resume, corrupt-manifest fallback and
"no fake success" on synthesis/audio failure. Everything runs on fake frame files in a
temporary directory - no models, no GPU, no ffmpeg. Frame processors and the ffmpeg-facing
utilities are stubbed via monkeypatch."""
import os

import pytest

import changeling.globals
import changeling.core as core
from changeling import job_state
from changeling.utilities import create_temp, get_temp_directory_path

FRAME_COUNT = 4


class FakeProcessor:
    """Stands in for a frame processor module (face_swapper / face_enhancer)."""

    NAME = 'CHANGELING.FAKE'

    def __init__(self, crash_after=None):
        self.pre_start_calls = 0
        self.post_process_calls = 0
        self.processed = []           # frame basenames this run was asked to process
        self.crash_after = crash_after  # raise after this many frames, simulating a kill/exception

    def pre_start(self):
        self.pre_start_calls += 1
        return True

    def process_video(self, source_path, temp_frame_paths, on_frames_done=None):
        for frame_path in temp_frame_paths:
            if self.crash_after is not None and len(self.processed) >= self.crash_after:
                raise RuntimeError('simulated crash mid-processing')
            self.processed.append(os.path.basename(frame_path))
            if on_frames_done:
                on_frames_done([frame_path])

    def post_process(self):
        self.post_process_calls += 1


@pytest.fixture
def job(tmp_path, monkeypatch):
    """A ready-to-run video job in a temp directory, with ffmpeg-facing functions faked."""
    source_path = tmp_path / 'source.png'
    target_path = tmp_path / 'target.mp4'
    output_path = tmp_path / 'output.mp4'
    source_path.write_bytes(b'fake source image')
    target_path.write_bytes(b'fake target video')

    monkeypatch.setattr(changeling.globals, 'source_path', str(source_path))
    monkeypatch.setattr(changeling.globals, 'target_path', str(target_path))
    monkeypatch.setattr(changeling.globals, 'output_path', str(output_path))
    monkeypatch.setattr(changeling.globals, 'headless', True)
    monkeypatch.setattr(changeling.globals, 'frame_processors', ['fake'])
    monkeypatch.setattr(changeling.globals, 'keep_fps', False)
    monkeypatch.setattr(changeling.globals, 'keep_frames', False)
    monkeypatch.setattr(changeling.globals, 'skip_audio', False)
    monkeypatch.setattr(changeling.globals, 'temp_frame_format', 'png')
    monkeypatch.setattr(changeling.globals, 'temp_frame_quality', 0)
    monkeypatch.setattr(changeling.globals, 'output_video_encoder', 'libx264')
    monkeypatch.setattr(changeling.globals, 'output_video_quality', 35)
    monkeypatch.setattr(changeling.globals, 'execution_threads', 1)

    state = {'extract_calls': 0, 'create_video_ok': True, 'restore_audio_ok': True}

    def fake_extract_frames(target, fps=30):
        state['extract_calls'] += 1
        temp_directory_path = get_temp_directory_path(target)
        for index in range(1, FRAME_COUNT + 1):
            with open(os.path.join(temp_directory_path, '%04d.png' % index), 'wb') as frame_file:
                frame_file.write(b'frame')

    def fake_create_video(target, fps=30):
        if not state['create_video_ok']:
            return False
        with open(os.path.join(get_temp_directory_path(target), 'temp.mp4'), 'wb') as temp_video:
            temp_video.write(b'video')
        return True

    def fake_restore_audio(target, output):
        if not state['restore_audio_ok']:
            return False
        with open(output, 'wb') as output_file:
            output_file.write(b'video with audio')
        return True

    monkeypatch.setattr(core, 'predict_video', lambda target: False)
    monkeypatch.setattr(core, 'detect_fps', lambda target: 30.0)
    monkeypatch.setattr(core, 'extract_frames', fake_extract_frames)
    monkeypatch.setattr(core, 'create_video', fake_create_video)
    monkeypatch.setattr(core, 'restore_audio', fake_restore_audio)

    class Env:
        pass

    env = Env()
    env.source_path = str(source_path)
    env.target_path = str(target_path)
    env.output_path = str(output_path)
    env.state = state
    env.processor = FakeProcessor()
    env.set_processor = lambda processor: monkeypatch.setattr(
        core, 'get_frame_processors_modules', lambda names: [processor])
    env.set_processor(env.processor)
    return env


# --- manifest unit tests -----------------------------------------------------


def test_manifest_roundtrip_and_match(tmp_path):
    target = str(tmp_path / 'target.mp4')
    source = str(tmp_path / 'source.png')
    create_temp(target)
    manifest = job_state.new_manifest(source, target, True, 'png')
    job_state.save_manifest(target, manifest)

    loaded = job_state.load_manifest(target)
    assert loaded == manifest
    assert job_state.manifest_matches(loaded, source, target, True, 'png')


def test_manifest_mismatch_on_target_fps_strategy_and_format(tmp_path):
    target = str(tmp_path / 'target.mp4')
    source = str(tmp_path / 'source.png')
    manifest = job_state.new_manifest(source, target, False, 'png')

    assert job_state.manifest_matches(manifest, source, target, False, 'png')
    # another target, another fps strategy (keep_fps) or another frame format: no match
    assert not job_state.manifest_matches(manifest, source, str(tmp_path / 'other.mp4'), False, 'png')
    assert not job_state.manifest_matches(manifest, source, target, True, 'png')
    assert not job_state.manifest_matches(manifest, source, target, False, 'jpg')
    # another source: no match either
    assert not job_state.manifest_matches(manifest, str(tmp_path / 'other.png'), target, False, 'png')
    # garbage in: no match, no exception
    assert not job_state.manifest_matches(None, source, target, False, 'png')
    assert not job_state.manifest_matches({'version': 999}, source, target, False, 'png')
    assert not job_state.manifest_matches('not a dict', source, target, False, 'png')


def test_corrupt_or_versioned_out_manifest_loads_as_none(tmp_path):
    target = str(tmp_path / 'target.mp4')
    create_temp(target)
    manifest_path = job_state.get_manifest_path(target)

    assert job_state.load_manifest(target) is None  # missing

    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        manifest_file.write('{ not json !')
    assert job_state.load_manifest(target) is None  # corrupt

    manifest = job_state.new_manifest(None, target, False, 'png')
    manifest['version'] = 999
    job_state.save_manifest(target, manifest)
    assert job_state.load_manifest(target) is None  # unknown version

    manifest = job_state.new_manifest(None, target, False, 'png')
    manifest['stage'] = 'levitating'
    job_state.save_manifest(target, manifest)
    assert job_state.load_manifest(target) is None  # unknown stage


def test_remaining_and_mark_frames_done(tmp_path):
    target = str(tmp_path / 'target.mp4')
    manifest = job_state.new_manifest(None, target, False, 'png')
    frames = [os.path.join('temp', 'target', '%04d.png' % index) for index in range(1, 5)]

    assert job_state.remaining_frames(manifest, 'fake', frames) == frames
    job_state.mark_frames_done(manifest, 'fake', frames[:2])
    assert job_state.remaining_frames(manifest, 'fake', frames) == frames[2:]
    # done frames are tracked per processor
    assert job_state.remaining_frames(manifest, 'other', frames) == frames


# --- run_job resume behaviour ------------------------------------------------


def test_clean_run_writes_and_removes_manifest(job):
    ok, message = core.run_job()
    assert ok, message
    assert os.path.isfile(job.output_path)
    assert job.processor.processed == ['0001.png', '0002.png', '0003.png', '0004.png']
    # a successful job cleans the temp directory including the manifest
    assert job_state.load_manifest(job.target_path) is None
    assert not os.path.isdir(get_temp_directory_path(job.target_path))


def test_resume_after_crash_processes_only_remaining_frames(job):
    job.processor.crash_after = 2
    with pytest.raises(RuntimeError):
        core.run_job()

    # the crash left a resumable manifest behind: 2 of 4 frames done, stage 'processing'
    manifest = job_state.load_manifest(job.target_path)
    assert manifest is not None
    assert manifest['stage'] == job_state.STAGE_PROCESSING
    assert manifest['done_frames']['fake'] == ['0001.png', '0002.png']

    # second run, same source/target: no re-extraction, only the remaining frames
    second_processor = FakeProcessor()
    job.set_processor(second_processor)
    ok, message = core.run_job()
    assert ok, message
    assert job.state['extract_calls'] == 1  # extraction happened only in the first run
    assert second_processor.processed == ['0003.png', '0004.png']
    # pre_start still ran again on the resumed run
    assert second_processor.pre_start_calls == 1
    assert os.path.isfile(job.output_path)
    # success cleaned up
    assert job_state.load_manifest(job.target_path) is None


def test_mismatched_manifest_triggers_full_rerun(job, monkeypatch):
    job.processor.crash_after = 2
    with pytest.raises(RuntimeError):
        core.run_job()
    assert job_state.load_manifest(job.target_path) is not None

    # same target, but the fps strategy changed: the old progress is worthless
    monkeypatch.setattr(changeling.globals, 'keep_fps', True)
    second_processor = FakeProcessor()
    job.set_processor(second_processor)
    ok, message = core.run_job()
    assert ok, message
    assert job.state['extract_calls'] == 2  # re-extracted
    assert second_processor.processed == ['0001.png', '0002.png', '0003.png', '0004.png']


def test_corrupt_manifest_degrades_to_full_rerun(job):
    job.processor.crash_after = 2
    with pytest.raises(RuntimeError):
        core.run_job()

    # the manifest is damaged while the process is down
    with open(job_state.get_manifest_path(job.target_path), 'w', encoding='utf-8') as manifest_file:
        manifest_file.write('{ broken')

    second_processor = FakeProcessor()
    job.set_processor(second_processor)
    ok, message = core.run_job()
    assert ok, message
    assert job.state['extract_calls'] == 2  # no trust in the corrupt manifest: full rerun
    assert second_processor.processed == ['0001.png', '0002.png', '0003.png', '0004.png']


def test_create_video_failure_is_not_a_success_and_resumes(job):
    job.state['create_video_ok'] = False
    ok, message = core.run_job()
    assert not ok
    # no half-finished output, and the failure is reported
    assert not os.path.exists(job.output_path)
    # the run's progress survives for the next try
    manifest = job_state.load_manifest(job.target_path)
    assert manifest is not None
    assert manifest['stage'] == job_state.STAGE_CREATING
    assert manifest['done_frames']['fake'] == ['0001.png', '0002.png', '0003.png', '0004.png']

    # next try: frames are all done, only the video creation runs again
    job.state['create_video_ok'] = True
    second_processor = FakeProcessor()
    job.set_processor(second_processor)
    ok, message = core.run_job()
    assert ok, message
    assert second_processor.processed == []
    assert os.path.isfile(job.output_path)


def test_restore_audio_failure_is_not_a_success(job):
    job.state['restore_audio_ok'] = False
    ok, message = core.run_job()
    assert not ok
    assert not os.path.exists(job.output_path)
    # progress is kept so the next run can retry the audio step
    assert job_state.load_manifest(job.target_path) is not None


def test_keep_frames_keeps_frames_but_not_the_manifest(job, monkeypatch):
    monkeypatch.setattr(changeling.globals, 'keep_frames', True)
    ok, message = core.run_job()
    assert ok, message
    # --keep-frames: the frames stay, but the finished job's manifest is gone
    temp_directory_path = get_temp_directory_path(job.target_path)
    assert len([name for name in os.listdir(temp_directory_path) if name.endswith('.png')]) == FRAME_COUNT
    assert job_state.load_manifest(job.target_path) is None
