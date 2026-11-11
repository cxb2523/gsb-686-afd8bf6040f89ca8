#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# opennsfw2 (the NSFW check) is written for Keras 2; TensorFlow 2.16+ ships Keras 3 unless told otherwise
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import threading
import onnxruntime
import tensorflow
import changeling.globals
import changeling.metadata
from changeling import progress
from changeling import job_state
import changeling.ui as ui
from changeling.predictor import predict_image, predict_video
from changeling.processors.frame.core import get_frame_processors_modules
from changeling.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help="frame processors (choices: face_swapper, face_enhancer, ...); the default swaps and then sharpens the face (GFPGAN), like the window's Sharper face switch", dest='frame_processor', default=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--face-enhancer-model', help='the Sharper face model: GFPGAN (smooth, clean), CodeFormer (the most faithful) or GPEN (the most detail)', dest='face_enhancer_model', default='GFPGAN', choices=['GFPGAN', 'CodeFormer', 'GPEN'])
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{changeling.metadata.name} {changeling.metadata.version}')

    args = program.parse_args()

    changeling.globals.source_path = args.source_path
    changeling.globals.target_path = args.target_path
    changeling.globals.output_path = normalize_output_path(changeling.globals.source_path, changeling.globals.target_path, args.output_path)
    changeling.globals.headless = changeling.globals.source_path is not None and changeling.globals.target_path is not None and changeling.globals.output_path is not None
    changeling.globals.frame_processors = args.frame_processor
    changeling.globals.keep_fps = args.keep_fps
    changeling.globals.keep_frames = args.keep_frames
    changeling.globals.skip_audio = args.skip_audio
    changeling.globals.many_faces = args.many_faces
    changeling.globals.face_enhancer_model = args.face_enhancer_model
    changeling.globals.reference_face_position = args.reference_face_position
    changeling.globals.reference_frame_number = args.reference_frame_number
    changeling.globals.similar_face_distance = args.similar_face_distance
    changeling.globals.temp_frame_format = args.temp_frame_format
    changeling.globals.temp_frame_quality = args.temp_frame_quality
    changeling.globals.output_video_encoder = args.output_video_encoder
    changeling.globals.output_video_quality = args.output_video_quality
    changeling.globals.max_memory = args.max_memory
    changeling.globals.execution_providers = decode_execution_providers(args.execution_provider)
    changeling.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if changeling.globals.max_memory:
        memory = changeling.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = changeling.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


STAGES = {'CHANGELING.FACE-SWAPPER': 'Swapping faces', 'CHANGELING.FACE-ENHANCER': 'Cleaning up faces'}


def update_status(message: str, scope: str = 'CHANGELING.CORE') -> None:
    print(f'[{scope}] {message}')
    if not changeling.globals.headless:
        progress.note(message)


def start() -> None:
    """The whole job, start to finish, reporting each stage to the window's completion bar."""
    try:
        ok, message = run_job()
    except SystemExit:
        raise
    except Exception as error:  # the window says what went wrong instead of the job vanishing
        ok, message = False, f'Failed: {error}'
        print(f'[CHANGELING.CORE] {message}')
    if ok and changeling.globals.output_path and os.path.exists(changeling.globals.output_path):
        try:
            from gallery import remember            # the Changeling gallery (gallery.py, next to run.py)
            remember(changeling.globals.output_path, 'Imago')
        except ImportError:
            pass
    progress.finish(message, ok)


def run_job() -> tuple:
    """(ok, message) for one job: an image or a video."""
    for frame_processor in get_frame_processors_modules(changeling.globals.frame_processors):
        if not frame_processor.pre_start():
            return False, progress.snapshot()['message'] or 'Not started.'
    target = changeling.globals.target_path
    # process image to image
    if has_image_extension(target):
        progress.begin('Checking the picture')
        if predict_image(target):
            blocked()
        shutil.copy2(target, changeling.globals.output_path)
        for frame_processor in get_frame_processors_modules(changeling.globals.frame_processors):
            progress.begin(STAGES.get(frame_processor.NAME, 'Working'))
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(changeling.globals.source_path, changeling.globals.output_path, changeling.globals.output_path)
            frame_processor.post_process()
        if is_image(target):
            update_status('Processing to image succeed!')
            return True, f'Done: {os.path.basename(changeling.globals.output_path)}'
        update_status('Processing to image failed!')
        return False, 'Processing the image failed.'
    # process image to videos
    progress.begin('Checking the video')
    if predict_video(target):
        blocked()
    update_status('Creating temporary resources...')
    create_temp(target)
    source = changeling.globals.source_path
    # resume bookkeeping: a matching manifest from an interrupted run lets us skip the
    # extraction and the frames already done; anything else means a full rerun
    manifest = job_state.load_manifest(target)
    if manifest is not None and job_state.manifest_matches(manifest, source, target, changeling.globals.keep_fps, changeling.globals.temp_frame_format):
        if get_temp_frame_paths(target):
            update_status('Resuming an interrupted run...')
        else:
            update_status('The previous run\'s frames are gone; starting over...')
            manifest = None
    else:
        if manifest is not None:
            update_status('The previous run\'s progress does not match this job; starting over...')
        manifest = None
    if manifest is None:
        job_state.reset_temp(target)
        manifest = job_state.new_manifest(source, target, changeling.globals.keep_fps, changeling.globals.temp_frame_format)
        job_state.save_manifest(target, manifest)
    if manifest['stage'] == job_state.STAGE_EXTRACTING:
        progress.begin('Extracting frames')
        if changeling.globals.keep_fps:
            fps = detect_fps(target)
            update_status(f'Extracting frames with {fps} FPS...')
            extract_frames(target, fps)
        else:
            update_status('Extracting frames with 30 FPS...')
            extract_frames(target)
        job_state.set_stage(manifest, job_state.STAGE_PROCESSING)
        job_state.save_manifest(target, manifest)
    temp_frame_paths = get_temp_frame_paths(target)
    if not temp_frame_paths:
        update_status('Frames not found...')
        return False, 'No frames came out of the video.'
    if manifest['stage'] == job_state.STAGE_PROCESSING:
        manifest_lock = threading.Lock()
        processor_names = list(changeling.globals.frame_processors)
        for processor_name, frame_processor in zip(processor_names, get_frame_processors_modules(processor_names)):
            remaining_frame_paths = job_state.remaining_frames(manifest, processor_name, temp_frame_paths)
            if not remaining_frame_paths:
                update_status('Already done in a previous run, skipping...', frame_processor.NAME)
                continue
            progress.begin(STAGES.get(frame_processor.NAME, 'Working'), total=len(remaining_frame_paths))
            update_status('Progressing...', frame_processor.NAME)

            def on_frames_done(done_frame_paths: List[str], _processor_name: str = processor_name) -> None:
                # persist after every finished batch, so a kill loses at most that batch
                with manifest_lock:
                    job_state.mark_frames_done(manifest, _processor_name, done_frame_paths)
                    job_state.save_manifest(target, manifest)

            frame_processor.process_video(source, remaining_frame_paths, on_frames_done=on_frames_done)
            frame_processor.post_process()
        job_state.set_stage(manifest, job_state.STAGE_CREATING)
        job_state.save_manifest(target, manifest)
    progress.begin('Making the video')
    if changeling.globals.keep_fps:
        fps = detect_fps(target)
        update_status(f'Creating video with {fps} FPS...')
        created = create_video(target, fps)
    else:
        update_status('Creating video with 30 FPS...')
        created = create_video(target)
    if not created:
        update_status('Creating the video failed; the progress so far is kept for another try.')
        return False, 'Could not create the video from the processed frames.'
    if changeling.globals.skip_audio:
        move_temp(target, changeling.globals.output_path)
        update_status('Skipping audio...')
    else:
        progress.begin('Putting the sound back')
        if changeling.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        if not restore_audio(target, changeling.globals.output_path):
            update_status('Restoring the audio failed; the progress so far is kept for another try.')
            return False, 'Could not restore the audio.'
    if is_video(target) and changeling.globals.output_path and os.path.isfile(changeling.globals.output_path):
        # only a fully finished job may clean up: the manifest and (unless --keep-frames) the frames
        progress.begin('Cleaning up')
        update_status('Cleaning temporary resources...')
        job_state.discard_manifest(target)
        clean_temp(target)
        update_status('Processing to video succeed!')
        return True, f'Done: {os.path.basename(changeling.globals.output_path)}'
    update_status('Processing to video failed!')
    return False, 'Processing the video failed.'


def blocked() -> None:
    update_status('The target was flagged as NSFW, so it was not processed.')
    # no clean_temp here: only a fully successful job may delete the temp frames and manifest
    if not changeling.globals.headless:  # the window stays open: just this job stops
        progress.finish('Refused: the target was flagged as NSFW.', ok=False)
    sys.exit(3)


def destroy() -> None:
    # the temp directory stays on purpose: its job manifest lets the next run resume
    # instead of extracting and processing every frame again
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    # every processor the window can switch on gets its models now, not just the ones picked
    for frame_processor in get_frame_processors_modules(sorted(set(changeling.globals.frame_processors) | {'face_swapper', 'face_enhancer'})):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if changeling.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()
