import os
from typing import Any, List, Callable, Optional
import cv2
import insightface
import threading

import changeling.globals
import changeling.processors.frame.core
from changeling.core import update_status
from changeling.face_analyser import get_one_face, get_many_faces, find_similar_face
from changeling.face_reference import get_face_reference, set_face_reference, clear_face_reference
from changeling.typing import Face, Frame
from changeling.utilities import conditional_download, models_directory, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'CHANGELING.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = os.path.join(models_directory(), 'inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=changeling.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = models_directory()
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(changeling.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(changeling.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(changeling.globals.target_path) and not is_video(changeling.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if changeling.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if changeling.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if changeling.globals.many_faces else get_one_face(target_frame, changeling.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str], on_frames_done: Optional[Callable[[List[str]], None]] = None) -> None:
    if not changeling.globals.many_faces and not get_face_reference():
        # on a resumed run temp_frame_paths holds only the frames still to do, so the
        # configured reference frame number may point past the end - clamp it
        reference_frame = cv2.imread(temp_frame_paths[min(changeling.globals.reference_frame_number, len(temp_frame_paths) - 1)])
        reference_face = get_one_face(reference_frame, changeling.globals.reference_face_position)
        set_face_reference(reference_face)
    changeling.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames, on_frames_done)
