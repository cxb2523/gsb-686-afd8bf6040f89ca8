import os
from typing import Any, List, Callable, Optional
import sys
import cv2
import numpy
import threading

# basicsr (a gfpgan dependency) imports torchvision.transforms.functional_tensor,
# which torchvision 0.17+ removed; its functions now live in transforms.functional.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    import torchvision.transforms.functional
    sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.functional

from gfpgan.utils import GFPGANer

import changeling.globals
import changeling.processors.frame.core
from changeling.core import update_status
from changeling.face_analyser import get_many_faces
from changeling.typing import Frame, Face
from changeling.utilities import conditional_download, models_directory, is_image, is_video

FACE_ENHANCER = None
# CodeFormer and GPEN (FaceFusion's ONNX copies), looked for in models/ and then F:\AIModels\faceswap
ONNX_ENHANCERS = {'CodeFormer': 'codeformer.onnx', 'GPEN': 'gpen_bfr_512.onnx'}
ONNX_SESSIONS = {}
FFHQ_512 = numpy.float32([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                          [0.39308822, 0.72541100], [0.61150205, 0.72490465]]) * 512
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'CHANGELING.FACE-ENHANCER'


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_directory(), 'GFPGANv1.4.pth')
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER


def get_device() -> str:
    if 'CUDAExecutionProvider' in changeling.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in changeling.globals.execution_providers:
        return 'mps'
    return 'cpu'


def clear_face_enhancer() -> None:
    global FACE_ENHANCER

    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = models_directory()
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(changeling.globals.target_path) and not is_video(changeling.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def onnx_path(model: str) -> Any:
    file = ONNX_ENHANCERS.get(model)
    if not file:
        return None
    for folder in (models_directory(), r'F:\AIModels\faceswap'):
        if os.path.exists(os.path.join(folder, file)):
            return os.path.join(folder, file)
    return None


def onnx_session(path: str) -> Any:
    with THREAD_LOCK:
        if path not in ONNX_SESSIONS:
            import onnxruntime
            ONNX_SESSIONS[path] = onnxruntime.InferenceSession(path, providers=changeling.globals.execution_providers)
    return ONNX_SESSIONS[path]


def enhance_face_onnx(target_face: Face, temp_frame: Frame, session: Any) -> Frame:
    """CodeFormer or GPEN: the face aligned to 512 x 512 (FFHQ's way), redrawn, and put back with a soft oval."""
    a, _ = cv2.estimateAffinePartial2D(target_face['kps'].astype(numpy.float32), FFHQ_512, method=cv2.LMEDS)
    if a is None:
        return temp_frame
    crop = cv2.warpAffine(temp_frame, a, (512, 512), borderMode=cv2.BORDER_REPLICATE)
    x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(numpy.float32).transpose(2, 0, 1)[None] / 127.5 - 1
    feeds = {i.name: (numpy.array(1.0, numpy.float64) if i.name == 'weight' else x) for i in session.get_inputs()}
    with THREAD_SEMAPHORE:
        y = session.run(None, feeds)[0][0]
    sharp = cv2.cvtColor(((numpy.clip(y, -1, 1) + 1) * 127.5).transpose(1, 2, 0).astype(numpy.uint8), cv2.COLOR_RGB2BGR)
    mask = numpy.zeros((512, 512), numpy.float32)
    cv2.ellipse(mask, (256, 280), (180, 225), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 16)
    h, w = temp_frame.shape[:2]
    back = cv2.invertAffineTransform(a)
    face = cv2.warpAffine(sharp, back, (w, h), borderValue=0.0).astype(numpy.float32)
    alpha = cv2.warpAffine(mask, back, (w, h), borderValue=0.0)[..., None]
    return (face * alpha + temp_frame.astype(numpy.float32) * (1 - alpha)).astype(numpy.uint8)


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    path = onnx_path(changeling.globals.face_enhancer_model)
    if path:
        return enhance_face_onnx(target_face, temp_frame, onnx_session(path))
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, temp_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=True
            )
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str], on_frames_done: Optional[Callable[[List[str]], None]] = None) -> None:
    changeling.processors.frame.core.process_video(None, temp_frame_paths, process_frames, on_frames_done)
