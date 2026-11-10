import threading
from typing import Any, Optional, List
import cv2
import insightface
import numpy

import changeling.globals
from changeling.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=changeling.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            faces = _get_padded(frame)
        return faces
    except ValueError:
        return None


def _get_padded(frame: Frame) -> List[Face]:
    """The detector misses a face that fills the whole picture (close-up portraits):
    look again with a border round it, then move the face back into the frame's coordinates."""
    height, width = frame.shape[:2]
    pad = max(height, width) // 2
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    faces = get_face_analyser().get(padded)
    for face in faces:
        for key in ('bbox', 'kps', 'landmark_2d_106', 'landmark_3d_68'):
            points = face.get(key)
            if points is not None:
                points = points.copy()
                if key == 'bbox':
                    points -= pad
                else:
                    points[..., :2] -= pad
                face[key] = points
    return faces


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < changeling.globals.similar_face_distance:
                    return face
    return None
