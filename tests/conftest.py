"""Test setup: changeling.core pulls in heavy native dependencies (onnxruntime, tensorflow,
cv2, insightface, the UI toolkit, ...) that the tests never exercise. Anything not installed
is replaced by a minimal stub module so `import changeling.core` works everywhere."""
import importlib.util
import sys
import types


def _ensure_module(name: str, **attributes) -> types.ModuleType:
    """The real module when importable, otherwise a stub with the given attributes."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is not None:
        return importlib.import_module(name)
    module = types.ModuleType(name)
    module.__path__ = []  # let submodule stubs (insightface.app.common) resolve under it
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _Anything:
    """Stand-in for any class the UI module references at import time."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Anything()

    def __getattr__(self, name):
        return _Anything()


def _any_attribute(name):
    if not name.startswith('__'):
        return _Anything
    raise AttributeError(name)


class _Face:
    pass


class _Model:
    pass


class _TkinterDnD:
    DnDWrapper = object


_ensure_module('onnxruntime', get_available_providers=lambda: ['CPUExecutionProvider'])
_ensure_module('tensorflow')
_ensure_module('cv2')
_ensure_module('psutil')
_ensure_module('insightface')
_ensure_module('insightface.app')
_ensure_module('insightface.app.common', Face=_Face)
_ensure_module('customtkinter', CTk=_Anything, __getattr__=_any_attribute)
_ensure_module('tkinterdnd2', TkinterDnD=_TkinterDnD, DND_ALL='*')
_ensure_module('opennsfw2')
_ensure_module('tf_keras', Model=_Model)
