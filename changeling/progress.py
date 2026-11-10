"""progress.py - how far a job is, for the window's completion bar.

The job runs on its own thread (and the frames on a pool of them); the window reads this every tenth of a
second and draws it. Nothing here touches the window, so any thread can report.

    begin('Swapping faces', total=600)   a stage with a known length: the bar fills as step() is called
    begin('Extracting frames')           a stage of unknown length: the bar sweeps
    step()                               one more frame done
    finish('Done: output.mp4')           the job's over (ok=False: it failed)
"""
import threading
import time
from typing import Optional

_lock = threading.Lock()
_state = {'busy': False, 'stage': '', 'done': 0, 'total': 0, 'started': 0.0, 'message': '', 'ok': True}


def begin(stage: str, total: int = 0) -> None:
    with _lock:
        _state.update(busy=True, stage=stage, done=0, total=total, started=time.monotonic(), message='')


def step(n: int = 1) -> None:
    with _lock:
        _state['done'] += n


def note(message: str) -> None:
    """A status line (shown under the bar when no stage is running: 'No face in source path detected.')."""
    with _lock:
        _state['message'] = message


def finish(message: str, ok: bool = True) -> None:
    with _lock:
        _state.update(busy=False, stage='', message=message, ok=ok)


def snapshot() -> dict:
    """{busy, stage, done, total, fraction (None: unknown length), eta (seconds or None), message, ok}"""
    with _lock:
        s = dict(_state)
    fraction: Optional[float] = None
    eta: Optional[float] = None
    if s['total']:
        fraction = min(1.0, s['done'] / s['total'])
        elapsed = time.monotonic() - s['started']
        if s['done'] and elapsed > 1:
            eta = elapsed / s['done'] * (s['total'] - s['done'])
    s.update(fraction=fraction, eta=eta)
    return s


def describe(s: dict) -> str:
    """'Swapping faces  240 / 600 frames  40%  1:10 left'"""
    if not s['busy']:
        return s['message']
    if s['fraction'] is None:
        return s['stage'] + ' ...'
    text = f"{s['stage']}  {s['done']} / {s['total']}  {s['fraction'] * 100:.0f}%"
    if s['eta'] is not None:
        m, sec = divmod(int(s['eta']), 60)
        text += f'  {m}:{sec:02d} left'
    return text
