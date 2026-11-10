import os
import sys
import threading
import webbrowser
import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_ALL
from typing import Any, Callable, Tuple, Optional
import cv2
from PIL import Image, ImageOps

import changeling.globals
import changeling.metadata
from changeling import progress
from changeling.face_analyser import get_one_face
from changeling.capturer import get_video_frame, get_video_frame_total
from changeling.face_reference import get_face_reference, set_face_reference, clear_face_reference
from changeling.predictor import predict_frame, clear_predictor
from changeling.processors.frame.core import get_frame_processors_modules
from changeling.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
progress_bar = None
busy_buttons = []
JOB = None
KOFI = 'https://ko-fi.com/cursedentertainment'
GITHUB = 'https://github.com/CursedPrograms/Changeling-Imago'


# todo: remove by native support -> https://github.com/TomSchimansky/CustomTkinter/issues/934
class CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, progress_bar, busy_buttons

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{changeling.metadata.name} {changeling.metadata.version}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())
    # the Changeling logo as the window's icon (after customtkinter puts its own in, ~200 ms)
    logo = resolve_relative_path('../logo.ico')
    if os.name == 'nt' and os.path.exists(logo):
        root.after(300, lambda: root.iconbitmap(logo))

    source_label = ctk.CTkLabel(root, text=None, fg_color=ctk.ThemeManager.theme.get('DropArea').get('fg_color'))
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)
    source_label.drop_target_register(DND_ALL)
    source_label.dnd_bind('<<Drop>>', lambda event: select_source_path(event.data))
    if changeling.globals.source_path:
        select_source_path(changeling.globals.source_path)

    target_label = ctk.CTkLabel(root, text=None, fg_color=ctk.ThemeManager.theme.get('DropArea').get('fg_color'))
    target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)
    target_label.drop_target_register(DND_ALL)
    target_label.dnd_bind('<<Drop>>', lambda event: select_target_path(event.data))
    if changeling.globals.target_path:
        select_target_path(changeling.globals.target_path)

    source_button = ctk.CTkButton(root, text='Select a face', cursor='hand2', command=lambda: select_source_path())
    source_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2', command=lambda: select_target_path())
    target_button.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)

    keep_fps_value = ctk.BooleanVar(value=changeling.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(root, text='Keep target fps', variable=keep_fps_value, cursor='hand2', command=lambda: setattr(changeling.globals, 'keep_fps', not changeling.globals.keep_fps))
    keep_fps_checkbox.place(relx=0.1, rely=0.6)

    keep_frames_value = ctk.BooleanVar(value=changeling.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(root, text='Keep temporary frames', variable=keep_frames_value, cursor='hand2', command=lambda: setattr(changeling.globals, 'keep_frames', keep_frames_value.get()))
    keep_frames_switch.place(relx=0.1, rely=0.65)

    skip_audio_value = ctk.BooleanVar(value=changeling.globals.skip_audio)
    skip_audio_switch = ctk.CTkSwitch(root, text='Skip target audio', variable=skip_audio_value, cursor='hand2', command=lambda: setattr(changeling.globals, 'skip_audio', skip_audio_value.get()))
    skip_audio_switch.place(relx=0.6, rely=0.6)

    many_faces_value = ctk.BooleanVar(value=changeling.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(root, text='Many faces', variable=many_faces_value, cursor='hand2', command=lambda: setattr(changeling.globals, 'many_faces', many_faces_value.get()))
    many_faces_switch.place(relx=0.6, rely=0.65)

    # inswapper draws the new face at 128 x 128, so on its own the face is soft and blocky next to the rest of the
    # picture; GFPGAN redraws it sharp afterwards (slower, most of all on videos)
    sharp_value = ctk.BooleanVar(value='face_enhancer' in changeling.globals.frame_processors)
    sharp_switch = ctk.CTkSwitch(root, text='Sharper face (GFPGAN: higher quality, slower)', variable=sharp_value, cursor='hand2', command=lambda: set_sharper(sharp_value.get()))
    sharp_switch.place(relx=0.1, rely=0.7)
    sharp_model = ctk.CTkSegmentedButton(root, values=['GFPGAN', 'CodeFormer', 'GPEN'], command=lambda v: setattr(changeling.globals, 'face_enhancer_model', v))
    sharp_model.set(changeling.globals.face_enhancer_model)
    sharp_model.place(relx=0.1, rely=0.745, relwidth=0.42)

    start_button = ctk.CTkButton(root, text='Start', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=0.15, rely=0.795, relwidth=0.2, relheight=0.045)

    stop_button = ctk.CTkButton(root, text='Quit', cursor='hand2', command=lambda: destroy())
    stop_button.place(relx=0.4, rely=0.795, relwidth=0.2, relheight=0.045)

    preview_button = ctk.CTkButton(root, text='Preview', cursor='hand2', command=lambda: toggle_preview())
    preview_button.place(relx=0.65, rely=0.795, relwidth=0.2, relheight=0.045)
    busy_buttons = [start_button, preview_button, source_button, target_button]

    gallery_button = ctk.CTkButton(root, text='Gallery', cursor='hand2', command=lambda: open_gallery(root))
    gallery_button.place(relx=0.6, rely=0.745, relwidth=0.3, relheight=0.04)

    # the completion bar: what stage the job's at, how far, and how long it has left
    progress_bar = ctk.CTkProgressBar(root, mode='determinate')
    progress_bar.set(0)
    progress_bar.place(relx=0.1, rely=0.86, relwidth=0.8, relheight=0.02)

    status_label = ctk.CTkLabel(root, text='Pick a face and a target, then Start.', justify='center')
    status_label.place(relx=0.1, rely=0.89, relwidth=0.8)

    link_color = ctk.ThemeManager.theme.get('Link').get('text_color')
    made_by = ctk.CTkLabel(root, text='Cursed Entertainment  ·  GitHub  ·  Ko-fi', justify='center', cursor='hand2', text_color=link_color)
    made_by.place(relx=0.1, rely=0.94, relwidth=0.8)
    made_by.bind('<Button>', lambda event: webbrowser.open(KOFI if event.x > made_by.winfo_width() * 0.7 else GITHUB))
    credit = ctk.CTkLabel(root, text='grown from roop by s0md3v (AGPL-3.0)', justify='center', text_color=('gray50', 'gray45'))
    credit.place(relx=0.1, rely=0.97, relwidth=0.8)

    root.after(100, watch_job)
    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    preview.bind('<Up>', lambda event: update_face_reference(1))
    preview.bind('<Down>', lambda event: update_face_reference(-1))
    return preview


def open_gallery(root) -> None:
    """Everything made with the Changelings (Start still asks where to save; the gallery lists it too)."""
    from gallery import show
    show(root, 'Imago')


def set_sharper(on: bool) -> None:
    """The face enhancer after the swap (on), or the swap alone (off)."""
    processors = [name for name in changeling.globals.frame_processors if name != 'face_enhancer']
    changeling.globals.frame_processors = processors + ['face_enhancer'] if on else processors


def update_status(text: str) -> None:
    """From the window's own thread only (the job reports through changeling.progress)."""
    status_label.configure(text=text)


def run_job(start: Callable[[], None]) -> None:
    """The job on its own thread, so the window stays alive and the bar can move."""
    global JOB
    if JOB is not None and JOB.is_alive():
        return
    progress.begin('Loading the models')
    for button in busy_buttons:
        button.configure(state='disabled')
    JOB = threading.Thread(target=start, daemon=True, name='changeling-job')
    JOB.start()


def watch_job() -> None:
    """Ten times a second: the bar and the line under it follow changeling.progress."""
    s = progress.snapshot()
    if s['busy']:
        if s['fraction'] is None:
            if progress_bar.cget('mode') != 'indeterminate':
                progress_bar.configure(mode='indeterminate')
                progress_bar.start()
        else:
            if progress_bar.cget('mode') != 'determinate':
                progress_bar.stop()
                progress_bar.configure(mode='determinate')
            progress_bar.set(s['fraction'])
    elif JOB is not None:
        if JOB.is_alive():                      # finished reporting, thread still closing
            pass
        else:
            if progress_bar.cget('mode') != 'determinate':
                progress_bar.stop()
                progress_bar.configure(mode='determinate')
            progress_bar.set(1 if s['ok'] else 0)
            for button in busy_buttons:
                button.configure(state='normal')
    text = progress.describe(s)
    if text and status_label.cget('text') != text:
        status_label.configure(text=text)
    ROOT.after(100, watch_job)


def select_source_path(source_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_SOURCE

    if PREVIEW:
        PREVIEW.withdraw()
    if source_path is None:
        source_path = ctk.filedialog.askopenfilename(title='select an source image', initialdir=RECENT_DIRECTORY_SOURCE)
    if is_image(source_path):
        changeling.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(changeling.globals.source_path)
        image = render_image_preview(changeling.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        changeling.globals.source_path = None
        source_label.configure(image=None)


def select_target_path(target_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_TARGET

    if PREVIEW:
        PREVIEW.withdraw()
    clear_face_reference()
    if target_path is None:
        target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=RECENT_DIRECTORY_TARGET)
    if is_image(target_path):
        changeling.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(changeling.globals.target_path)
        image = render_image_preview(changeling.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        changeling.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(changeling.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        changeling.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT

    if is_image(changeling.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save image output file', defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(changeling.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save video output file', defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        changeling.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(changeling.globals.output_path)
        run_job(start)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.unbind('<Right>')
        PREVIEW.unbind('<Left>')
        PREVIEW.withdraw()
        clear_predictor()
    elif changeling.globals.source_path and changeling.globals.target_path:
        init_preview()
        update_preview(changeling.globals.reference_frame_number)
        PREVIEW.deiconify()


def init_preview() -> None:
    PREVIEW.title('Preview [ ↕ Reference face ]')
    if is_image(changeling.globals.target_path):
        preview_slider.pack_forget()
    if is_video(changeling.globals.target_path):
        video_frame_total = get_video_frame_total(changeling.globals.target_path)
        if video_frame_total > 0:
            PREVIEW.title('Preview [ ↕ Reference face ] [ ↔ Frame number ]')
            PREVIEW.bind('<Right>', lambda event: update_frame(int(video_frame_total / 20)))
            PREVIEW.bind('<Left>', lambda event: update_frame(int(video_frame_total / -20)))
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(changeling.globals.reference_frame_number)


def update_preview(frame_number: int = 0) -> None:
    if changeling.globals.source_path and changeling.globals.target_path:
        temp_frame = get_video_frame(changeling.globals.target_path, frame_number)
        if predict_frame(temp_frame):
            sys.exit()
        source_face = get_one_face(cv2.imread(changeling.globals.source_path))
        if not get_face_reference():
            reference_frame = get_video_frame(changeling.globals.target_path, changeling.globals.reference_frame_number)
            reference_face = get_one_face(reference_frame, changeling.globals.reference_face_position)
            set_face_reference(reference_face)
        else:
            reference_face = get_face_reference()
        for frame_processor in get_frame_processors_modules(changeling.globals.frame_processors):
            temp_frame = frame_processor.process_frame(
                source_face,
                reference_face,
                temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)


def update_face_reference(steps: int) -> None:
    clear_face_reference()
    reference_frame_number = int(preview_slider.get())
    changeling.globals.reference_face_position += steps
    changeling.globals.reference_frame_number = reference_frame_number
    update_preview(reference_frame_number)


def update_frame(steps: int) -> None:
    frame_number = preview_slider.get() + steps
    preview_slider.set(frame_number)
    update_preview(preview_slider.get())
