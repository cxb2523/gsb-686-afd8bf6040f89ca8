[![Twitter: @NorowaretaGemu](https://img.shields.io/badge/X-@NorowaretaGemu-blue.svg?style=flat)](https://x.com/NorowaretaGemu)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

<br>
<div align="center">
  <a href="https://ko-fi.com/cursedentertainment">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi" style="width: 20%;"/>
  </a>
</div>
<br>

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python%20-%23323330.svg?&style=for-the-badge&logo=python&logoColor=white"/>
</div>

<p align="center"><img src="https://raw.githubusercontent.com/CursedPrograms/Changeling-Studio/main/site/logo.png" alt="Changeling" width="120"></p>

<div align="center">
  <img alt="TensorFlow" src="https://img.shields.io/badge/tensorflow%20-%23323330.svg?&style=for-the-badge&logo=tensorflow&logoColor=white"/>
</div>
<div align="center">
    <img alt="Git" src="https://img.shields.io/badge/git%20-%23323330.svg?&style=for-the-badge&logo=git&logoColor=white"/>
  <img alt="PowerShell" src="https://img.shields.io/badge/PowerShell-%23323330.svg?&style=for-the-badge&logo=powershell&logoColor=white"/>
  <img alt="Shell" src="https://img.shields.io/badge/Shell-%23323330.svg?&style=for-the-badge&logo=gnu-bash&logoColor=white"/>
  <img alt="Batch" src="https://img.shields.io/badge/Batch-%23323330.svg?&style=for-the-badge&logo=windows&logoColor=white"/>
  </div>
<br>

# Changeling Imago

**Website:** https://cursedprograms.github.io/Changeling-Imago/

*Imago* is Latin for a likeness, and the Romans' word for the wax masks cast from a person's face and kept in the family hall: a copy of a face that someone could wear. (It's also where "image" comes from.)

![A target photo, the face to put on it, and Changeling's result](samples/changeling_sample.jpg)

Put a face from one photo onto another photo or a whole video. *Part of the Changeling family, all together in [Changeling Studio](https://github.com/CursedPrograms/Changeling-Studio):
**Imago** ·
[Imago Realtime](https://github.com/CursedPrograms/Changeling-Imago-Realtime) ·
[Tiresias](https://github.com/CursedPrograms/Changeling-Tiresias) (gender) ·
[Chronos](https://github.com/CursedPrograms/Changeling-Chronos) (age) ·
[Natio](https://github.com/CursedPrograms/Changeling-Natio) (where in the world) ·
[Cosmeta](https://github.com/CursedPrograms/Changeling-Cosmeta) (makeup, hair, beards) ·
[Echo](https://github.com/CursedPrograms/Changeling-Echo) (voices) ·
[Tonsor](https://github.com/CursedPrograms/Changeling-Tonsor) (hairstyles) ·
realtime: [Tiresias](https://github.com/CursedPrograms/Changeling-Tiresias-Realtime) ·
[Chronos](https://github.com/CursedPrograms/Changeling-Chronos-Realtime) ·
[Natio](https://github.com/CursedPrograms/Changeling-Natio-Realtime) ·
[Cosmeta](https://github.com/CursedPrograms/Changeling-Cosmeta-Realtime).*

Pick the face, pick the target, press
**Start**, and a completion bar shows what it's doing (loading, checking, extracting frames, swapping
faces, making the video), how far it is and how long it has left. **Sharper face** (on by default) runs GFPGAN after the swap: the swap model draws the new face at 128 x 128, so on its own the face looks soft and blocky next to the rest of the picture; GFPGAN redraws it sharp. Switch it off for quicker tests, most of all on videos.

Changeling Imago grew out of [roop](https://github.com/s0md3v/roop) by s0md3v, and like roop it's free software
under the **GNU AGPL-3.0** (see `LICENSE`). Its NSFW check stays on: explicit targets are refused.

Use it on faces you have the right to use: AI-generated people, yourself, or people who agreed to it.

## Getting started

You need **Python 3.9 - 3.12** (3.11 is best; get it from [python.org](https://www.python.org/downloads/)) and **ffmpeg** on your PATH for videos. Python 3.13 and newer won't work (TensorFlow and PyTorch don't support them yet), but they can stay installed: the scripts find a 3.9 - 3.12 through the `py` launcher.

Then just run it:

| Windows | PowerShell | Linux / macOS / Git Bash |
|---|---|---|
| `run.bat` | `.\run.ps1` | `./run.sh` |

The first run sets everything up by itself, then opens the app:

1. makes a virtual environment, `psdenv`, in this folder
2. installs the packages from `requirements.txt` (several GB the first time: PyTorch and TensorFlow, so give it a while)
3. downloads the models into `models/` (`inswapper_128.onnx`, `GFPGANv1.4.pth`)

After that it starts straight away. Running it again only redoes what's missing or changed, so if a download fails halfway, run it again and it carries on.

- **The GPU:** with an NVIDIA card it uses CUDA automatically (no CUDA Toolkit or cuDNN to install: PyTorch brings its own). Add `--execution-provider cpu` to force the CPU.
- **Older cards (GTX 10xx, 4 GB):** they work. CUDA 12 needs NVIDIA driver **531 or newer**; check with `nvidia-smi` ("Driver Version") and update from nvidia.com if it's older, or the swap quietly falls back to the slow CPU. Out of video memory? Add `--execution-threads 2`. The face enhancer (GFPGAN) is the slow part: leave it off for tests.
- **Set up without opening the app:** `setup.bat`, `.\setup.ps1` or `./setup.sh`.
- **A prompt inside the environment:** `activate_env.bat`.
- **PowerShell won't run scripts?** Once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **Start over:** delete the `psdenv` folder and run it again.

### Without the app (headless)

Give a source face, a target and an output, and it swaps without opening the window:

```
run.bat -s face.png -t clip.mp4 -o out.mp4
run.bat -s face.png -t clip.mp4 -o out.mp4 --keep-fps
```

The swap refuses explicit targets: the NSFW check (`changeling/predictor.py`) stays on.

<br>

## Usage

Start the program with arguments:

```
python run.py [options]

-h, --help                                                                 show this help message and exit
-s SOURCE_PATH, --source SOURCE_PATH                                       select an source image
-t TARGET_PATH, --target TARGET_PATH                                       select an target image or video
-o OUTPUT_PATH, --output OUTPUT_PATH                                       select output file or directory
--frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]                    frame processors (choices: face_swapper, face_enhancer, ...)
--keep-fps                                                                 keep target fps
--keep-frames                                                              keep temporary frames
--skip-audio                                                               skip target audio
--many-faces                                                               process every face
--reference-face-position REFERENCE_FACE_POSITION                          position of the reference face
--reference-frame-number REFERENCE_FRAME_NUMBER                            number of the reference frame
--similar-face-distance SIMILAR_FACE_DISTANCE                              face distance used for recognition
--temp-frame-format {jpg,png}                                              image format used for frame extraction
--temp-frame-quality [0-100]                                               image quality used for frame extraction
--output-video-encoder {libx264,libx265,libvpx-vp9,h264_nvenc,hevc_nvenc}  encoder used for the output video
--output-video-quality [0-100]                                             quality used for the output video
--max-memory MAX_MEMORY                                                    maximum amount of RAM in GB
--execution-provider {cpu} [{cpu} ...]                                     available execution provider (choices: cpu, ...)
--execution-threads EXECUTION_THREADS                                      number of execution threads
-v, --version                                                              show program's version number and exit
```

<br>

### Headless

Using the `-s/--source`, `-t/--target` and `-o/--output` argument will run the program in headless mode.

---

<br>
<div align="center">
© Cursed Entertainment 2026
</div>
<br>
<div align="center">
<a href="https://cursed-entertainment.itch.io/" target="_blank">
    <img src="https://raw.githubusercontent.com/CursedPrograms/cursedentertainment/main/images/logos/logo-wide-grey.png"
        alt="CursedEntertainment Logo" style="width:250px;">
</a>
</div>
<br>
<div align="center">
  <a href="https://github.com/SynthWomb" target="_blank">
    <img src="https://raw.githubusercontent.com/SynthWomb/synth.womb/main/logos/synthwomb07.png" alt="SynthWomb" style="width:200px;"/>
  </a>
</div>
