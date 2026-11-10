"""Says which GPU Changeling will use, and warns when the NVIDIA driver is too old for CUDA 12.

Stdlib only, never fails: setup runs it every time, and a problem here is a warning, not a stop.
"""
import platform
import shutil
import subprocess

# CUDA 12.1 (the PyTorch and onnxruntime-gpu builds in requirements.txt) needs at least these.
MIN_DRIVER = 531 if platform.system() == 'Windows' else 530


def main():
    if not shutil.which('nvidia-smi'):
        print('GPU: no NVIDIA card found - Changeling runs on the CPU (slow, but it works).')
        return
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
        name, memory, driver = [part.strip() for part in out.split(',')]
        major = int(driver.split('.')[0])
    except Exception:
        print('GPU: nvidia-smi is there but didn\'t answer - Changeling will try CUDA anyway.')
        return
    print(f'GPU: {name}, {int(memory) // 1024} GB, driver {driver}')
    if major < MIN_DRIVER:
        print(f'  WARNING: this driver is too old for CUDA 12 (needs {MIN_DRIVER} or newer).')
        print('  Update it from https://www.nvidia.com/Download/index.aspx - until then the swap')
        print('  falls back to the CPU and is very slow.')
    elif int(memory) < 5000:
        print('  4 GB card: if it runs out of video memory, add  --execution-threads 2')


if __name__ == '__main__':
    main()
