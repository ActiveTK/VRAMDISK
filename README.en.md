# VRAMDISK: Create VRAM disk on Windows

VRAMDISK is an app that mounts your GPU's VRAM as a Windows drive through its own file system.

<img width="928" height="472" alt="image" src="https://github.com/user-attachments/assets/677a0817-f60f-4a86-bed0-a275abfb7a69" />

Data is stored in VRAM in an object-storage-like layout, so the GPU's massive parallelism can be used to compress files, compute hashes, and encode files directly on the GPU.

While a disk is mounted, the following GPU tools are available (from the GUI tool panels or the `$VRAMDISK` internal API).

| Tool | Description |
|---|---|
| Hashing | Computes MD5 / SHA-1 / SHA-256 / FNV-1a 64, automatically dispatched to GPU or CPU |
| Compress / extract | Creates and extracts tar.zst / tar.lz4 / tar.gz / zip on the GPU (requires nvCOMP) |
| Encoding | Encodes and decodes Base64 / hex (hex text) on the GPU |

Note: all data is lost when you unmount or the process exits.

## Getting started

Requirements: Windows (10 || 11) && NVIDIA GPU: > Maxwell (sm_50, CUDA 12.8)

### Step 1. Install WinFsp

First, download and install WinFsp from [here](https://github.com/winfsp/winfsp/releases/download/v2.1/winfsp-2.1.25156.msi).

### Step 2. Download `vramdisk.exe`

Get the desktop app `vramdisk.exe` from [Releases](../../releases). It is a standalone executable with no installer. Just copy it somewhere and run it.

### Step 3. Mount a VRAM disk

Start `vramdisk.exe`, pick a GPU device, a mount point (drive letter || folder), and a size, then press "Mount".

<img width="405" height="446" alt="image" src="https://github.com/user-attachments/assets/67bdbd04-631a-4fae-8ac1-0a5cadbcf293" />

That's it, the disk is mounted!

### (Bonus) Step 4. Enable compression (optional)

Installing the nvCOMP runtime DLL lets you compress files on the GPU (the "Compress data" option on the mount screen becomes available, and so does the file compression tool after mounting).

You can get nvCOMP here:

https://developer.nvidia.com/nvcomp-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

## Building

A Visual Studio Dev Shell environment is required. Build with:

```powershell
npm install            # first time only (fetches @tauri-apps/cli)
.\build-gui.ps1        # release build -> src-tauri\target\release\vramdisk.exe
.\build-gui.ps1 dev    # dev run (with devtools)
```

For the detailed internals, feed [DEV.md](DEV.md) to an LLM.

## License

This program is released under The MIT License.

(c) 2026 ActiveTK.
https://github.com/ActiveTK/gff/blob/master/LICENSE
