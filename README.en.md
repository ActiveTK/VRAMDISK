# VRAMDISK: Create VRAM disk on Windows

VRAMDISK is an app that mounts your GPU's VRAM as a Windows drive through its own file system.

<img width="928" height="472" alt="image" src="https://github.com/user-attachments/assets/42a44f7e-3a8c-4ab4-a85b-dad7a1769782" />

Data is stored in VRAM in an object-storage-like layout, so the GPU's massive parallelism can be used to compress files, compute hashes, and encode files directly on the GPU.

While a disk is mounted, the following GPU tools are available (from the GUI tool panels or the `$VRAMDISK` internal API).

| Tool | Description |
|---|---|
| Hashing | Computes MD5 / SHA-1 / SHA-256 / FNV-1a 64, automatically dispatched to GPU or CPU |
| Compress / extract | Creates and extracts tar.zst / tar.lz4 / tar.gz / zip on the GPU (requires nvCOMP) |
| Encoding | Encodes and decodes Base64 / hex (hex text) on the GPU |

Note: data in VRAM is lost when you unmount or the process exits. When you unmount or quit from the GUI, you get the option to save the drive's contents to a ZIP file on your PC first.

## Getting started

Requirements: Windows (10 || 11) && NVIDIA GPU: > Maxwell (sm_50, CUDA 12.8)

### Step 1. Install WinFsp

First, download and install WinFsp from [here](https://github.com/winfsp/winfsp/releases/download/v2.1/winfsp-2.1.25156.msi).

### Step 2. Download the app

Get `vramdisk.zip` from [Releases](../../releases) and extract it.

### Step 3. Mount a VRAM disk

Start `vramdisk.exe`, pick a GPU device, a mount point (drive letter || folder), and a size, then press "Mount".

<img width="405" height="446" alt="image" src="https://github.com/user-attachments/assets/67bdbd04-631a-4fae-8ac1-0a5cadbcf293" />

That's it, the disk is mounted!

### (Bonus) Step 4. Enable compression (optional)

Installing the nvCOMP runtime DLL lets you compress files on the GPU (the "Compress data" option on the mount screen becomes available, and so does the file compression tool after mounting).

You can get nvCOMP here:

https://developer.nvidia.com/nvcomp-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

## Installation errors

### "The code execution cannot proceed because VCRUNTIME140.dll was not found"

Install the Microsoft Visual C++ Redistributable (VC++ runtime) from:

https://aka.ms/vc14/vc_redist.x64.exe

### "Could not find the WebView2 Runtime"

Install the WebView2 runtime from the link below. The Evergreen Standalone Installer is fine.

https://developer.microsoft.com/en-us/microsoft-edge/webview2?form=MA13LH#download

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
https://github.com/ActiveTK/VRAMDISK/blob/main/LICENSE
