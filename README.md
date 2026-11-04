# VRAMDISK：在 Windows 上创建 VRAM 磁盘

[English](README.en.md) | 简体中文

VRAMDISK 可以通过自有文件系统将 GPU 的显存（VRAM）挂载为 Windows 磁盘。

<img width="928" height="472" alt="VRAMDISK 主界面" src="https://github.com/user-attachments/assets/42a44f7e-3a8c-4ab4-a85b-dad7a1769782" />

数据以类似对象存储的结构保存在 VRAM 中，因此可以利用 GPU 的并行计算能力直接压缩文件、计算哈希值和进行编码。

挂载磁盘后，可通过 GUI 工具面板或 `$VRAMDISK` 内部 API 使用以下 GPU 工具：

| 工具 | 说明 |
|---|---|
| 哈希计算 | 自动选择 GPU 或 CPU，计算 MD5、SHA-1、SHA-256 和 FNV-1a 64 |
| 压缩与解压 | 在 GPU 上创建或解压 tar.zst、tar.lz4、tar.gz 和 zip（需要 nvCOMP） |
| 编码 | 在 GPU 上进行 Base64 和十六进制编码、解码 |
| 全文搜索 | 使用 GPU 扫描磁盘中的所有文件；数据通过 VRAM 带宽读取，无需经过 PCIe |

注意：卸载磁盘或退出进程后，VRAM 中的数据会丢失。通过 GUI 卸载磁盘或退出程序时，可以先将磁盘内容保存为 ZIP 文件。

## 开始使用

**系统要求：** Windows 10 或 11；NVIDIA Maxwell（sm_50）及更新架构的 GPU；CUDA 12.8。

### 1. 安装 WinFsp

从[此处](https://github.com/winfsp/winfsp/releases/download/v2.1/winfsp-2.1.25156.msi)下载并安装 WinFsp。

### 2. 下载程序

从 [Releases](../../releases) 下载 `vramdisk.zip` 并解压。

### 3. 挂载 VRAM 磁盘

运行 `vramdisk.exe`，选择 GPU、挂载位置（盘符或文件夹）和容量，然后点击“Mount”。

<img width="405" height="446" alt="挂载设置界面" src="https://github.com/user-attachments/assets/67bdbd04-631a-4fae-8ac1-0a5cadbcf293" />

### 4. 启用压缩（可选）

安装 nvCOMP 运行时 DLL 后，即可在 GPU 上压缩文件。挂载界面中的“Compress data”选项和挂载后的文件压缩工具都会启用。

可从 NVIDIA [nvCOMP 5.3.0 Windows x64 下载页面](https://developer.nvidia.com/nvcomp-5-3-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)获取兼容版本。

## 常见安装问题

### 提示找不到 `VCRUNTIME140.dll`

请安装 [Microsoft Visual C++ Redistributable（VC++ 运行库）](https://aka.ms/vc14/vc_redist.x64.exe)。

### 提示找不到 WebView2 Runtime

请安装 [WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2?form=MA13LH#download)。选择 Evergreen Standalone Installer 即可。

## 构建

需要 Visual Studio Dev Shell 环境。运行以下命令：

```powershell
npm install            # 仅首次运行：安装 @tauri-apps/cli
.\build-gui.ps1        # 发布构建：生成 src-tauri\target\release\vramdisk.exe
.\build-gui.ps1 dev    # 启动开发版本（含开发者工具）
```

详细开发说明见 [DEV.md](DEV.md)。

## 许可证

本程序采用 MIT 许可证发布。

© 2026 ActiveTK。详见 [LICENSE](LICENSE)。
