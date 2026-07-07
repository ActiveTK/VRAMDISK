# VRAMDISK: Create VRAM disk on Windows

VRAMDISKは、GPUのVRAMを独自のファイルシステム経由でWindowsにマウントできるアプリです。

<img width="928" height="472" alt="image" src="https://github.com/user-attachments/assets/677a0817-f60f-4a86-bed0-a275abfb7a69" />

VRAM上にはオブジェクトストレージのような構造でデータを保存しているため、GPUの高い並列計算性能を活用してGPU上でそのままファイルを圧縮したり、ハッシュ値を求めたり、エンコードしたりすることができます。

※アンマウント/プロセス終了でデータは消えます。

## 導入方法

前提条件: Windows (10 || 11) && NVIDIA GPU: > Maxwell (sm_50, CUDA 12.8)

### Step1. WinFspをインストール

まず初めに、[こちら](https://github.com/winfsp/winfsp/releases/download/v2.1/winfsp-2.1.25156.msi)からWinFspのインストーラーを入手して導入してください。

### Step2. `vramdisk.exe` をダウンロード

[Releases](../../releases) からデスクトップアプリ `vramdisk.exe` を入手します。単体で動く実行ファイルで、インストーラはありません。コピーしてそのまま実行してください。

### Step3. VRAMディスクをマウント

`vramdisk.exe` を起動し、GPUデバイス / マウント先（ドライブレター || フォルダ）/ サイズを選んで「マウント」を押します。

<img width="405" height="446" alt="image" src="https://github.com/user-attachments/assets/67bdbd04-631a-4fae-8ac1-0a5cadbcf293" />

無事にマウントできました！

### （オマケ）Step4. 圧縮機能を有効化（任意）

nvCOMPのランタイムDLLを導入すると、GPU上でファイルを圧縮できるようになります（マウント画面の「データを圧縮」が押せるようになるのと、マウント後にファイル圧縮ツールが使えるようになります）。

nvCOMPは以下から入手できます。

https://developer.nvidia.com/nvcomp-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

## ビルド方法

Visual Studio Dev Shell環境が必要です。以下のコマンドでビルドできます。

```powershell
npm install            # 初回のみ（@tauri-apps/cliを取得）
.\build-gui.ps1        # リリースビルド → src-tauri\target\release\vramdisk.exe
.\build-gui.ps1 dev    # 開発起動（devtools付き）
```

細かい開発仕様は [DEV.md](DEV.md)をLLMに投げてください。

## ライセンス

このプログラムは The MIT License の下で公開されています。

© 2026 ActiveTK.
https://github.com/ActiveTK/gff/blob/master/LICENSE
