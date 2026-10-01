# screenshots/

すべて実機の実行結果である。合成した画像や、数字を差し替えたものは無い。

計測環境: Intel Core i9-12900K / 128 GB / **NVIDIA GeForce RTX 4070 (12 GB)** /
Windows 11 Pro 22635 / WinFsp 2025。

GUI のキャプチャは WebView2 の DevTools プロトコル（`Page.captureScreenshot`）で
`deviceScaleFactor=2` を指定して撮っている。Win32 の画面キャプチャは WebView の
初回描画と競合し、DPI 仮想化の影響も受けて白画面になることがある。

| ファイル | 内容 |
|---|---|
| `01-setup.png` | セットアップ画面。ドライブレターと容量を選ぶだけ |
| `02-mounted.png` | マウント済み画面。使用量と GPU 処理メニュー |
| `03-search-form.png` | 全文検索のフォーム |
| `04-search-result.png` | 1.00 GB を 0.4 秒で走査、**1298 万件ヒット**、10,139 MB/s（下記参照） |
| `05-hash-result.png` | SHA-256（逐次アルゴリズムなので CPU に振り分けられる） |
| `06-archive-form.png` | 圧縮のフォーム |
| `07-archive-result.png` | **1.00 GB → 41.87 MB を 0.5 秒、2052 MB/s**（nvCOMP deflate） |
| `08-encode-result.png` | Base64 エンコード |
| `09-mounted-with-data.png` | ジョブ実行後のマウント済み画面 |
| `10-teardown-prompt.png` | アンマウント前の ZIP 保存プロンプト。VRAM は揮発性なので、消す前に救出する導線 |
| `12-setup-compress-dedup.png` | 圧縮と重複排除を有効にしたセットアップ |
| `13-setup-english.png` | 英語 UI のセットアップ |
| `14-mounted-english.png` | 英語 UI。**論理 2.00 GB が実使用 74.63 MB（1.8%）**、重複排除で 1.00 GB、圧縮で 1.93 GB 節約 |
| `15-explorer.png` | エクスプローラー。左の一覧に `RamDisk (R:)` と `VRAMDISK (V:)` が並ぶ |
| `16-cli-bench.png` | `vramdisk cli --bench` の実行画面 |
| `16-cli-bench.txt` | 同じベンチのフル出力（画面に収まらない先頭部分を含む） |
| `20-crystaldiskmark-vramdisk.png` | CrystalDiskMark の改造版。下記参照 |

## CrystalDiskMark の改造版について

`20-crystaldiskmark-vramdisk.png` は普通の CrystalDiskMark ではない。バックエンドを
DiskSpd から `vramdisk.exe cdm-bench` に差し替えたフォーク
（`K:\dev\2026\CrystalDiskMarkForVRAMDISK`、`src/cdm_bench.rs`）である。

**ファイルベンチマークでは GPU 内で完結する転送を測れない。** DiskSpd も
CrystalDiskMark も、読み書きするバッファはベンチマークプロセスのアドレス空間、
つまりシステムメモリにある。だから VRAM ディスクに向ければ必ず PCIe をまたぎ、
同じ計測を RAM ディスクに向ければまたがない。これはツールの実装ではなく Win32 の
ファイル API の性質なので、パッチでは変えられない。

そこでこのフォークはファイル層を捨て、VRAM 領域を「ディスク」として read は
そこからのコピー、write はそこへのコピーとし、どちらも device-to-device で行う。
CrystalDiskMark 側は終了コードでスループット、名前付き共有メモリでレイテンシを
受け取るだけなので、その規約さえ守れば UI はそのまま使える。

画面に出ている `C:` というターゲット表示とテストサイズは無関係である。そこには
一切読み書きしていない。

## 数字を引用するときの注意

- `04-search-result.png` のスループットは**その 1 回の実測値**であって、代表値ではない。
  同じ検索を 3 回連続で回したときの実測は **752 / 24,976 / 10,139 MB/s** と大きく振れた。
  GUI から単発で押す使い方だと、コピーエンジンしか使っていない間に GPU が低クロックに
  落ちており、最初の数カーネルがその状態で走るためである（原因の切り分けは `../hikaku.md` の
  §7 に実測付きで書いてある）。**連続して回したときの定常値**が知りたければ
  `16-cli-bench.txt` の `[5] Full-Text Search` を見ること。カーネル単体で 195.65 GB/s、
  CPU スキャン比 45.7 倍である。
  スライドで「速い」と言うなら CLI の数字を、「実際に押すとこう見える」と言うなら
  この画像を使うのが正確。
- マウント後の最初の GPU ジョブだけ約 5 秒かかる。NVRTC のコンパイル待ちで、
  背景スレッドが終わる前にジョブが来た場合に発生する。2 回目以降は 10〜11 ms。
- 3 者比較（VRAMDISK / GpuRamDrive / RAM ディスク）の詳細は `../hikaku.md`。
