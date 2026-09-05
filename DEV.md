# VRAMDISKの技術仕様書

VRAMDISK は、NVIDIA GPU の VRAM を揮発性 RAM ディスクとして Windows に
マウントするファイルシステムアプリケーションである。WinFsp による
Windows ファイルシステム統合、64 KiB チャンク単位のストレージ管理、
任意の圧縮・重複排除、GPU 上でのハッシュ処理、仮想内部 API を提供する。

---

## 1. 目的と範囲

VRAMDISK が提供する主要機能は次の通り。

- 連続確保した VRAM 領域を backing store とする Windows マウントポイント。
- WinFsp 経由の通常ファイル・ディレクトリ操作。
- 64 KiB チャンク単位の物理配置管理。
- スパースファイル。
- 任意のチャンク重複排除。
- 任意のチャンク圧縮。
- GPU 上でのハッシュ計算。
- GPU 上でのファイルの Base64 / hex エンコード・デコード。
- マウント済みファイルシステム内に公開される `$VRAMDISK` 仮想 API。
- 共通 Rust エンジンを利用する CLI と Tauri GUI。

VRAMDISK は揮発性ストレージである。データはマウント中のプロセスが保持して
いる間だけ存在し、アンマウントまたはプロセス終了で失われる。

---

## 2. 動作環境

| 項目 | 要件 |
|---|---|
| OS | Windows |
| Rust | MSVC toolchain |
| CUDA | 対象 NVIDIA GPU と互換性のある CUDA Toolkit / CUDA runtime |
| GPU | 十分な VRAM を持つ NVIDIA GPU |
| WinFsp | システムにインストール済みであること |
| Visual Studio | C++ build tools |
| LLVM | bindgen 用 `libclang.dll` |
| nvCOMP | GPU 圧縮を使う場合に必要。未検出時は CPU zstd にフォールバック |

ビルド時は WinFsp bindgen のため、Visual Studio 開発シェル相当の環境と
`LIBCLANG_PATH` が必要である。リポジトリにはこれを自動設定するスクリプトを
含む。

```powershell
.\build.ps1            # Rust ライブラリのビルド確認
.\build.ps1 --release

.\build-gui.ps1        # Tauri アプリケーションの build/dev
.\build-gui.ps1 --release
```

圧縮有効時は、既定パスまたは環境変数 `NVCOMP_DLL` で指定された nvCOMP DLL を
検索する。見つからない場合でも起動は継続し、可能な経路では CPU zstd を使う。

---

## 3. 実行モード

配布される実行ファイルは `vramdisk.exe` である。GUI と CLI の両方をこの単一
バイナリから起動する。

```powershell
vramdisk.exe
vramdisk.exe cli --mount R:
vramdisk.exe cli --dedup --mount R:
vramdisk.exe cli --compress --mount R:
vramdisk.exe cli --compress --dedup --mount R:
vramdisk.exe benchmark
vramdisk.exe cli --bench --size 1GiB --device 1
vramdisk.exe cli --bench-io
```

### CLI ディスパッチ

プロセスの実体は Tauri 側のエントリポイントである。先頭引数が `cli` または
`benchmark` の場合のみ CLI ランナーへ処理を委譲し、GUI は起動しない。
`benchmark` は CLI ベンチマークモードのショートカットである。

実行ファイルは Windows GUI サブシステムのため、CLI モードでは親コンソールへ
接続してから標準入出力を利用する。

### GUI 初期値シード

`cli` / `benchmark` 以外の引数は、GUI の初期値上書きとして緩く解釈できる。
この指定は自動マウントを行わず、明示された値だけを GUI 状態へ反映する。

---

## 4. CLI オプション

| オプション | 既定値 | 説明 |
|---|---:|---|
| `-s, --size <SIZE>` | `max(0.8 x GPU[0] VRAM, 2GiB)` | ディスクサイズ。64 KiB 単位に切り上げる。`2GB`、`512MiB`、バイト数などを受け付ける。 |
| `-c, --compress` | off | チャンク圧縮を有効にする。nvCOMP LZ4 を優先し、利用不可時は CPU zstd を使う。 |
| `-d, --dedup` | off | チャンク重複排除を有効にする。圧縮と併用可能。 |
| `--dedup-trust-hash` | off | dedup の共有判定を FNV-1a hash 一致のみで行い、byte 比較を省く。速いが、書き手を信頼できない環境では誤共有が起きる。`--dedup` 無しでは無効。 |
| `-m, --mount <PT>` | `R:` | ドライブレターまたはディレクトリマウントポイント。 |
| `--device <N>` | `0` | CUDA デバイス序数。 |
| `--bench` | off | 合成ベンチマークを実行して終了する。マウント系オプションとは排他。 |
| `--bench-io` | off | 一時マウントした VRAMDISK 上で実ファイルシステム越しの sequential write/read を計測する。 |

CLI でマウントした場合、Enter、Ctrl-C、またはプロセス終了までマウントを保持
する。標準入力が EOF の場合は、プロセスが終了するまでマウントを維持する。

---

## 5. 全体アーキテクチャ

Rust クレートが VRAM backed filesystem の中核を提供し、Tauri アプリケーション
が GUI、トレイ、CLI ディスパッチ、プロセスライフサイクルを担当する。

```text
lib.rs
  cli.rs          CLI パーサ、サイズパーサ、GUI 初期値スキャン
  cli_run.rs      CLI 実行本体
  bench.rs        合成ベンチマーク
  cuda.rs         VRAM 確保と byte I/O
  chunk.rs        64 KiB 物理チャンクアロケータ
  arena.rs        圧縮 blob 用 byte-granular アロケータ
  lookup.rs       名前空間、メタデータ、論理チャンク配置
  nvcomp.rs       nvCOMP 動的 FFI と batched codec
  gpu_hash.rs     GPU FNV-1a hash kernel wrapper
  api_kernel.rs   仮想 API 用 CUDA kernel
  internal_api.rs $VRAMDISK 仮想 API
  engine.rs       byte-range I/O、sparse、dedup、compression、encode jobs
  fs.rs           WinFsp filesystem 実装

src-tauri/
  main.rs         Tauri entry point、CLI dispatch、tray/window lifecycle
  manager.rs      単一マウント manager thread
  commands.rs     GUI command surface

ui/
  Tauri から読み込む静的 HTML/CSS/JS frontend
```

基本ストレージ単位は次の定数である。

```text
CHUNK_SIZE = 64 KiB
```

VRAM 確保サイズは常にこの倍数に丸められる。

---

## 6. CUDA / VRAM 層

`Vram` は単一の連続 GPU メモリ領域を所有し、byte-addressed な操作を提供する。

- `device_total_mem`
- `device_name`
- `write_at` / `read_at`
- `write_at_async` / `read_at_async`
- `zero_at` / `zero_at_async`
- `copy_within`

host と VRAM の転送経路はサイズに応じて切り替える。

- 小さい転送は固定費を避けるため pageable host memory 経路を使う。
- 大きい転送は最適化経路を使う。
- さらに大きい転送では `cudaHostRegister` による一時 page-lock を試す。
- 登録に失敗した場合は pinned staging buffer にフォールバックする。
- H2D staging は write-combined pinned memory を使う。
- D2H staging は通常 pinned memory を使う。
- 大きな登録済み転送は複数 CUDA copy stream へ分割できる。

WinFsp callback は複数 worker thread から到達し得るため、CUDA 操作の前に
CUDA context を呼び出し thread へ bind する。

---

## 7. ストレージモデル

### 物理チャンク

VRAM 上の物理領域は 64 KiB チャンク単位で管理する。

- 1 bit が 1 物理チャンクを表す。
- `0` は空き、`1` は使用中を表す。
- allocation は cursor 付き first-fit で行う。
- 実チャンク数を超える末尾 padding bit は常に使用中扱いにする。

圧縮済み payload は通常チャンクとは別に、byte-granular な arena に格納する。
arena は圧縮 blob 用の packed 領域を払い出し、参照が消えた blob の領域を解放
する。

### 名前空間とメタデータ

ファイルシステム名前空間は次の構造で管理する。

```text
HashMap<normalized_path, Node>
```

lookup 用のパスキーは大小無視で正規化し、表示名は元のケースを保持する。
ディレクトリ node は子名集合を持つ。各 node はサイズ、属性、タイムスタンプ、
安定 index number、security descriptor、論理チャンク配置を保持する。

タイムスタンプは Windows FILETIME 形式で扱う。

### 論理配置

各ファイルは論理チャンク番号に対応する配置配列を持つ。

```rust
Vec<Option<Placement>>
```

各論理チャンクは次のいずれかである。

- `None`: スパースなゼロ穴。
- `Placement::Raw { chunk }`: VRAM 上の物理 64 KiB チャンク。
- `Placement::Compressed { offset, len, codec }`: 圧縮 arena 内の blob。

対応 codec は次の通り。

- `Lz4`: nvCOMP による GPU LZ4。
- `Zstd`: nvCOMP 不在時などに使う CPU zstd fallback。

---

## 8. StorageEngine

`StorageEngine` は VRAM I/O、名前空間、チャンク割り当て、圧縮、重複排除、
copy-on-write を統合する。

主要操作は次の通り。

- `read`
- `write`
- `set_size`
- `remove`
- `clone_range`

対応する性質は次の通り。

- スパースファイルの成長。
- 複数チャンクにまたがる byte-range read/write。
- 部分チャンクの read-modify-write。
- truncate 時の物理資源解放。
- スパース穴のゼロ埋め read。
- 連続 raw チャンクの大きな転送への結合。

### エラー分類

`EngineError` は、呼び出し側がエラー文字列を検査せずに扱いを決められるよう
分類されている。WinFsp 層はこの分類から `NTSTATUS` を、Jobs API はメッセージを
選ぶ。`Display` は variant 名も Rust の quoting も含まない 1 行の英文を返し、
その文字列がそのまま `result.json` と GUI に出る。

| variant | 意味 | NTSTATUS |
|---|---|---|
| `Lookup` | 名前空間の解決失敗 | lookup 種別ごと |
| `NoSpace` | 空き物理チャンクなし | `STATUS_DISK_FULL` |
| `OutOfVram` | ジョブ用の一時 VRAM 不足（対処を含むメッセージ） | `STATUS_DISK_FULL` |
| `NotAFile` | ファイルを期待した位置がディレクトリ | `STATUS_FILE_IS_A_DIRECTORY` |
| `Cancelled` | 協調キャンセル | `STATUS_ACCESS_DENIED` |
| `Cuda` | CUDA / nvCOMP の実失敗のみ | `STATUS_ACCESS_DENIED` |
| `InvalidInput` | 入力データ・引数が不正（archive framing、base64、path 等） | `STATUS_INVALID_PARAMETER` |
| `Unsupported` | 入力は正しいがこのビルド／構成で扱えない | `STATUS_INVALID_DEVICE_REQUEST` |
| `Internal` | 自前で格納したデータが round-trip しない（不変条件違反） | `STATUS_DATA_ERROR` |

`Cuda` を名乗ってよいのは `cuda()` helper が包む実際の driver / kernel 失敗だけ
である。parse や検証の失敗をここに混ぜると、利用者は「GPU が壊れた」と読んで
見当違いの調査を始める。

### Raw モード

圧縮と重複排除がどちらも無効な場合、連続する full-chunk write は可能な限り
連続物理チャンクとして確保し、大きな H2D 転送として処理する。read も物理的に
連続する raw チャンクをまとめて D2H 転送する。

### Dedup モード

重複排除は content hash と参照数で管理する。

- `refcount`: raw 物理チャンクごとの参照数。
- `hash_index`: content hash から placement への索引。
- `chunk_hash` / `compressed_hash`: 逆引きメタデータ。

content hash には GPU と CPU で同じ結果になる 2 段階 FNV-1a 64-bit を使う。
full-chunk write は hash 一致候補を `hash_index` から探す。

**hash の一致だけでは共有しない。** 共有を確定する前に、候補の中身と書き込もうと
しているバイト列を厳密に比較する。

- raw placement: 候補チャンク 64 KiB を device-to-host で読み戻して比較する。
- compressed placement: blob を解凍して比較する。

不一致なら候補を捨てて通常どおり新しいチャンクへ格納する（エラーにはしない）。
拒否件数は `dedup.rejected_chunks` として trace に出る。

FNV-1a は衝突耐性を持たず、衝突は意図的に構成できる。hash の再計算だけで確認して
いた頃は、任意のバイト列を書ける主体が無関係なファイルのチャンクを自分のものへ
別名化でき、そのファイルは黙って攻撃者のデータを読み出すようになっていた。
バイト比較はこれを塞ぐ。

batch 経路では、候補を採用する瞬間に `hash_index` を読み直す。batch 開始時の
スナップショットを使うと、同一 batch 内で解放済みのチャンクを採用してしまい、
allocator が空きとみなしているチャンクをファイルが指す状態を作れる
（`[A,B]` を `[B,A]` で上書きする書き込みで発生する）。batch 中に索引の項目は
削除されるだけで置き換えられないため、その場で引ける候補はスナップショットと
同じ placement であることが保証される。

検証のコストは重複が実際に見つかったときだけ発生する。unique write は影響を
受けず、全重複の write は 64 KiB ごとの D2H 読み戻しぶん約 3〜4 倍遅くなる
（実測で約 6 GB/s → 約 1.5〜1.9 GB/s）。すべての書き手を信頼できる環境では
`--dedup-trust-hash` で従来の hash のみの判定に戻せる。

共有 placement は部分書き込み時に直接変更しない。部分書き込みでは次の
copy-on-write 手順を取る。

1. 現在のチャンクを materialize または device-to-device copy する。
2. 非共有 placement に対して変更を適用する。
3. 参照数と配置メタデータを更新する。

物理チャンクまたは圧縮 blob は参照数が 0 になった時だけ解放する。

### 圧縮モード

圧縮はチャンク単位で行う。full-chunk group は可能な限り batch 化する。

- nvCOMP batch は最大 256 チャンクを処理する。
- LZ4 圧縮で payload が縮む場合は圧縮 arena へ格納する。
- 圧縮で縮まない場合は raw チャンクとして格納する。
- nvCOMP が利用できない場合は CPU zstd level 3 を使う。

単発経路では圧縮前に次のヒューリスティクスを適用する。

1. 非ゼロバイトが 1024 未満なら raw で格納する。
2. 8 箇所の 256 byte window でシャノンエントロピーをサンプリングする。
3. 平均 entropy が 7.2 bit/byte 以上なら、既圧縮または乱数的なデータとして
   raw で格納する。
4. それ以外は圧縮を試し、縮んだ場合だけ compressed placement にする。

compressed read では、LZ4 blob を arena から codec scratch へ device-to-device
copy し、GPU 上で batch decompression する。部分 read では 64 KiB 全体を host へ
戻さず、要求 slice だけを D2H 転送する。

### 圧縮 + 重複排除

圧縮と重複排除を同時に有効化した場合は、圧縮前の平文チャンク hash を共有判定
に使う。

- raw placement と compressed placement の両方を共有対象にする。
- full-chunk write では既存共有候補を先に batch 検証する。
- miss したチャンクだけを圧縮または raw 格納する。
- 圧縮 blob は immutable object として扱う。
- 部分書き込みは materialize、変更、再登録の流れで処理する。

`clone_range` は chunk-aligned range であれば payload をコピーせず、placement と
参照数の更新で共有する。端数部分は通常 read/write にフォールバックする。

---

## 9. GPU ハッシュ

`GpuHasher` は埋め込み PTX kernel を読み込み、VRAM 上の 64 KiB チャンクを GPU で
hash する。

アルゴリズムは 2 段階 FNV-1a 64-bit である。

- 第1段: 64 KiB チャンクを 256 個の 256 byte segment として個別 hash する。
- 第2段: 256 個の中間 hash 値をさらに FNV-1a で hash する。

batch 実行では 1 チャンクを 1 CUDA block に割り当てる。

```text
gridDim = (num_chunks, 1, 1)
```

host と GPU の間で転送するのは offset 配列と出力配列だけであり、チャンク本体は
VRAM に残る。この性質を dedup 候補検証や内部 API の hash 処理で利用する。

---

## 10. WinFsp ファイルシステム

`VramDiskFs` は WinFsp の filesystem context を実装する。

実装対象の主な操作は次の通り。

- security lookup / descriptor retrieval。
- open / create / close。
- read / write。
- append、write-to-EOF、constrained I/O。
- overwrite。
- file size 変更。
- rename。
- basic info 更新。
- delete-pending と cleanup による削除。
- file info / volume info。
- `.` と `..` を含む directory read。
- flush。
- duplicate extents 用 device control。

ボリューム特性は次の通り。

- sector size: 512 bytes。
- allocation unit: 64 KiB。
- case-insensitive lookup。
- case-preserving names。
- persistent ACLs。
- user-mode device control。

### ロック戦略

WinFsp の callback 全体を大域直列化せず、共有状態を mutex で保護する。
現行の engine 操作は `StorageEngine` mutex 内で直列化され、CUDA stream の利用も
この範囲で整合させる。

長時間ジョブがこの mutex を保持し続けるとボリューム全体が応答しなくなるため、
job executor は engine 呼び出しを分割し、その境界ごとに lock を取り直す。

| job | lock 保持の粒度 |
|---|---|
| CPU hash | 32 MiB window ごと |
| GPU hash | GPU launch budget（約 0.1 秒ぶん）で区切った batch ごと |
| archive / encode | 1 回の engine 呼び出し |

lock を手放す境界では、並行 writer によって window / batch 間で観測時点の
異なるスナップショットが混ざり得る。ジョブの応答性を優先してこれを許容する。

engine は `RwLock` で保護する。メタデータのみを読む callback（stat、
ディレクトリ列挙、security 照会、volume 情報）は shared guard を取り、互いに
並行して走る。状態を変更するものは exclusive guard を取る。

`read` callback はまず shared guard で `StorageEngine::read_into_shared` を
試す。要求が触れる論理チャンクがすべて `Raw` またはスパース穴であれば、
`Vram::read_at`（`&self` かつ内部で同期済み）だけで応答できるため、読み取りは
互いに並行して走る。圧縮 placement を 1 つでも含む場合は、事前走査の段階で
`Ok(None)` を返す（出力バッファには一切書き込まない）。呼び出し側は shared
guard を解放してから exclusive guard を取り直し、ファイルを解決し直して
`read_into` で再実行する。`RwLock` は再入不可であり、両方を同時に保持すると
自スレッドで deadlock するため、この解放は必須である。

trace カウンタは `AtomicU64` であり、shared guard 上の読み取りも exclusive
path と同一に計上される（`trace_snapshot()` は従来通り plain `u64` の
`EngineTrace` を返す）。

Windows の `RwLock` は SRWLOCK であり fair ではないため、理屈の上では読み手が
書き手を待たせ続け得る。WinFsp の dispatch thread 数は有界で、これらの callback
は短時間で終わるため、実際には到達しない。

より細かい粒度（ファイル単位・領域単位の lock と複数 CUDA stream の併用）は
未実装である。write callback、および圧縮データを含む read は依然として
exclusive guard 上で直列化される。

### メタデータキャッシュ

WinFsp のキャッシュタイムアウトは `file_info` のみ設定されており、`dir_info` /
`volume_info` / `security` / `stream_info` は 0（キャッシュしない）だった。
これらは 1 つ miss するごとに user-mode への往復が発生する。実測でこのスタックの
往復は約 14 マイクロ秒で、cold path への `GetFileAttributesEx` はその往復を
何度も踏む。特に security は path 解決のたびに問い合わされる。

現在は 5 つとも 1000 ms を設定している。代償は staleness で、別ハンドル経由の
ACL / サイズ変更が観測されるまで最大 1 秒かかる。これは元から `file_info` が
受け入れていた条件と同じである。

### open handle と rename

open handle は現在パスと delete-pending 状態を保持する。rename 時は対象サブツリー
配下の open handle のパスも更新し、handle 経由の後続操作が rename 後の対象を
指し続けるようにする。

### ACL

security descriptor は node ごとに self-relative Windows security descriptor として
保持する。create 時には WinFsp から渡された descriptor を保存し、security 更新時は
既存 descriptor に modification descriptor を適用する。

明示 descriptor を持たない node と `$VRAMDISK` 仮想 node は、マウントしたユーザー
自身と SYSTEM / Administrators だけにフルアクセスを与える既定 SDDL を使う。
`<user>` はプロセス token の user SID である。

```text
O:<user>G:BAD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;<user>)
```

token の照会に失敗した場合に限り、Everyone フルアクセス
（`O:BAG:BAD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;WD)`）へ fallback し、その旨を
stderr に警告する。開けないボリュームより緩いボリュームの方がましだという判断で
あり、実際にこの経路へ落ちることは想定していない。

### CUDA kernel の事前コンパイル

API kernel の PTX は NVRTC で実行時にコンパイルする。これに約 6 秒かかり、
以前は最初の hash / archive / encode job がその全額を負担していた。

現在はプロセスにつき 1 回だけコンパイルし（`OnceLock`）、`StorageEngine::new`
から detached thread で先行して開始する。マウント時に同期実行すると mount が
0.20 秒から 5.60 秒に伸びるため、背景で走らせている。mount は 0.23 秒のまま、
最初の job は module load と calibration の約 0.2 秒だけを負担する。

### WinFsp DLL 解決

WinFsp は delay-load し、実行時にインストール先レジストリまたは既定パスから
`winfsp-x64.dll` をフルパスで読み込む。これにより WinFsp の bin ディレクトリが
`PATH` に無い環境でも起動できる。

---

## 11. 内部仮想 API

マウントされたファイルシステムのルート直下に、読み取り専用の仮想 system
directory を公開する。

```text
\$VRAMDISK
```

この namespace は通常の lookup table には実 node を作らず、WinFsp 層で専用
dispatch する。通常の変更操作が `$VRAMDISK` 配下または `$VRAMDISK` 自体を対象に
した場合は拒否する。

### 情報取得 API

| パス | 内容 |
|---|---|
| `\$VRAMDISK\help.txt` | API の簡易説明。 |
| `\$VRAMDISK\stats.txt` | 容量、物理使用量、圧縮・dedup 状態などの text 表示。 |
| `\$VRAMDISK\stats.json` | stats の JSON 表現。 |
| `\$VRAMDISK\trace.txt` | read/write 経路や batch 回数などの counter 表示。 |
| `\$VRAMDISK\trace.json` | trace の JSON 表現。 |
| `\$VRAMDISK\chunks.json\<path>` | 通常ファイルの論理チャンク配置 JSON。 |

`chunks.json` は sparse/raw/compressed、論理 offset、物理 chunk または blob 座標、
codec、refcount、共有状態、利用可能な content hash を返す。

### Jobs API

job は次の pending descriptor を `CREATE_NEW` で作成し、JSON を書き込んで close
することで投入する。

```text
\$VRAMDISK\jobs\pending\<id>.json
```

`id` は ASCII 英数字、`.`、`-`、`_` のみを許可する。投入後は次の path で状態や
結果を取得する。

```text
\$VRAMDISK\jobs\<id>\status.json
\$VRAMDISK\jobs\<id>\result.json
\$VRAMDISK\jobs\<id>\wait
\$VRAMDISK\jobs\<id>\cancel
```

`status.json` は `id` / `state` / `terminal` / `submitted_at_ms` /
`updated_at_ms` / `error` に加えて `progress` を持つ。

```json
"progress": {"done_bytes": 1234, "total_bytes": 5678}
```

`progress` は executor が申告した時点の値であり、まだ何も報告していない job や、
総量を事前に知り得ない job では `null` になる。`total_bytes` は見積もりであり
途中で上方修正され得るため、消費側は `done_bytes` が古い `total_bytes` を
上回る場合を想定すること。job が成功した時点で `done_bytes` は `total_bytes` に
揃えられる。

対応 job family は次の通り。

- `hash`
- `archive.compress`
- `archive.extract`
- `encode`
- `search`

#### Hash jobs

descriptor 例:

```json
{
  "op": "hash",
  "algorithm": "sha256",
  "paths": ["\\data"],
  "recursive": true
}
```

対応 algorithm:

- `md5`
- `sha1`
- `sha256`
- `fnv1a64`

hash は file ごとに GPU/CPU を自動選択する。小さい file で、全 placement が
raw/sparse/LZ4 の場合は CUDA API kernel の init/update/final 段で処理する。
巨大 file、または CPU zstd fallback chunk を含む file は CPU streaming path を使い、
`StorageEngine::read` で 32 MiB window ごとに materialize して RustCrypto hash へ
逐次投入する。

hash kernel は message に対して逐次であるため、GPU 側は **1 スレッド**で走る。
1 ファイルのスループットでは CPU に勝てないのが普通で、実測では RTX 4070 の
SHA-256 が 0.055 GB/s に対し、SHA-NI を持つ i9-12900K は 1.64 GB/s だった
（約 30 倍）。GPU が効くのは多数のファイルを 1 launch にまとめる batch 経路である。

そのため初回 hash 時には、同じ約 1 MiB の sample で **GPU と CPU の両方**を測り、
per-byte でどちらが速いかを比較して振り分けを決める。

- GPU hash launch budget: `GPU throughput x 0.10 s` を `[4 MiB, 512 MiB]` に clamp。
- CPU routing threshold:
  - GPU が CPU 以上に速い場合は上限なし（`u64::MAX`）。
    `hash_file_gpu_cancellable` は launch budget 単位で分割して流すので、
    1 回の kernel に無制限の仕事が入ることはない。
  - GPU の方が遅い場合は `GPU throughput x 0.25 s` と launch budget の小さい方。
    batch を共有できる大きさのファイルだけを GPU へ入れる。

以前は GPU 側のスループットだけを見て閾値を決めており（上限 128 MiB の clamp
付き）、CPU と比較していなかったため、GPU の方が遅いサイズ帯まで GPU へ送って
いた。

raw チャンクは VRAM address descriptor として kernel に渡す。LZ4 compressed
チャンクは nvCOMP で device scratch に D2D 解凍し、その scratch address を hash
kernel に渡す。スパース穴は GPU kernel 内でゼロ列として合成する。

CPU zstd fallback で格納された compressed chunk は GPU kernel へは渡さず、暗黙
エラーではなく CPU hash path に振り替える。

#### Archive jobs

圧縮 descriptor 例:

```json
{
  "op": "archive.compress",
  "format": "tar.zst",
  "paths": ["\\data"],
  "output": "\\out.tar.zst",
  "recursive": true
}
```

展開 descriptor 例:

```json
{
  "op": "archive.extract",
  "format": "tar.zst",
  "archive": "\\out.tar.zst",
  "output_dir": "\\restore"
}
```

対応 format:

- `tar.zst`
- `tar.lz4`
- `tar.gz`
- `zip`

#### CRC32

zip は payload ごとに CRC-32 を必要とする。`vramdisk_crc32_many` は batch 要素
1 つにつき 1 スレッドしか割り当てないため、範囲全体を 1 要素として渡すと単一
スレッドが逐次走査することになる（実測 256 MiB で 7.98 秒、約 5 MB/s）。

現在は範囲を `CRC32_LANE_BYTES`（64 KiB）の lane に分割し、`CRC32_LAUNCH_BYTES`
（512 MiB）ごとに 1 launch でまとめて計算したうえで、GF(2) 上の CRC-32 連結
（zlib の `crc32_combine` 相当。lane 長が全て等しいので operator ladder は 1 回
だけ構築する）で畳み込む。同じ 256 MiB が 11.2 ms（約 24 GB/s）になった。

compressed placement を含む範囲は、lane ごとに一時 chunk を要するため hash の
launch budget に従う。

archive jobs では、対応できる範囲でファイル本文を GPU 上に保持する。CPU は
descriptor、archive header、path metadata を扱う。source file と input archive は
raw / sparse / compressed placement を受け付ける。LZ4 placement は nvCOMP で
device scratch へ展開し、CPU zstd fallback placement は CPU で解凍して H2D
アップロードする。対応外の path 形式や、一時 raw 展開ぶんの VRAM が足りない場合は
原因と対処が分かる明示エラーにする。

format ごとの処理方針は次の通り。

- `tar.zst`: VRAM 上に連続 tar stream を構築し、nvCOMP Zstd で圧縮・展開する。
- `tar.lz4`: CPU で LZ4 frame header を構築し、payload block を nvCOMP LZ4 で処理する。
- `tar.gz`: gzip multi-member layout と nvCOMP Deflate payload を使う。
- `zip`: ZIP Deflate method 8、GPU CRC32、ZIP64 record、VRAMDISK 専用 chunk-size table を使う。

archive job が失敗またはキャンセルされた場合、書きかけの出力 archive と
`\.__vramdisk_*` staging temp file は自動的に除去される。展開途中の出力
ファイルは診断用に残る。

#### Encode jobs

GPU 上で Base64 / hex のエンコード・デコードを行う。descriptor 例:

```json
{
  "op": "encode",
  "codec": "base64",
  "direction": "encode",
  "input": "\a.bin",
  "output": "\a.b64"
}
```

- `codec`: `base64` | `hex`。`direction`: `encode` | `decode`。
- 固定サイズの staging パス（既定 48 MiB、空き VRAM に応じて縮小）単位で、
  入力 slice を連続 raw staging に materialize → GPU kernel で変換 →
  出力ファイルへ device-to-device で scatter する。ファイル全長の連続
  VRAM 確保は不要で、raw / sparse / compressed のどの placement も入力にできる。
- Base64 は 3 byte → 4 文字（`=` padding、行折り返しなし）。decode は
  単一行の標準 Base64 のみ受け付け、末尾の ASCII 空白（改行等）は無視する。
  `=` を含む最終グループだけ CPU で処理する。
- hex は小文字で出力し、decode は大文字小文字両方を受け付ける。
- 不正な入力文字は kernel の status flag 経由で明示エラーになる。
- 失敗・キャンセル時は書きかけの出力ファイルと staging temp を除去する。

#### Search jobs

VRAM 上のファイル本文を GPU で全文検索する。descriptor 例:

```json
{
  "op": "search",
  "pattern": "TODO",
  "paths": ["\src"],
  "ignore_case": false,
  "max_offsets": 64
}
```

- `pattern` は正規表現ではなくリテラル。UTF-8 バイト列として照合する。
  任意のバイト列を探す場合は `pattern` の代わりに `pattern_hex`（偶数長の 16 進）
  を使う。上限は 256 バイト。
- `paths` 省略時はボリューム全体を対象にする。他の job family と異なりこれを
  既定にしているのは、全体検索が最も普通の使い方だからである。
- `ignore_case` は ASCII の `A-Z` のみ畳む。対象は任意のバイト列であり、
  特定のエンコーディングのテキストとは限らないため、Unicode の case folding は
  行わない。
- `max_offsets` は 1 ファイルあたりに**報告する**位置の数（既定 64）。
  一致件数は常に正確で、位置だけが切り詰められる。切り詰めた場合は
  `offsets_truncated` が真になる。

この機能がこのハードウェアで意味を持つ理由は、データが既に VRAM にあることに尽きる。
CPU 側の grep は同じ仕事をするのにボリューム全体を PCIe 越しに引き出す必要があるが、
GPU カーネルはデバイスメモリ帯域でそのまま走査できる。

実装はファイルのチャンクが連続とは限らず圧縮されている場合もあるため、
`SEARCH_STAGE_BYTES`（64 MiB）の窓へ device-to-device で materialize してから
走査する。窓は `窓長 - (pattern 長 - 1)` ずつ進めるので、継ぎ目をまたぐ一致も
検出される。カーネルの最後の候補位置は次の窓の最初の候補位置のちょうど 1 つ手前
なので、二重計上は起きない。

カーネルは 1 候補位置につき 1 スレッドの grid-stride で、pattern の**先頭と末尾**の
バイトで先に落とす。大多数を占める不一致がロード 2 回で済むため、全体が
帯域律速になる。

---

## 12. ベンチマーク

`--bench` は合成ベンチマークを実行して終了する。主な計測対象は次の通り。

- raw VRAM H2D / D2H bandwidth。
- raw storage engine write/read throughput。
- dedup unique write / duplicate write throughput。
- compressible / incompressible データでの compressed engine write/read throughput。
- GPU LZ4 と CPU zstd の圧縮・展開 throughput。
- GPU FNV-1a batch throughput と single-chunk latency。

`--bench-io` は一時的に VRAMDISK をマウントし、実ファイルシステム越しの sequential
write/read を次の 4 モードで計測する。

- raw
- compress
- dedup
- compress+dedup

### 検証

実 VRAM を確保する（`Vram::new`）、CUDA kernel を起動する、nvCOMP を読み込む、の
いずれかを行う test は `gpu-tests` feature で gate する。この feature は既定で
有効なので、開発機での `cargo test` は従来どおり全件走る。GPU の無い CI は
`--no-default-features` を付けて、GPU 不要なぶんだけを実行する。

| 手段 | 実行場所 | 対象 |
|---|---|---|
| `cargo test` | ローカル（GPU 必須） | 全 unit test |
| `cargo test --no-default-features` | CI | GPU 不要な unit test（名前空間、chunk allocator、圧縮 blob arena、job registry、CLI parse、CRC table） |
| `cargo clippy --all-targets -- -D warnings` | CI | lint。GPU 必須の test も型検査・lint される |
| `cargo build` | CI | `vramdisk.exe`（GUI と CLI は同一バイナリ） |
| `scripts\e2e_robustness.ps1` | ローカル（GPU 必須） | 通常の filesystem 面 |
| `scripts\e2e_jobs.ps1` | ローカル（GPU 必須） | `$VRAMDISK` 仮想 API 全面 |

GitHub-hosted runner には NVIDIA GPU が無いため、GPU 必須の unit test と E2E は
CI で実行できない。リリース前に手動で回す。詳細は `scripts/README.md`。

---

## 13. デスクトップ GUI

GUI は Tauri v2 アプリケーションであり、CLI と同じ Rust engine を利用する。
GUI 管理下で同時に存在できるマウントは 1 つだけである。

### 実行モデル

- 単一プロセス。
- システムトレイ常駐。
- 専用 manager thread が唯一の `MountedVramDisk` を所有する。
- GUI command は channel 越しに manager へ依頼する。
- 非 `Send` な mount handle は manager thread から出さない。
- GUI mount の CUDA 所有も manager thread に固定する。

### window / tray

未マウント時に window を閉じるとプロセスは終了する。マウント中に window を閉じる
と window は hidden になり、マウントは維持される。tray menu から window 表示、
unmount、tool panel 表示、exit を実行できる。exit 時は active mount を撤去する。

マウント成功時は main window を隠し、通知 dialog を表示し、その dialog が閉じられた
後に Explorer で mount point を開く。

### mount point

GUI は次の mount point をサポートする。

- drive letter。
- directory。

directory mount では、対象 path は存在しないか空 directory でなければならない。
WinFsp は directory mount point の生存期間を所有するため、既存の空 directory を
選んだ場合は mount 直前に削除し、mount 失敗時には復元する。

### frontend

frontend は Tauri が直接読み込む静的 HTML/CSS/JS である。別途 Node bundling は
不要である。Tauri の global API で backend command を invoke し、mount state event
を購読する。

主な画面は次の通り。

- mount 設定。
- mounted status / stats。
- hash job。
- archive compress / extract。
- encode（Base64 / hex）。
- search（GPU 全文検索）。

window はリサイズ可能（既定 404x580、最小 380x460）。

### UI 言語

UI は日本語 / 英語の二言語対応で、右上のセレクタで切り替えられる。

- 既定はシステムの表示言語からの自動判定（`sys-locale`、ja 以外は en）。
- 明示的な選択は `HKCU\Software\VRAMDISK` の `UiLanguage`（`ja` / `en`）へ保存され、次回以降はそれが優先される。
- 切り替えは window 内の全ラベル、tray menu、native dialog（unmount 確認・マウント完了通知など）へ即時反映される。
- frontend の文字列は `ui/app.js` の `I18N` 辞書、backend の文字列は
  `src-tauri/src/main.rs` の `tr()` / `trf()` にある。`tr()` は静的文字列、
  `trf()` は `{0}` / `{1}` プレースホルダを持つメッセージ用で、置換は
  1 パスのみ行う（Windows のパスが `{1}` を含み得るため、置換結果を再置換しない）。
- Tauri command と manager が返すエラーメッセージも同じ catalog を通す。
  command は `UiLang` state から、manager thread は Tauri より先に起動されるため
  リクエストに言語を載せて受け取る。
- 未知のキーはキー名そのものを返す。空文字を返すと、tray の項目やダイアログが
  無言で空欄になり、翻訳漏れではなくアプリの故障に見える。
- engine crate（`vramdisk`）が返すエラーは英語のまま通過する。`EngineError` の
  `Display` は技術的な 1 行メッセージであり、翻訳対象にしていない。

### Tauri commands

| Command | 内容 |
|---|---|
| `list_gpus` | CUDA device ordinal、name、VRAM、default size を返す。 |
| `list_free_drives` | 未使用 drive letter を返す。 |
| `browse_folder` | native folder picker を開く。 |
| `mount(cfg)` | VRAM 確保、StorageEngine 作成、WinFsp mount を実行する。 |
| `unmount` | active mount を撤去する。 |
| `mount_status` | active mount status または `null` を返す。 |
| `stats` | `\$VRAMDISK\stats.json` を読み取る。 |
| `hash_job` | hash job を投入し job id を返す。 |
| `archive_compress_job` | archive compression job を投入し job id を返す。 |
| `archive_extract_job` | archive extraction job を投入し job id を返す。 |
| `encode_job` | Base64/hex encode job を投入し job id を返す。 |
| `search_job` | 全文検索 job を投入し job id を返す。 |
| `job_status` | 指定 job の `status.json` を返す。 |
| `job_result` | 指定 job の `result.json` を返す。 |
| `job_cancel` | 指定 job のキャンセルを要求する。 |
| `get_ui_language` | 現在の UI 言語（`ja` / `en`）を返す。 |
| `set_ui_language` | UI 言語を registry へ保存し、tray menu を再構築する。 |
| `browse_export_zip` | ZIP 保存先を選ぶダイアログを開く。 |
| `export_zip` | ボリューム全体をホスト上の ZIP へ書き出す。 |
| `export_cancel` | 実行中の export をキャンセルする。 |
| `quit_app` | window 側から終了処理を完了させる。 |

job 系 command は同期ブロックしない。frontend は job id を受け取って
`job_status` をポーリングし、terminal になったら `job_result` を読む。これにより
数十 GB 級のジョブも UI を塞がず、GUI から安全にキャンセルできる。

実行中の表示は次の通り。

- `status.json` の `progress` が使える場合は、確定的なプログレスバー、
  処理済み／総バイト数、パーセント、および残り時間の推定を表示する。
- `progress` が `null`、`total_bytes` が 0、値が不正のいずれかの場合は経過秒数
  のみの表示へ fallback する。
- 残り時間の推定は、経過が短すぎる間や進捗率が低すぎる間は雑音が大きいので
  出さない。
- 中止ボタンは常に表示する。

終了判定は `state` フィールドで行う（`cancelled` か否かをエラーメッセージの
文字列一致で判定してはならない）。

hash / archive の path 入力は active mount point 相対に正規化する。mount point 外の
絶対 path は拒否する。

### ZIP エクスポート

アンマウントおよび終了の直前に、ボリュームの内容をホスト上の ZIP として保存する
選択肢を提示する。

- 対象は window の「アンマウント」ボタン、tray の「アンマウント」、tray の「終了」の
  3 経路すべて。いずれも window 内のモーダルで「ZIP に保存して〜」「保存せず〜」
  「キャンセル」の 3 択を出す。window を表示できない場合のみ従来の 2 択 native
  dialog へ fallback する。
- export は engine の GPU ZIP writer ではなく、ホスト側で `std::fs` によりボリューム
  を走査してストリーミング書き出しする。GPU 経路は出力 ZIP を VRAM ディスク上に
  作るため、archive とほぼ同サイズの空き VRAM を要求する。「データを失わないための
  保存」がディスク満杯時にちょうど失敗するのは受け入れられない。
- `$VRAMDISK` はボリューム root 直下でのみ合成される仮想ディレクトリなので、
  深さ 0 でのみ大文字小文字を無視して除外する。同名のユーザーディレクトリが
  下位階層にあればそれは実データとして格納する。
- 進捗は `export-progress` イベントで通知する。payload は jobs API の `progress` と
  同じ `{"done_bytes": N, "total_bytes": M}` 形状で、frontend は job 用の
  プログレスバー実装をそのまま再利用する。総量は事前走査で確定させる。
- 個々のファイルの読み取り失敗は致命的にせず収集して報告する。export が完全に
  成功した場合のみアンマウント／終了へ進む。キャンセル・失敗・一部失敗のいずれでも
  ボリュームはマウントしたまま残し、データを回収できるようにする。

---

## 14. 堅牢性と安全性

WinFsp callback は FFI 境界をまたぐため、panic がプロセスや mount 全体を巻き込ま
ないようにする必要がある。VRAMDISK は次の安全規則を持つ。

- 1 ファイルが要求できる論理チャンク数を volume 総チャンク数以下に制限する。
- offset と length の加算は checked arithmetic で行う。
- `\a` から `\a\b` のような自己サブツリー rename を拒否する。
- mutex poison は `into_inner` で復帰し、後続 callback の連鎖 panic を避ける。
- job worker は executor の panic を catch して job を failed にする。
- 未投入（`Receiving`）の job に対する `wait` はブロックせず status を返す。
- zip / gzip / tar / LZ4 frame の parse は全ての short read を長さ検証する。
- rename で置換された宛先 file の placement は engine 経由で解放される
  （lookup 層だけで rename すると VRAM が leak する）。
- 大文字小文字のみの rename は表示名を更新する（case-preserving）。
- job registry は上限到達時に最古の terminal job を evict する。
- unsupported な block clone path は安全に失敗させる。
- `$VRAMDISK` は通常の filesystem mutation API からは read-only として扱う。

Windows / WinFsp 構成では NVIDIA GPUDirect Storage / cuFile は利用しない。

---

## 15. 公開仕様上の制約

- ストレージは揮発性であり、アンマウントまたはプロセス終了で内容は失われる。
  GUI からのアンマウント／終了時には ZIP としてホストへ保存する選択肢を出すが、
  プロセスの異常終了、ホストのクラッシュ、GPU リセットに対する保護は無い。
- dedup は FNV-1a 64-bit hash で候補を索引し、共有を確定する前に内容を byte 比較
  する。したがって既定では、意図的な hash 衝突による誤共有は起きない。
  `--dedup-trust-hash` を指定した場合に限り byte 比較を省き、その場合は
  任意のバイト列を書ける主体による誤共有（データ化け）が可能になる。
  このフラグはすべての書き手を信頼できる環境専用である。
- GUI 管理下では同時に 1 つのマウントのみをサポートする。
- GPU 圧縮には互換性のある nvCOMP DLL が必要である。
- hash API は large file と zstd fallback chunk を自動で CPU へ振り分ける。
  archive jobs は raw / sparse / compressed placement を透過処理するが、compressed
  source を raw に展開する一時 VRAM が不足すると明示エラーになる。
- volume の既定 SD はマウントしたユーザー自身と SYSTEM / Administrators のみに
  フルアクセスを与える。ただし `$VRAMDISK` の jobs / hash API はファイル単位の
  DACL を確認しないため、明示的に緩い DACL を設定したファイルであっても、
  ボリュームを開ける主体からは内容を読み出せる。
- directory mount point は WinFsp が mount lifetime を所有し、unmount 時に削除される。
- block clone は OS / WinFsp から source handle path を復元できる環境でのみ有効に働く。
- GUI には合成ベンチマーク起動ビューを提供しない。
