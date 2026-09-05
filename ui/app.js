// VRAMDISK GUI frontend.
//
// Uses the global Tauri API (app.withGlobalTauri = true) so no bundler/import
// step is required. Only one disk can ever be mounted at a time, so the UI is
// a simple screen switch: setup (nothing mounted) <-> mounted (+ hash /
// archive / encode tool panels reachable from it).
//
// GPU jobs are asynchronous: the backend returns a job id immediately, the
// frontend polls `job_status` (keeping the UI responsive and the job
// cancellable via `job_cancel`), and reads `job_result` once terminal.

"use strict";

const invoke = window.__TAURI__.core.invoke;
const listen = window.__TAURI__.event.listen;

const el = (id) => document.getElementById(id);

// GB/MB use the Windows convention (binary, 1024-based) so the size field
// agrees with the default hint and the GPU's reported VRAM.
const UNIT_BYTES = { MB: 1024 ** 2, GB: 1024 ** 3 };

const JOB_POLL_MS = 400;

let currentStatus = null; // MountStatus (from backend) or null
let gpuList = []; // cached from list_gpus()

// --- i18n -------------------------------------------------------------------
//
// The language is persisted by the backend (registry, HKCU\Software\VRAMDISK)
// and defaults to the Windows display language; `navigator.language` is only
// the fallback if the backend command itself fails.

const I18N = {
  ja: {
    device: "GPU",
    target: "マウント先",
    driveLetter: "ドライブレター",
    folder: "フォルダ",
    browse: "参照...",
    capacity: "容量",
    options: "オプション",
    compressOpt: "データを圧縮（nvCOMP / GPU）",
    dedupOpt: "同一内容のチャンクを共有",
    compressHint:
      "nvCOMP が見つからないため、この GUI では圧縮を利用できません（CLI では CPU zstd にフォールバックします）",
    mount: "マウント",
    statFiles: "ファイル数",
    statLogical: "論理データ",
    statDedup: "共有による節約",
    statPacked: "圧縮による節約",
    toolsLabel: "マウント中のファイルを GPU で処理",
    toolHash: "ハッシュ計算",
    toolArchive: "圧縮・展開",
    toolEncode: "エンコード",
    unmount: "アンマウント",
    back: "← 戻る",
    hashPath: "対象（ファイル / フォルダ）",
    algorithm: "アルゴリズム",
    recursive: "サブフォルダも含める",
    hashRun: "計算する",
    cancel: "中止",
    compressTab: "圧縮",
    extractTab: "展開",
    archiveSource: "対象（フォルダは常に再帰）",
    archiveOutput: "出力先アーカイブ",
    format: "形式",
    formatZst: "tar.zst（推奨・高速）",
    archiveInput: "展開するアーカイブ",
    archiveOutdir: "展開先フォルダ",
    encodeTab: "エンコード",
    decodeTab: "デコード",
    codec: "形式",
    hexOption: "hex（16進テキスト）",
    encodeInput: "変換するファイル",
    encodeOutput: "出力ファイル",

    noCudaOption: "CUDA デバイスが見つかりません",
    noGpuError:
      "NVIDIA GPU と CUDA ドライバが見つかりません。ドライバを更新して再起動してください。",
    sizeDefaultHint: "空欄で {0}",
    sizePlaceholder: "既定",
    sizeInvalid: "サイズが不正です",
    sizeTooBig: "サイズが GPU の VRAM 容量 ({0}) を超えています",
    folderPickFail: "フォルダを選択できません: {0}",
    nvcompToolTitle: "nvCOMP が見つからないため利用できません",
    chooseDrive: "ドライブレターを選択してください",
    chooseFolder: "フォルダを指定してください",
    mounting: "マウント中…",
    mountFail: "マウントできません: {0}",
    cancelBtn: "キャンセル",
    unmounting: "アンマウント中…",
    unmountFail: "アンマウントできません: {0}",
    quitting: "終了しています…",
    quitFail: "終了できません: {0}",
    teardownTitle: "アンマウントの前に",
    teardownBody:
      "アンマウントするとドライブ上のデータは全て失われます。\nZIP ファイルに保存してから続行できます。",
    teardownQuitTitle: "終了の前に",
    teardownQuitBody:
      "終了するとドライブ上のデータは全て失われます。\nZIP ファイルに保存してから続行できます。",
    saveAndUnmount: "ZIP に保存してアンマウント",
    unmountWithoutSaving: "保存せずアンマウント",
    saveAndQuit: "ZIP に保存して終了",
    quitWithoutSaving: "保存せず終了",
    exporting: "ZIP に保存中…",
    exportCancelled: "保存を中止しました。データはそのままです。",
    exportDone: "{0} に保存しました（{1} ファイル）",
    exportFail: "保存できません: {0}",
    exportPartial: "{0} 件を保存できませんでした。データを失わないよう、そのままにしました。",
    exportMoreFailures: "ほか {0} 件",
    badgeCompress: "圧縮",
    badgeDedup: "共有",
    usageUsed: "使用 {0} / {1}",
    interrupted: "アンマウントされたため中断しました",
    jobStatusFail: "失敗: ジョブの状態を確認できません",
    processing: "GPU で処理中… {0} 秒",
    progressLabel: "進捗",
    progressBytes: "{0} / {1}",
    progressPct: "{0}%",
    etaSeconds: "残り約 {0} 秒",
    etaMinutes: "残り約 {0} 分",
    etaHours: "残り約 {0} 時間 {1} 分",
    jobFailed: "失敗: {0}",
    jobError: "処理に失敗しました",
    cancelled: "中止しました",
    done: "完了（{0} 秒）",
    cancelling: "中止しています…",
    resultKey: "結果",
    noFiles: "対象ファイルがありません",
    runCompress: "圧縮を実行",
    runExtract: "展開を実行",
    outputRequired: "出力先を入力してください",
    archiveRequired: "アーカイブのパスを入力してください",
    formatUnknown: "拡張子から形式を判定できません（.tar.zst / .tar.lz4 / .tar.gz / .zip）",
    kvOutput: "出力",
    kvArchive: "アーカイブ",
    kvOutdir: "展開先",
    kvFileCount: "ファイル数",
    kvInputSize: "入力サイズ",
    kvArchiveSize: "アーカイブサイズ",
    kvOutputSize: "出力サイズ",
    kvThroughput: "スループット",
    runEncode: "エンコードを実行",
    runDecode: "デコードを実行",
    inputRequired: "変換するファイルを指定してください",
    encodeOutputRequired: "出力ファイルを指定してください",
    stepGpus: "GPU の列挙",
    stepDrives: "ドライブの列挙",
    stepNvcomp: "nvCOMP の確認",
    stepCli: "起動オプションの反映",
    stepMount: "マウント状態の取得",
    stepEvents: "イベント購読",
    stepFail: "{0}に失敗しました: {1}",
    bootFail: "初期化に失敗しました: {0}",
  },
  en: {
    device: "GPU",
    target: "Mount point",
    driveLetter: "Drive letter",
    folder: "Folder",
    browse: "Browse...",
    capacity: "Capacity",
    options: "Options",
    compressOpt: "Compress data (nvCOMP / GPU)",
    dedupOpt: "Share identical chunks",
    compressHint:
      "nvCOMP was not found, so compression is unavailable in this GUI (the CLI falls back to CPU zstd).",
    mount: "Mount",
    statFiles: "Files",
    statLogical: "Logical data",
    statDedup: "Saved by dedup",
    statPacked: "Saved by compression",
    toolsLabel: "Process mounted files on the GPU",
    toolHash: "Hash",
    toolArchive: "Compress / extract",
    toolEncode: "Encode",
    unmount: "Unmount",
    back: "← Back",
    hashPath: "Target (file or folder)",
    algorithm: "Algorithm",
    recursive: "Include subfolders",
    hashRun: "Compute",
    cancel: "Cancel",
    compressTab: "Compress",
    extractTab: "Extract",
    archiveSource: "Source (folders recurse)",
    archiveOutput: "Output archive",
    format: "Format",
    formatZst: "tar.zst (recommended, fast)",
    archiveInput: "Archive to extract",
    archiveOutdir: "Destination folder",
    encodeTab: "Encode",
    decodeTab: "Decode",
    codec: "Format",
    hexOption: "hex (hex text)",
    encodeInput: "Input file",
    encodeOutput: "Output file",

    noCudaOption: "No CUDA device found",
    noGpuError:
      "No NVIDIA GPU / CUDA driver found. Update the driver and restart.",
    sizeDefaultHint: "Leave empty for {0}",
    sizePlaceholder: "default",
    sizeInvalid: "Invalid size",
    sizeTooBig: "Size exceeds the GPU's VRAM ({0})",
    folderPickFail: "Could not pick a folder: {0}",
    nvcompToolTitle: "Unavailable because nvCOMP was not found",
    chooseDrive: "Choose a drive letter",
    chooseFolder: "Enter a folder",
    mounting: "Mounting…",
    mountFail: "Could not mount: {0}",
    cancelBtn: "Cancel",
    unmounting: "Unmounting…",
    unmountFail: "Could not unmount: {0}",
    quitting: "Exiting…",
    quitFail: "Could not exit: {0}",
    teardownTitle: "Before unmounting",
    teardownBody:
      "Unmounting loses everything on the drive.\nYou can save it to a ZIP file first.",
    teardownQuitTitle: "Before exiting",
    teardownQuitBody:
      "Exiting loses everything on the drive.\nYou can save it to a ZIP file first.",
    saveAndUnmount: "Save to ZIP, then unmount",
    unmountWithoutSaving: "Unmount without saving",
    saveAndQuit: "Save to ZIP, then exit",
    quitWithoutSaving: "Exit without saving",
    exporting: "Saving to ZIP…",
    exportCancelled: "Saving cancelled. Nothing was discarded.",
    exportDone: "Saved to {0} ({1} files)",
    exportFail: "Could not save: {0}",
    exportPartial: "{0} item(s) could not be saved, so nothing was discarded.",
    exportMoreFailures: "{0} more",
    badgeCompress: "compressed",
    badgeDedup: "dedup",
    usageUsed: "{0} of {1} used",
    interrupted: "Interrupted by unmount",
    jobStatusFail: "Failed: cannot read the job's status",
    processing: "Processing on the GPU… {0} s",
    progressLabel: "Progress",
    progressBytes: "{0} / {1}",
    progressPct: "{0}%",
    etaSeconds: "About {0} s left",
    etaMinutes: "About {0} min left",
    etaHours: "About {0} h {1} min left",
    jobFailed: "Failed: {0}",
    jobError: "The operation failed",
    cancelled: "Cancelled",
    done: "Done ({0} s)",
    cancelling: "Cancelling…",
    resultKey: "Result",
    noFiles: "No matching files",
    runCompress: "Compress",
    runExtract: "Extract",
    outputRequired: "Enter an output path",
    archiveRequired: "Enter the archive's path",
    formatUnknown: "Cannot infer the format from the extension (.tar.zst / .tar.lz4 / .tar.gz / .zip)",
    kvOutput: "Output",
    kvArchive: "Archive",
    kvOutdir: "Extracted to",
    kvFileCount: "Files",
    kvInputSize: "Input size",
    kvArchiveSize: "Archive size",
    kvOutputSize: "Output size",
    kvThroughput: "Throughput",
    runEncode: "Encode",
    runDecode: "Decode",
    inputRequired: "Enter the file to convert",
    encodeOutputRequired: "Enter an output file",
    stepGpus: "GPU enumeration",
    stepDrives: "drive enumeration",
    stepNvcomp: "nvCOMP detection",
    stepCli: "applying launch options",
    stepMount: "reading the mount state",
    stepEvents: "event subscription",
    stepFail: "{0} failed: {1}",
    bootFail: "Initialization failed: {0}",
  },
};

let LANG = "ja";

// Missing keys degrade to the key name, never to undefined/"": an untranslated
// or typo'd key must still show *something* rather than silently blanking a
// label, and the key name makes the omission obvious instead of invisible.
function t(key, ...args) {
  let s = (I18N[LANG] && I18N[LANG][key]) || I18N.ja[key] || String(key);
  for (let i = 0; i < args.length; i++) {
    s = s.replaceAll(`{${i}}`, String(args[i]));
  }
  return s;
}

// Re-render every localized string for the current LANG: static labels via
// data-i18n, then the handful of dynamic bits that reflect current state.
function applyLanguage() {
  document.documentElement.lang = LANG;
  el("lang-select").value = LANG;
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  for (const node of document.querySelectorAll("[data-i18n-label]")) {
    node.setAttribute("aria-label", t(node.dataset.i18nLabel));
  }
  el("size-value").placeholder = t("sizePlaceholder");
  updateSizeHint();
  applyNvcompAvailability(nvcompOk);
  setArchiveMode(archiveMode());
  setEncodeDirection(encodeDirection(), { keepOutput: true });
  renderTeardownTexts();
  if (currentStatus) {
    renderMountedHead(currentStatus);
    pollStats();
  }
}

async function switchLanguage(lang) {
  LANG = lang;
  applyLanguage();
  try {
    await invoke("set_ui_language", { language: lang });
  } catch (e) {
    console.error("set_ui_language", e);
  }
}

// --- helpers ---------------------------------------------------------------

function formatSize(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let v = Number(bytes);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return i === 0 ? `${bytes} B` : `${v.toFixed(2)} ${units[i]}`;
}

function setText(id, msg, kind) {
  const s = el(id);
  s.textContent = msg || "";
  s.className = "status-text" + (kind ? " " + kind : "");
}

function screenName() {
  for (const s of ["setup", "mounted", "hash", "archive", "encode"]) {
    if (!el("screen-" + s).hidden) return s;
  }
  return null;
}

function showScreen(name) {
  for (const s of ["setup", "mounted", "hash", "archive", "encode"]) {
    el("screen-" + s).hidden = s !== name;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function kvRow(key, val) {
  return `<div class="kv-row"><span class="kv-key">${escapeHtml(key)}</span><span class="kv-val">${escapeHtml(val)}</span></div>`;
}

// --- config persistence (frontend-only, via localStorage) ------------------

const CFG_KEY = "vramdisk.lastConfig";

function saveConfig() {
  const cfg = {
    device: el("device").value,
    mountMode: mountMode(),
    drive: el("drive").value,
    folder: el("mount-folder").value,
    sizeValue: el("size-value").value,
    sizeUnit: el("size-unit").value,
    compress: el("compress").checked,
    dedup: el("dedup").checked,
  };
  try {
    localStorage.setItem(CFG_KEY, JSON.stringify(cfg));
  } catch (e) {
    /* ignore */
  }
}

function loadSavedConfig() {
  try {
    return JSON.parse(localStorage.getItem(CFG_KEY) || "null");
  } catch (e) {
    return null;
  }
}

// --- setup screen: devices / drives / size hint -----------------------------

async function loadDevices() {
  const sel = el("device");
  const saved = loadSavedConfig();
  const gpus = await invoke("list_gpus");
  gpuList = gpus;
  sel.innerHTML = "";
  if (!gpus.length) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = t("noCudaOption");
    o.disabled = true;
    o.selected = true;
    sel.appendChild(o);
    el("mount-btn").disabled = true;
    setText("setup-status", t("noGpuError"), "error");
    return;
  }
  for (const g of gpus) {
    const o = document.createElement("option");
    o.value = String(g.ordinal);
    o.dataset.default = String(g.default_size);
    o.dataset.total = String(g.total_vram);
    o.textContent = `[${g.ordinal}] ${g.name} — ${formatSize(g.total_vram)}`;
    sel.appendChild(o);
  }
  if (saved && saved.device != null) {
    const match = Array.from(sel.options).find((o) => o.value === String(saved.device));
    if (match) sel.value = String(saved.device);
  }
  updateSizeHint();
  validateSizeField();
}

// GPU model name for a device ordinal, or null if the list hasn't loaded.
function gpuName(ordinal) {
  const g = gpuList.find((g) => g.ordinal === ordinal);
  return g ? g.name : null;
}

async function loadFreeDrives() {
  const sel = el("drive");
  const saved = loadSavedConfig();
  const drives = await invoke("list_free_drives");
  sel.innerHTML = "";
  for (const d of drives) {
    const o = document.createElement("option");
    o.value = d;
    o.textContent = d;
    sel.appendChild(o);
  }
  if (saved && saved.drive && drives.includes(saved.drive)) {
    sel.value = saved.drive;
  } else if (drives.includes("R:")) {
    sel.value = "R:";
  }
}

function updateSizeHint() {
  const sel = el("device");
  const opt = sel.options[sel.selectedIndex];
  const def = opt && opt.dataset.default ? Number(opt.dataset.default) : null;
  el("size-default-hint").textContent = def ? t("sizeDefaultHint", formatSize(def)) : "";
}

// Re-checks the size field against the selected GPU's total VRAM whenever
// either changes, instead of waiting for the mount attempt to fail.
function validateSizeField() {
  if (!gpuList.length) return false;
  const raw = el("size-value").value.trim();
  let error = "";
  if (raw) {
    const n = Number(raw);
    if (!Number.isFinite(n) || n <= 0) {
      error = t("sizeInvalid");
    } else {
      const bytes = n * UNIT_BYTES[el("size-unit").value];
      const sel = el("device");
      const opt = sel.options[sel.selectedIndex];
      const total = opt && opt.dataset.total ? Number(opt.dataset.total) : null;
      if (total && bytes > total) {
        error = t("sizeTooBig", formatSize(total));
      }
    }
  }
  el("size-error").textContent = error;
  el("size-error").hidden = !error;
  el("mount-btn").disabled = !!error;
  return !error;
}

function restoreSizeAndFlags() {
  const saved = loadSavedConfig();
  if (!saved) return;
  if (saved.sizeValue) el("size-value").value = saved.sizeValue;
  if (saved.sizeUnit && el("size-unit").querySelector(`option[value="${saved.sizeUnit}"]`)) {
    el("size-unit").value = saved.sizeUnit;
  }
  if (saved.folder) el("mount-folder").value = saved.folder;
  el("compress").checked = !!saved.compress;
  el("dedup").checked = !!saved.dedup;
  if (saved.mountMode) setMountMode(saved.mountMode);
}

// --- mount mode (drive letter vs. folder) -----------------------------------

function setMountMode(mode) {
  for (const tab of document.querySelectorAll("#screen-setup .seg-btn")) {
    tab.classList.toggle("active", tab.dataset.mountMode === mode);
  }
  el("mount-drive-field").hidden = mode !== "drive";
  el("mount-folder-field").hidden = mode !== "folder";
}

function mountMode() {
  const active = document.querySelector("#screen-setup .seg-btn.active");
  return active ? active.dataset.mountMode : "drive";
}

async function doBrowseFolder() {
  try {
    const picked = await invoke("browse_folder");
    if (picked) {
      el("mount-folder").value = picked;
      saveConfig();
    }
  } catch (e) {
    setText("setup-status", t("folderPickFail", e), "error");
  }
}

// --- nvCOMP availability (gates GPU compression; no silent CPU fallback here) ---

let nvcompOk = true;

function applyNvcompAvailability(available) {
  nvcompOk = available;

  const compressCb = el("compress");
  compressCb.disabled = !available;
  if (!available) compressCb.checked = false;
  el("compress-hint").hidden = available;

  const archiveBtn = el("open-archive");
  archiveBtn.disabled = !available;
  archiveBtn.title = available ? "" : t("nvcompToolTitle");
}

// --- CLI-flag seed (e.g. a shortcut with `vramdisk.exe --mount R: --compress`) ---

function bytesToSizeField(bytes) {
  if (bytes >= UNIT_BYTES.GB) return { value: +(bytes / UNIT_BYTES.GB).toFixed(2), unit: "GB" };
  return { value: +(bytes / UNIT_BYTES.MB).toFixed(2), unit: "MB" };
}

function applyCliOverrides(ov) {
  if (!ov) return;

  if (ov.mount) {
    const mount = ov.mount.trim();
    if (/^[A-Za-z]:?\\?$/.test(mount)) {
      setMountMode("drive");
      const letter = mount.replace(/[\\:]+$/, "").toUpperCase() + ":";
      if (Array.from(el("drive").options).some((o) => o.value === letter)) {
        el("drive").value = letter;
      }
    } else {
      setMountMode("folder");
      el("mount-folder").value = mount;
    }
  }

  if (ov.size_bytes != null) {
    const { value, unit } = bytesToSizeField(ov.size_bytes);
    el("size-value").value = value;
    el("size-unit").value = unit;
  }

  if (ov.compress != null && nvcompOk) el("compress").checked = ov.compress;
  if (ov.dedup != null) el("dedup").checked = ov.dedup;

  if (ov.device != null) {
    const value = String(ov.device);
    if (Array.from(el("device").options).some((o) => o.value === value)) {
      el("device").value = value;
      updateSizeHint();
    }
  }
}

// --- mount / unmount ---------------------------------------------------------

async function doMount(ev) {
  ev.preventDefault();
  if (!gpuList.length) return;
  const btn = el("mount-btn");
  const mode = mountMode();
  const mountPoint = mode === "drive" ? el("drive").value : el("mount-folder").value.trim();
  const device = Number(el("device").value);

  if (!mountPoint) {
    setText("setup-status", mode === "drive" ? t("chooseDrive") : t("chooseFolder"), "error");
    return;
  }
  if (!validateSizeField()) return;

  let size = null;
  const raw = el("size-value").value.trim();
  if (raw) {
    size = Math.round(Number(raw) * UNIT_BYTES[el("size-unit").value]);
  }

  btn.disabled = true;
  btn.textContent = t("mounting");
  setText("setup-status", "");
  try {
    await invoke("mount", {
      cfg: {
        size,
        mount_point: mountPoint,
        device,
        compress: el("compress").checked,
        dedup: el("dedup").checked,
      },
    });
    saveConfig();
    // The backend hides the window and shows a confirmation; our screen
    // switches via the "mount-changed" event listener.
  } catch (e) {
    setText("setup-status", t("mountFail", e), "error");
  } finally {
    btn.textContent = t("mount");
    validateSizeField();
  }
}

// --- teardown: unmount / quit, with an optional ZIP rescue -------------------
//
// Both unmounting and exiting throw the whole volume away, so both go through
// one three-way prompt: save the disk to a ZIP on the host filesystem and then
// tear down, tear down without saving, or cancel. It is an in-window modal
// rather than a native dialog because native message dialogs only offer two
// buttons; the tray reaches it through the "request-teardown" event after
// showing the window.

let teardownIntent = null; // "unmount" | "quit" while the prompt is up
let exportRunning = false;
let exportStartedAt = 0;

// How many failed entries to name before collapsing the rest into a count.
const EXPORT_FAILURES_SHOWN = 20;

function openTeardown(intent) {
  if (!currentStatus || exportRunning) return;
  teardownIntent = intent;
  el("teardown-actions").hidden = false;
  el("export-job-row").hidden = true;
  el("export-failures").innerHTML = "";
  setText("export-done", "");
  renderJobProgress("export", null, 0);
  renderTeardownTexts();
  el("teardown-backdrop").hidden = false;
  el("teardown-save").focus();
}

// Deliberately refuses to close mid-export: the archive is still being written
// and this modal is the only thing saying so.
function closeTeardown() {
  if (exportRunning) return;
  teardownIntent = null;
  el("teardown-backdrop").hidden = true;
}

// The prompt's wording differs between the two intents (and has to survive a
// language switch while it is open), so it is rendered rather than static.
function renderTeardownTexts() {
  if (!teardownIntent) return;
  const quitting = teardownIntent === "quit";
  el("teardown-title").textContent = t(quitting ? "teardownQuitTitle" : "teardownTitle");
  el("teardown-body").textContent = t(quitting ? "teardownQuitBody" : "teardownBody");
  el("teardown-save").textContent = t(quitting ? "saveAndQuit" : "saveAndUnmount");
  el("teardown-discard").textContent = t(quitting ? "quitWithoutSaving" : "unmountWithoutSaving");
  el("teardown-cancel").textContent = t("cancelBtn");
}

// Tear down for real, once the user has either saved or chosen not to.
async function completeTeardown() {
  const quitting = teardownIntent === "quit";
  el("teardown-actions").hidden = true;
  setText("export-done", t(quitting ? "quitting" : "unmounting"));
  try {
    await invoke(quitting ? "quit_app" : "unmount");
    // Unmounting fires "mount-changed", which closes this modal; quitting is
    // already on its way out of the process.
  } catch (e) {
    setText("export-done", t(quitting ? "quitFail" : "unmountFail", e), "error");
    el("teardown-actions").hidden = false;
  }
}

// "Save to ZIP, then tear down". The teardown only happens if the export
// really succeeded — if anything was left behind, the volume stays mounted so
// the data is still recoverable.
async function saveThenTeardown() {
  let destination;
  try {
    destination = await invoke("browse_export_zip");
  } catch (e) {
    setText("export-done", t("exportFail", e), "error");
    return;
  }
  if (!destination) return; // save dialog dismissed: back to the three choices

  el("teardown-actions").hidden = true;
  el("export-job-row").hidden = false;
  el("export-failures").innerHTML = "";
  setText("export-done", "");
  setText("export-status", t("exporting"));
  renderJobProgress("export", null, 0);
  exportRunning = true;
  exportStartedAt = Date.now();

  let report = null;
  try {
    report = await invoke("export_zip", { destination });
  } catch (e) {
    setText("export-done", t("exportFail", e), "error");
  } finally {
    exportRunning = false;
    el("export-job-row").hidden = true;
  }
  if (!report) {
    el("teardown-actions").hidden = false;
    return;
  }

  if (report.cancelled) {
    setText("export-done", t("exportCancelled"));
    el("teardown-actions").hidden = false;
    return;
  }
  const failures = report.failures || [];
  if (failures.length) {
    renderExportFailures(failures);
    setText("export-done", t("exportPartial", failures.length), "error");
    el("teardown-actions").hidden = false;
    return;
  }
  setText("export-done", t("exportDone", report.destination, report.file_count), "ok");
  await completeTeardown();
}

function renderExportFailures(failures) {
  const rows = failures
    .slice(0, EXPORT_FAILURES_SHOWN)
    .map((f) => kvRow(f.path, f.error));
  if (failures.length > EXPORT_FAILURES_SHOWN) {
    rows.push(kvRow("", t("exportMoreFailures", failures.length - EXPORT_FAILURES_SHOWN)));
  }
  el("export-failures").innerHTML = rows.join("");
}

async function cancelExport() {
  if (!exportRunning) return;
  setText("export-status", t("cancelling"));
  try {
    await invoke("export_cancel");
  } catch (e) {
    /* the export may have already finished */
  }
}

// --- mounted screen rendering ------------------------------------------------

function renderMountedHead(status) {
  const drive = el("mounted-drive");
  drive.textContent = status.mount_point;
  drive.title = status.mount_point;
  drive.classList.toggle("long", status.mount_point.length > 5);
  const name = gpuName(status.device);
  const gpuLabel = name ? name : `GPU ${status.device}`;
  el("mounted-sub").textContent =
    `${gpuLabel} · ${formatSize(status.size)}` +
    (status.compress ? ` · ${t("badgeCompress")}` : "") +
    (status.dedup ? ` · ${t("badgeDedup")}` : "");
}

function renderStats(stats) {
  const used = Number(stats.volume.used_physical_bytes);
  const total = Number(stats.volume.total_bytes);
  const frac = total > 0 ? Math.min(1, used / total) : 0;

  const fill = el("usage-fill");
  fill.style.width = (frac * 100).toFixed(1) + "%";
  fill.classList.toggle("warn", frac >= 0.9);

  el("usage-text-left").textContent = t("usageUsed", formatSize(used), formatSize(total));
  el("usage-text-pct").textContent = (frac * 100).toFixed(1) + "%";
  el("stat-files").textContent = stats.namespace.file_count;
  el("stat-logical").textContent = formatSize(stats.namespace.logical_file_bytes);
  el("stat-dedup").textContent = formatSize(stats.dedup.saved_bytes);
  el("stat-compress").textContent = formatSize(stats.compression.saved_bytes);
}

async function pollStats() {
  if (screenName() !== "mounted" || !currentStatus) return;
  try {
    renderStats(await invoke("stats"));
  } catch (e) {
    /* transient; next tick will retry */
  }
}

// --- mount state changes (from UI or tray) -----------------------------------

function applyMountStatus(status) {
  currentStatus = status || null;
  setText("mounted-status", "");
  setText("setup-status", "");
  if (!currentStatus) {
    // The volume is gone, so the teardown prompt has nothing left to offer —
    // close it even if an export was somehow still marked as running.
    exportRunning = false;
    teardownIntent = null;
    el("teardown-backdrop").hidden = true;
    // Unmounting invalidates every in-flight job: bump generations so their
    // poll loops stop, and reset the per-panel job UI.
    for (const panel of Object.keys(jobs)) {
      const job = jobs[panel];
      if (job.running) {
        job.generation++;
        job.running = false;
        job.jobId = null;
        jobUi(panel, false);
        setText(`${panel}-done`, t("interrupted"));
      }
    }
    showScreen("setup");
    loadFreeDrives().catch(() => {});
    return;
  }
  renderMountedHead(currentStatus);
  const s = screenName();
  if (s !== "hash" && s !== "archive" && s !== "encode") {
    showScreen("mounted");
  }
  pollStats();
}

// --- async GPU jobs ----------------------------------------------------------
//
// One in-flight job per panel. Each start bumps the panel's generation so a
// stale poll loop (cancelled, superseded, or from a closed panel) can never
// clobber newer UI state.

const jobs = {
  hash: { running: false, generation: 0, jobId: null },
  archive: { running: false, generation: 0, jobId: null },
  encode: { running: false, generation: 0, jobId: null },
};

function jobUi(panel, running) {
  el(`${panel}-job-row`).hidden = !running;
  el(`${panel}-btn`).disabled = running;
  // The bar only appears once a poll actually reports byte counts, so both
  // edges of a job (start and finish) reset it to hidden.
  renderJobProgress(panel, null, 0);
}

// `job_status.progress` is optional: the backend omits it for job families
// that can't measure themselves, and reports total_bytes 0 while a job is
// still enumerating. Anything we can't turn into a real fraction is treated as
// "no progress", so the UI falls back to elapsed seconds instead of drawing a
// 0% / NaN% bar.
function jobProgress(status) {
  const p = status && status.progress;
  if (!p) return null;
  const total = Number(p.total_bytes);
  const done = Number(p.done_bytes);
  if (!Number.isFinite(total) || total <= 0) return null;
  if (!Number.isFinite(done) || done < 0) return null;
  return { done: Math.min(done, total), total, frac: Math.min(1, done / total) };
}

// An ETA extrapolated from a barely-started job swings by minutes between
// polls (a slow first chunk reads as "hours left"), so it stays hidden until
// both enough wall time and enough of the work have gone by to mean something.
const ETA_MIN_ELAPSED_MS = 2000;
const ETA_MIN_FRACTION = 0.03;

function etaText(frac, elapsedMs) {
  if (elapsedMs < ETA_MIN_ELAPSED_MS || frac < ETA_MIN_FRACTION || frac >= 1) return "";
  const remainMs = (elapsedMs * (1 - frac)) / frac;
  if (!Number.isFinite(remainMs) || remainMs <= 0) return "";
  const secs = Math.round(remainMs / 1000);
  if (secs < 60) return t("etaSeconds", Math.max(1, secs));
  if (secs < 3600) return t("etaMinutes", Math.max(1, Math.round(secs / 60)));
  // Clamp the minutes so rounding can never produce a nonsense "1 h 60 min".
  return t("etaHours", Math.floor(secs / 3600), Math.min(59, Math.round((secs % 3600) / 60)));
}

// `progress` is a jobProgress() object, or null to hide the bar entirely.
function renderJobProgress(panel, progress, elapsedMs) {
  const box = el(`${panel}-progress`);
  if (!progress) {
    box.hidden = true;
    el(`${panel}-progress-fill`).style.width = "0%";
    return;
  }
  const pct = progress.frac * 100;
  box.hidden = false;
  box.setAttribute("aria-valuenow", pct.toFixed(0));
  el(`${panel}-progress-fill`).style.width = pct.toFixed(1) + "%";
  el(`${panel}-progress-bytes`).textContent = t(
    "progressBytes",
    formatSize(progress.done),
    formatSize(progress.total)
  );
  el(`${panel}-progress-pct`).textContent = t("progressPct", pct.toFixed(1));
  el(`${panel}-progress-eta`).textContent = etaText(progress.frac, elapsedMs);
}

async function runJob(panel, submit, renderResult) {
  const job = jobs[panel];
  const generation = ++job.generation;
  job.running = true;
  job.jobId = null;
  jobUi(panel, true);
  setText(`${panel}-done`, "");
  el(`${panel}-result`).innerHTML = "";
  const startedAt = Date.now();

  const finish = (msg, kind) => {
    if (job.generation !== generation) return;
    job.running = false;
    job.jobId = null;
    jobUi(panel, false);
    setText(`${panel}-done`, msg, kind);
  };

  try {
    const jobId = await submit();
    if (job.generation !== generation) return;
    job.jobId = jobId;

    // Poll until the job turns terminal.
    let pollFailures = 0;
    let lastStatus = null;
    for (;;) {
      await new Promise((r) => setTimeout(r, JOB_POLL_MS));
      if (job.generation !== generation) return;
      let status;
      try {
        status = await invoke("job_status", { jobId });
        pollFailures = 0;
      } catch (e) {
        // Transient read failures are retried, but a wall of them (e.g. the
        // volume disappeared underneath the job) must not poll forever.
        if (++pollFailures > 12) {
          finish(t("jobStatusFail"), "error");
          return;
        }
        continue;
      }
      if (job.generation !== generation) return;
      lastStatus = status;
      if (status.terminal) break;
      const elapsedMs = Date.now() - startedAt;
      setText(`${panel}-status`, t("processing", (elapsedMs / 1000).toFixed(0)));
      renderJobProgress(panel, jobProgress(status), elapsedMs);
    }

    const result = await invoke("job_result", { jobId });
    if (job.generation !== generation) return;
    if (!result.ok) {
      // Branch on the machine-readable JobState ("cancelled" / "failed"), not
      // on the wording of the backend's error message: that text is English,
      // free to change, and never something the UI should parse.
      const state = (result && result.state) || (lastStatus && lastStatus.state) || "";
      if (state === "cancelled") {
        finish(t("cancelled"));
        return;
      }
      finish(t("jobFailed", result.error || t("jobError")), "error");
      return;
    }
    const secs = ((Date.now() - startedAt) / 1000).toFixed(1);
    finish(t("done", secs), "ok");
    el(`${panel}-result`).innerHTML = renderResult(result);
  } catch (e) {
    finish(t("jobFailed", e), "error");
  }
}

async function cancelJob(panel) {
  const job = jobs[panel];
  if (!job.jobId) return;
  setText(`${panel}-status`, t("cancelling"));
  try {
    await invoke("job_cancel", { jobId: job.jobId });
  } catch (e) {
    /* job may have already finished */
  }
}

// --- hash panel ---------------------------------------------------------------

function doHashJob(ev) {
  ev.preventDefault();
  const path = el("hash-path").value.trim() || "\\";
  const algorithm = el("hash-algo").value;
  const recursive = el("hash-recursive").checked;

  runJob(
    "hash",
    () => invoke("hash_job", { paths: [path], algorithm, recursive }),
    (res) => {
      const files = res.files || [];
      return (
        files.map((f) => kvRow(f.path, f.digest)).join("") ||
        kvRow(t("resultKey"), t("noFiles"))
      );
    }
  );
}

// --- archive panel ------------------------------------------------------------

function setArchiveMode(mode) {
  for (const tab of document.querySelectorAll("#screen-archive .seg-btn")) {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  }
  el("archive-compress-fields").hidden = mode !== "compress";
  el("archive-extract-fields").hidden = mode !== "extract";
  el("archive-btn").textContent = mode === "compress" ? t("runCompress") : t("runExtract");
}

function archiveMode() {
  const active = document.querySelector("#screen-archive .seg-btn.active");
  return active ? active.dataset.mode : "compress";
}

// Extract mode doesn't ask for a format; infer it from the archive's extension.
function detectArchiveFormat(path) {
  const p = path.toLowerCase();
  if (p.endsWith(".tar.zst")) return "tar.zst";
  if (p.endsWith(".tar.lz4")) return "tar.lz4";
  if (p.endsWith(".tar.gz") || p.endsWith(".tgz")) return "tar.gz";
  if (p.endsWith(".zip")) return "zip";
  return null;
}

// Join the mount point (drive letter "R:" or folder "C:\vramdisk") with a
// child name into a normal Windows absolute path.
function mountJoin(name) {
  const mount = (currentStatus && currentStatus.mount_point) || "";
  return mount.replace(/\\+$/, "") + "\\" + name;
}

function updateArchivePlaceholders() {
  if (!currentStatus) return;
  el("archive-paths").placeholder = mountJoin("data");
  el("archive-output").placeholder = mountJoin("out.tar.zst");
  el("archive-input").placeholder = mountJoin("out.tar.zst");
  el("archive-outdir").placeholder = mountJoin("restore");
}

function doArchiveJob(ev) {
  ev.preventDefault();
  const mode = archiveMode();

  if (mode === "compress") {
    const format = el("archive-format").value;
    const paths = el("archive-paths").value.trim() || "\\";
    const output = el("archive-output").value.trim();
    if (!output) {
      setText("archive-done", t("outputRequired"), "error");
      return;
    }
    runJob(
      "archive",
      () => invoke("archive_compress_job", { req: { format, paths: [paths], output } }),
      renderArchiveResult
    );
  } else {
    const archive = el("archive-input").value.trim();
    const outputDir = el("archive-outdir").value.trim() || "\\";
    if (!archive) {
      setText("archive-done", t("archiveRequired"), "error");
      return;
    }
    const format = detectArchiveFormat(archive);
    if (!format) {
      setText("archive-done", t("formatUnknown"), "error");
      return;
    }
    runJob(
      "archive",
      () => invoke("archive_extract_job", { req: { format, archive, output_dir: outputDir } }),
      renderArchiveResult
    );
  }
}

function renderArchiveResult(res) {
  const rows = [];
  if (res.output) rows.push(kvRow(t("kvOutput"), res.output));
  if (res.archive) rows.push(kvRow(t("kvArchive"), res.archive));
  if (res.output_dir) rows.push(kvRow(t("kvOutdir"), res.output_dir));
  if (res.file_count != null) rows.push(kvRow(t("kvFileCount"), res.file_count));
  if (res.input_bytes != null) rows.push(kvRow(t("kvInputSize"), formatSize(res.input_bytes)));
  if (res.archive_bytes != null) rows.push(kvRow(t("kvArchiveSize"), formatSize(res.archive_bytes)));
  if (res.output_bytes != null) rows.push(kvRow(t("kvOutputSize"), formatSize(res.output_bytes)));
  if (res.throughput_mib_s != null) {
    rows.push(kvRow(t("kvThroughput"), `${Number(res.throughput_mib_s).toFixed(1)} MB/s`));
  }
  return rows.join("");
}

// --- encode panel -------------------------------------------------------------

function encodeDirection() {
  const active = document.querySelector("#screen-encode .seg-btn.active");
  return active ? active.dataset.direction : "encode";
}

function setEncodeDirection(direction, opts) {
  for (const tab of document.querySelectorAll("#screen-encode .seg-btn")) {
    tab.classList.toggle("active", tab.dataset.direction === direction);
  }
  el("encode-btn").textContent = direction === "encode" ? t("runEncode") : t("runDecode");
  updateEncodePlaceholders();
  if (!(opts && opts.keepOutput)) suggestEncodeOutput();
}

function encodeExt() {
  return el("encode-codec").value === "base64" ? ".b64" : ".hex";
}

function updateEncodePlaceholders() {
  if (!currentStatus) return;
  const enc = encodeDirection() === "encode";
  el("encode-input").placeholder = enc ? mountJoin("data.bin") : mountJoin("data.bin" + encodeExt());
  el("encode-output").placeholder = enc ? mountJoin("data.bin" + encodeExt()) : mountJoin("data.bin");
}

// Suggest an output path from the input path: append the codec extension when
// encoding, strip it when decoding. Never overwrites what the user typed.
let encodeOutputTouched = false;

function suggestEncodeOutput() {
  if (encodeOutputTouched) return;
  const input = el("encode-input").value.trim();
  if (!input) {
    el("encode-output").value = "";
    return;
  }
  const ext = encodeExt();
  if (encodeDirection() === "encode") {
    el("encode-output").value = input + ext;
  } else {
    el("encode-output").value = input.toLowerCase().endsWith(ext)
      ? input.slice(0, -ext.length)
      : input + ".decoded";
  }
}

function doEncodeJob(ev) {
  ev.preventDefault();
  const codec = el("encode-codec").value;
  const direction = encodeDirection();
  const input = el("encode-input").value.trim();
  const output = el("encode-output").value.trim();
  if (!input) {
    setText("encode-done", t("inputRequired"), "error");
    return;
  }
  if (!output) {
    setText("encode-done", t("encodeOutputRequired"), "error");
    return;
  }
  runJob(
    "encode",
    () => invoke("encode_job", { req: { codec, direction, input, output } }),
    (res) => {
      const rows = [];
      rows.push(kvRow(t("kvOutput"), res.output));
      if (res.input_bytes != null) rows.push(kvRow(t("kvInputSize"), formatSize(res.input_bytes)));
      if (res.output_bytes != null) rows.push(kvRow(t("kvOutputSize"), formatSize(res.output_bytes)));
      if (res.throughput_mib_s != null) {
        rows.push(kvRow(t("kvThroughput"), `${Number(res.throughput_mib_s).toFixed(1)} MB/s`));
      }
      return rows.join("");
    }
  );
}

// --- boot --------------------------------------------------------------------

// Each boot step is individually guarded: one failing invoke (e.g. a driver
// hiccup during GPU enumeration) must degrade that feature, not leave the
// whole window dead with no listeners attached.
async function boot() {
  const step = async (fn, whatKey) => {
    try {
      await fn();
    } catch (e) {
      console.error(whatKey, e);
      setText("setup-status", t("stepFail", t(whatKey), e), "error");
    }
  };

  // Language first, so every later step renders its strings in the right one.
  try {
    LANG = await invoke("get_ui_language");
  } catch (e) {
    LANG = String(navigator.language || "").toLowerCase().startsWith("ja") ? "ja" : "en";
  }
  applyLanguage();
  el("lang-select").addEventListener("change", () => switchLanguage(el("lang-select").value));

  await step(loadDevices, "stepGpus");
  await step(loadFreeDrives, "stepDrives");
  restoreSizeAndFlags();
  await step(async () => applyNvcompAvailability(await invoke("nvcomp_available")), "stepNvcomp");
  await step(async () => applyCliOverrides(await invoke("initial_overrides")), "stepCli");

  el("mount-form").addEventListener("submit", doMount);
  el("unmount-btn").addEventListener("click", () => openTeardown("unmount"));
  el("teardown-save").addEventListener("click", saveThenTeardown);
  el("teardown-discard").addEventListener("click", completeTeardown);
  el("teardown-cancel").addEventListener("click", closeTeardown);
  el("export-cancel").addEventListener("click", cancelExport);
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !el("teardown-backdrop").hidden) closeTeardown();
  });
  el("device").addEventListener("change", () => {
    updateSizeHint();
    validateSizeField();
  });
  el("size-value").addEventListener("input", validateSizeField);
  el("size-unit").addEventListener("change", validateSizeField);
  el("browse-folder-btn").addEventListener("click", doBrowseFolder);

  for (const tab of document.querySelectorAll("#screen-setup .seg-btn")) {
    tab.addEventListener("click", () => {
      setMountMode(tab.dataset.mountMode);
      saveConfig();
    });
  }

  for (const idn of [
    "device",
    "drive",
    "mount-folder",
    "size-value",
    "size-unit",
    "compress",
    "dedup",
  ]) {
    el(idn).addEventListener("change", saveConfig);
  }

  el("open-hash").addEventListener("click", () => showScreen("hash"));
  el("open-archive").addEventListener("click", () => {
    if (nvcompOk) {
      updateArchivePlaceholders();
      showScreen("archive");
    }
  });
  el("open-encode").addEventListener("click", () => {
    updateEncodePlaceholders();
    suggestEncodeOutput();
    showScreen("encode");
  });
  for (const back of document.querySelectorAll("[data-back]")) {
    back.addEventListener("click", () => showScreen("mounted"));
  }

  el("hash-form").addEventListener("submit", doHashJob);
  el("hash-cancel").addEventListener("click", () => cancelJob("hash"));

  for (const tab of document.querySelectorAll("#screen-archive .seg-btn")) {
    tab.addEventListener("click", () => setArchiveMode(tab.dataset.mode));
  }
  el("archive-form").addEventListener("submit", doArchiveJob);
  el("archive-cancel").addEventListener("click", () => cancelJob("archive"));
  setArchiveMode("compress");

  for (const tab of document.querySelectorAll("#screen-encode .seg-btn")) {
    tab.addEventListener("click", () => setEncodeDirection(tab.dataset.direction));
  }
  el("encode-form").addEventListener("submit", doEncodeJob);
  el("encode-cancel").addEventListener("click", () => cancelJob("encode"));
  el("encode-codec").addEventListener("change", () => {
    updateEncodePlaceholders();
    suggestEncodeOutput();
  });
  el("encode-input").addEventListener("input", suggestEncodeOutput);
  el("encode-output").addEventListener("input", () => {
    encodeOutputTouched = el("encode-output").value.trim() !== "";
  });

  // Native pick dialogs for path fields; the chosen absolute path is
  // normalized against the mount point on the backend.
  const pickInto = async (cmd, inputId, after) => {
    try {
      const picked = await invoke(cmd);
      if (picked) {
        el(inputId).value = picked;
        if (after) after();
      }
    } catch (e) {
      /* dialog dismissed or unavailable */
    }
  };
  el("hash-path-browse").addEventListener("click", () => pickInto("browse_file", "hash-path"));
  el("archive-paths-browse").addEventListener("click", () => pickInto("browse_folder", "archive-paths"));
  el("archive-output-browse").addEventListener("click", () => pickInto("browse_save", "archive-output"));
  el("archive-input-browse").addEventListener("click", () => pickInto("browse_file", "archive-input"));
  el("archive-outdir-browse").addEventListener("click", () => pickInto("browse_folder", "archive-outdir"));
  el("encode-input-browse").addEventListener("click", () =>
    pickInto("browse_file", "encode-input", suggestEncodeOutput)
  );
  el("encode-output-browse").addEventListener("click", () =>
    pickInto("browse_save", "encode-output", () => {
      encodeOutputTouched = true;
    })
  );

  await step(async () => applyMountStatus(await invoke("mount_status")), "stepMount");

  await step(() => listen("mount-changed", (e) => applyMountStatus(e.payload)), "stepEvents");
  // The tray's unmount / exit entries hand their prompt to this window.
  await step(
    () =>
      listen("request-teardown", (e) => openTeardown(e.payload === "quit" ? "quit" : "unmount")),
    "stepEvents"
  );
  // Closing the window while mounted only hides it, so an unanswered teardown
  // prompt would still be up the next time the tray reopens the window. Treat
  // the hide as a キャンセル and reopen on a clean screen. `closeTeardown`
  // still refuses mid-export, which is right: that one must keep reporting.
  await step(() => listen("window-hidden", closeTeardown), "stepEvents");
  // The export reports `{done_bytes, total_bytes}`, the same shape the job
  // system publishes, so the job progress helpers render it as-is.
  await step(
    () =>
      listen("export-progress", (e) => {
        if (!exportRunning) return;
        renderJobProgress(
          "export",
          jobProgress({ progress: e.payload }),
          Date.now() - exportStartedAt
        );
      }),
    "stepEvents"
  );
  await step(
    () =>
      listen("open-archive-panel", () => {
        if (currentStatus && nvcompOk) {
          updateArchivePlaceholders();
          showScreen("archive");
        }
      }),
    "stepEvents"
  );
  await step(
    () =>
      listen("open-hash-panel", () => {
        if (currentStatus) showScreen("hash");
      }),
    "stepEvents"
  );
  await step(
    () =>
      listen("open-encode-panel", () => {
        if (currentStatus) {
          updateEncodePlaceholders();
          suggestEncodeOutput();
          showScreen("encode");
        }
      }),
    "stepEvents"
  );

  setInterval(pollStats, 1500);
}

window.addEventListener("DOMContentLoaded", () => {
  boot().catch((e) => {
    console.error("boot failed", e);
    setText("setup-status", t("bootFail", e), "error");
  });
});
