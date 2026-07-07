// VRAMDISK GUI frontend.
//
// Uses the global Tauri API (app.withGlobalTauri = true) so no bundler/import
// step is required. Only one disk can ever be mounted at a time, so the UI is
// a simple screen switch: setup (nothing mounted) <-> mounted (+ hash /
// archive tool panels reachable from it).

"use strict";

const invoke = window.__TAURI__.core.invoke;
const listen = window.__TAURI__.event.listen;
const dialog = window.__TAURI__.dialog;

const el = (id) => document.getElementById(id);
const UNMOUNT_WARNING = "本当にアンマウントしますか？\nドライブ上のデータは全て失われます。";

// GB/MB use the Windows convention (binary, 1024-based) so the size field
// agrees with the "既定値: X" hint and the GPU's reported VRAM.
const UNIT_BYTES = { MB: 1024 ** 2, GB: 1024 ** 3 };

let currentStatus = null; // MountStatus (from backend) or null
let gpuList = []; // [{ordinal, name, total_vram, default_size}, ...], cached from list_gpus()

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
  for (const s of ["setup", "mounted", "hash", "archive"]) {
    if (!el("screen-" + s).hidden) return s;
  }
  return null;
}

function showScreen(name) {
  for (const s of ["setup", "mounted", "hash", "archive"]) {
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
  try {
    const gpus = await invoke("list_gpus");
    gpuList = gpus;
    sel.innerHTML = "";
    if (!gpus.length) {
      const o = document.createElement("option");
      o.textContent = "CUDA デバイスが見つかりません";
      o.disabled = true;
      sel.appendChild(o);
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
  } catch (e) {
    alert("GPU 列挙に失敗: " + e);
  }
}

// GPU model name for a device ordinal (e.g. "RTX 4070"), or null if unknown
// (e.g. the cached list hasn't loaded yet).
function gpuName(ordinal) {
  const g = gpuList.find((g) => g.ordinal === ordinal);
  return g ? g.name : null;
}

async function loadFreeDrives() {
  const sel = el("drive");
  const saved = loadSavedConfig();
  try {
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
  } catch (e) {
    alert("ドライブ列挙に失敗: " + e);
  }
}

function updateSizeHint() {
  const sel = el("device");
  const opt = sel.options[sel.selectedIndex];
  const def = opt && opt.dataset.default ? Number(opt.dataset.default) : null;
  el("size-label").textContent = def
    ? `容量（既定値 = ${formatSize(def)}）：`
    : "容量：";
}

// Re-checks the size field against the selected GPU's total VRAM every time
// either changes, instead of waiting for the mount attempt to fail. Grays out
// the mount button while the value is invalid or too large.
function validateSizeField() {
  const raw = el("size-value").value.trim();
  let error = "";
  if (raw) {
    const n = Number(raw);
    if (!Number.isFinite(n) || n <= 0) {
      error = "サイズが不正です";
    } else {
      const bytes = n * UNIT_BYTES[el("size-unit").value];
      const sel = el("device");
      const opt = sel.options[sel.selectedIndex];
      const total = opt && opt.dataset.total ? Number(opt.dataset.total) : null;
      if (total && bytes > total) {
        error = `サイズが GPU の VRAM 容量 (${formatSize(total)}) を超えています`;
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
  // Ignore a stale saved unit (e.g. old "GiB"/"TiB") that no longer exists as
  // an option, so el("size-unit").value stays a valid UNIT_BYTES key.
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
  for (const tab of document.querySelectorAll("#screen-setup .tab")) {
    tab.classList.toggle("active", tab.dataset.mountMode === mode);
  }
  el("mount-drive-field").hidden = mode !== "drive";
  el("mount-folder-field").hidden = mode !== "folder";
}

function mountMode() {
  const active = document.querySelector("#screen-setup .tab.active");
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
    alert("フォルダ選択に失敗: " + e);
  }
}

// --- nvCOMP availability (gates GPU compression, no silent CPU fallback in the GUI) ---

let nvcompOk = true;

function applyNvcompAvailability(available) {
  nvcompOk = available;

  const compressCb = el("compress");
  compressCb.disabled = !available;
  if (!available) compressCb.checked = false;
  el("compress-hint").hidden = available;

  const archiveBtn = el("open-archive");
  archiveBtn.disabled = !available;
  archiveBtn.title = available ? "" : "nvCOMP が見つからないため利用できません";
}

// --- CLI-flag seed (e.g. a shortcut with `vramdisk.exe --mount R: --compress`) ---
// Pre-fills the setup screen's fields; it never mounts automatically. Only
// fields the user actually passed are touched (see `vramdisk::cli::scan_overrides`
// on the Rust side) so this layers on top of, rather than replacing, the
// saved `localStorage` config restored by `restoreSizeAndFlags()`.

function bytesToSizeField(bytes) {
  if (bytes >= UNIT_BYTES.GB) return { value: +(bytes / UNIT_BYTES.GB).toFixed(2), unit: "GB" };
  return { value: +(bytes / UNIT_BYTES.MB).toFixed(2), unit: "MB" };
}

function applyCliOverrides(ov) {
  if (!ov) return;

  if (ov.mount) {
    const mount = ov.mount.trim();
    if (/^[A-Za-z]:\\?$/.test(mount)) {
      setMountMode("drive");
      const letter = mount.replace(/\\$/, "").toUpperCase();
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

  // Respect the nvCOMP gate: never force-check a disabled "compress" box.
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
  const btn = el("mount-btn");
  const mode = mountMode();
  const mountPoint = mode === "drive" ? el("drive").value : el("mount-folder").value.trim();
  const device = Number(el("device").value);

  if (!mountPoint) {
    alert(mode === "drive" ? "ドライブレターを選択してください" : "フォルダを指定してください");
    return;
  }
  if (Number.isNaN(device)) {
    alert("GPU デバイスを選択してください");
    return;
  }
  if (!validateSizeField()) {
    alert(el("size-error").textContent);
    return;
  }

  let size = null;
  const raw = el("size-value").value.trim();
  if (raw) {
    size = Math.round(Number(raw) * UNIT_BYTES[el("size-unit").value]);
  }

  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = "マウント中…";
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
    alert("マウント失敗: " + e);
  } finally {
    btn.textContent = originalLabel;
    validateSizeField();
  }
}

async function doUnmount() {
  const confirmed = dialog
    ? await dialog.confirm(UNMOUNT_WARNING, {
        title: "VRAMDISK",
        kind: "warning",
        okLabel: "続行",
        cancelLabel: "キャンセル",
      })
    : confirm(UNMOUNT_WARNING);
  if (!confirmed) return;

  const btn = el("unmount-btn");
  btn.disabled = true;
  setText("mounted-status", "アンマウント中…");
  try {
    await invoke("unmount");
  } catch (e) {
    setText("mounted-status", "アンマウント失敗: " + e, "error");
  } finally {
    btn.disabled = false;
  }
}

// --- mounted screen rendering ------------------------------------------------

function renderMountedHead(status) {
  const el1 = el("mounted-drive");
  el1.textContent = status.mount_point;
  el1.title = status.mount_point;
  el1.classList.toggle("long", status.mount_point.length > 5);
  const name = gpuName(status.device);
  const gpuLabel = name ? `[GPU ${status.device}] ${name}` : `GPU ${status.device}`;
  el("mounted-sub").textContent =
    `${gpuLabel} · ${formatSize(status.size)}` +
    (status.compress ? " · 圧縮" : "") +
    (status.dedup ? " · dedup" : "");
}

function renderStats(stats) {
  const used = Number(stats.volume.used_physical_bytes);
  const total = Number(stats.volume.total_bytes);
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  el("usage-bar").style.width = pct.toFixed(1) + "%";
  el("usage-text-left").textContent = `物理使用 ${formatSize(used)} / ${formatSize(total)}`;
  el("usage-text-pct").textContent = pct.toFixed(1) + "%";
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
  if (!currentStatus) {
    showScreen("setup");
    return;
  }
  renderMountedHead(currentStatus);
  if (screenName() !== "hash" && screenName() !== "archive") {
    showScreen("mounted");
  }
  pollStats();
}

// --- GPU hash panel -----------------------------------------------------------

async function doHashJob(ev) {
  ev.preventDefault();
  const btn = el("hash-btn");
  const path = el("hash-path").value.trim() || "\\";
  const algorithm = el("hash-algo").value;
  const recursive = el("hash-recursive").checked;

  btn.disabled = true;
  setText("hash-status", "計算中…");
  el("hash-result").innerHTML = "";
  try {
    const res = await invoke("hash_job", { paths: [path], algorithm, recursive });
    if (!res.ok) throw new Error(res.error || "failed");
    const files = res.files || [];
    setText("hash-status", `${files.length} 件`, "ok");
    el("hash-result").innerHTML =
      files.map((f) => kvRow(f.path, f.digest)).join("") ||
      kvRow("結果", "対象ファイルなし");
  } catch (e) {
    setText("hash-status", "失敗: " + e, "error");
  } finally {
    btn.disabled = false;
  }
}

// --- GPU archive panel --------------------------------------------------------

function setArchiveMode(mode) {
  for (const tab of document.querySelectorAll("#screen-archive .tab")) {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  }
  el("archive-compress-fields").hidden = mode !== "compress";
  el("archive-extract-fields").hidden = mode !== "extract";
  el("archive-btn").textContent = mode === "compress" ? "圧縮を実行" : "展開を実行";
}

function archiveMode() {
  return document.querySelector("#screen-archive .tab.active").dataset.mode;
}

// Extract mode doesn't ask for a format; infer it from the archive's extension.
function detectArchiveFormat(path) {
  const p = path.toLowerCase();
  if (p.endsWith(".tar.zst")) return "tar.zst";
  if (p.endsWith(".tar.lz4")) return "tar.lz4";
  if (p.endsWith(".tar.gz")) return "tar.gz";
  if (p.endsWith(".zip")) return "zip";
  return null;
}

// Join the mount point (drive letter "R:" or folder "C:\vramdisk") with a
// child name into a normal Windows absolute path.
function mountJoin(name) {
  const mount = (currentStatus && currentStatus.mount_point) || "";
  return mount.replace(/\\+$/, "") + "\\" + name;
}

// Fill the archive fields' placeholders with absolute paths under the current
// mount point (a "\..."-relative input still works, it's just not advertised).
function updateArchivePlaceholders() {
  if (!currentStatus) return;
  el("archive-paths").placeholder = mountJoin("data");
  el("archive-output").placeholder = mountJoin("out.tar.zst");
  el("archive-input").placeholder = mountJoin("out.tar.zst");
  el("archive-outdir").placeholder = mountJoin("restore");
}

async function doArchiveJob(ev) {
  ev.preventDefault();
  const btn = el("archive-btn");
  const mode = archiveMode();

  btn.disabled = true;
  setText("archive-status", mode === "compress" ? "圧縮中…" : "展開中…");
  el("archive-result").innerHTML = "";
  try {
    let res;
    if (mode === "compress") {
      const format = el("archive-format").value;
      const paths = el("archive-paths").value.trim() || "\\";
      const output = el("archive-output").value.trim();
      if (!output) throw new Error("出力先を入力してください");
      res = await invoke("archive_compress_job", {
        req: { format, paths: [paths], output },
      });
    } else {
      const archive = el("archive-input").value.trim();
      const outputDir = el("archive-outdir").value.trim() || "\\";
      if (!archive) throw new Error("アーカイブパスを入力してください");
      const format = detectArchiveFormat(archive);
      if (!format) {
        throw new Error("拡張子から形式を判定できません（.tar.zst / .tar.lz4 / .tar.gz / .zip）");
      }
      res = await invoke("archive_extract_job", {
        req: { format, archive, output_dir: outputDir },
      });
    }
    if (!res.ok) throw new Error(res.error || "failed");
    setText("archive-status", "完了", "ok");
    el("archive-result").innerHTML = renderArchiveResult(res);
  } catch (e) {
    setText("archive-status", "失敗: " + e, "error");
  } finally {
    btn.disabled = false;
  }
}

function renderArchiveResult(res) {
  const rows = [];
  if (res.output) rows.push(kvRow("出力", res.output));
  if (res.archive) rows.push(kvRow("アーカイブ", res.archive));
  if (res.output_dir) rows.push(kvRow("展開先", res.output_dir));
  if (res.file_count != null) rows.push(kvRow("ファイル数", res.file_count));
  if (res.input_bytes != null) rows.push(kvRow("入力サイズ", formatSize(res.input_bytes)));
  if (res.archive_bytes != null) rows.push(kvRow("アーカイブサイズ", formatSize(res.archive_bytes)));
  if (res.output_bytes != null) rows.push(kvRow("展開後サイズ", formatSize(res.output_bytes)));
  if (res.elapsed_ms != null) rows.push(kvRow("所要時間", `${res.elapsed_ms} ms`));
  if (res.throughput_mib_s != null && res.throughput_mib_s !== null) {
    rows.push(kvRow("スループット", `${Number(res.throughput_mib_s).toFixed(1)} MB/s`));
  }
  return rows.join("");
}

// --- boot --------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", async () => {
  await loadDevices();
  await loadFreeDrives();
  restoreSizeAndFlags();
  applyNvcompAvailability(await invoke("nvcomp_available"));
  applyCliOverrides(await invoke("initial_overrides"));

  el("mount-form").addEventListener("submit", doMount);
  el("unmount-btn").addEventListener("click", doUnmount);
  el("device").addEventListener("change", () => {
    updateSizeHint();
    validateSizeField();
  });
  el("size-value").addEventListener("input", validateSizeField);
  el("size-unit").addEventListener("change", validateSizeField);
  el("browse-folder-btn").addEventListener("click", doBrowseFolder);

  for (const tab of document.querySelectorAll("#screen-setup .tab")) {
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
  for (const back of document.querySelectorAll("[data-back]")) {
    back.addEventListener("click", () => showScreen("mounted"));
  }

  el("hash-form").addEventListener("submit", doHashJob);

  for (const tab of document.querySelectorAll("#screen-archive .tab")) {
    tab.addEventListener("click", () => setArchiveMode(tab.dataset.mode));
  }
  el("archive-form").addEventListener("submit", doArchiveJob);
  setArchiveMode("compress");

  // Native pick dialogs for the archive paths ("参照..." buttons). The chosen
  // absolute path is normalized against the mount point on the backend.
  const pickInto = async (cmd, inputId) => {
    const picked = await invoke(cmd);
    if (picked) el(inputId).value = picked;
  };
  el("archive-paths-browse").addEventListener("click", () => pickInto("browse_folder", "archive-paths"));
  el("archive-output-browse").addEventListener("click", () => pickInto("browse_save", "archive-output"));
  el("archive-input-browse").addEventListener("click", () => pickInto("browse_file", "archive-input"));
  el("archive-outdir-browse").addEventListener("click", () => pickInto("browse_folder", "archive-outdir"));

  applyMountStatus(await invoke("mount_status"));

  await listen("mount-changed", (e) => applyMountStatus(e.payload));
  await listen("open-archive-panel", () => {
    if (currentStatus && nvcompOk) {
      updateArchivePlaceholders();
      showScreen("archive");
    }
  });
  await listen("open-hash-panel", () => {
    if (currentStatus) showScreen("hash");
  });

  setInterval(pollStats, 1500);
});
