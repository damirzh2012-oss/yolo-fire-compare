async function readError(res) {
  try {
    const data = await res.json();
    return data.detail || JSON.stringify(data);
  } catch {
    return res.statusText || "Ошибка запроса";
  }
}

const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const btnBrowse = document.getElementById("btn-browse");
const btnRun = document.getElementById("btn-run");
const statusEl = document.getElementById("status");
const overlay = document.getElementById("overlay");
const fileNameEl = document.getElementById("file-name");
const previewWrap = document.getElementById("preview-wrap");
const previewImg = document.getElementById("preview-img");
const previewVideo = document.getElementById("preview-video");

let uploaded = null;
let chart = null;

btnBrowse.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

btnRun.addEventListener("click", runCompare);

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function showPreview(url, mediaType) {
  previewWrap.classList.remove("hidden");
  const isVideo = mediaType === "video";
  previewImg.classList.toggle("hidden", isVideo);
  previewVideo.classList.toggle("hidden", !isVideo);
  if (isVideo) {
    previewVideo.src = url;
  } else {
    previewImg.src = url;
  }
}

function showResult(prefix, url, mediaType) {
  const img = document.getElementById(`${prefix}-img`);
  const video = document.getElementById(`${prefix}-video`);
  const wrap = document.getElementById(`${prefix}-media`);
  wrap.classList.remove("placeholder");
  const placeholder = wrap.querySelector("p");
  if (placeholder) placeholder.classList.add("hidden");
  const isVideo = mediaType === "video";
  img.classList.toggle("hidden", isVideo);
  video.classList.toggle("hidden", !isVideo);
  const bust = `${url}?t=${Date.now()}`;
  if (isVideo) {
    video.src = bust;
  } else {
    img.src = bust;
  }
}

async function handleFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["jpg", "jpeg", "png", "mp4", "avi"].includes(ext)) {
    setStatus("Поддерживаются jpg, png, mp4, avi", true);
    return;
  }
  setStatus("Загрузка…");
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(await readError(res));
    uploaded = await res.json();
    fileNameEl.textContent = uploaded.original_name;
    showPreview(uploaded.url, uploaded.media_type);
    btnRun.disabled = false;
    setStatus("Файл загружен. Можно запускать сравнение.");
  } catch (err) {
    setStatus(err.message || "Ошибка загрузки", true);
  }
}

function pct(v) {
  return `${(v * 100).toFixed(1)}%`;
}

function dash(v) {
  return v === null || v === undefined ? "нет GT" : String(v);
}

function renderStats(data) {
  document.getElementById("stat-det-base").textContent = data.base.detections_count;
  document.getElementById("stat-det-ft").textContent = data.finetuned.detections_count;
  document.getElementById("stat-conf-base").textContent = pct(data.base.avg_confidence);
  document.getElementById("stat-conf-ft").textContent = pct(data.finetuned.avg_confidence);
  document.getElementById("stat-fp-base").textContent = dash(data.base.false_positives);
  document.getElementById("stat-fp-ft").textContent = dash(data.finetuned.false_positives);
  document.getElementById("stat-t-base").textContent = `${data.base.inference_time_sec} с`;
  document.getElementById("stat-t-ft").textContent = `${data.finetuned.inference_time_sec} с`;

  const note = data.has_ground_truth
    ? "по precision относительно ground truth"
    : "прокси по среднему confidence (нет ground truth)";
  document.getElementById("verdict").textContent =
    `Прирост точности: ${data.accuracy_gain_pct}% (${note})`;

  const ctx = document.getElementById("chart");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Детекции", "Avg conf. %"],
      datasets: [
        {
          label: "Базовая",
          data: [data.base.detections_count, data.base.avg_confidence * 100],
          backgroundColor: "#58a6ff",
        },
        {
          label: "Доработанная",
          data: [data.finetuned.detections_count, data.finetuned.avg_confidence * 100],
          backgroundColor: "#f97316",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#e6edf3" } } },
      scales: {
        x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
        y: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" }, beginAtZero: true },
      },
    },
  });
}

async function runCompare() {
  if (!uploaded) return;
  overlay.classList.remove("hidden");
  setStatus("Обработка…");
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: uploaded.filename }),
    });
    if (!res.ok) throw new Error(await readError(res));
    const data = await res.json();
    showResult("base", data.base.output_url, data.media_type);
    showResult("ft", data.finetuned.output_url, data.media_type);
    renderStats(data);
    setStatus("Готово.");
  } catch (err) {
    setStatus(err.message || "Ошибка инференса", true);
  } finally {
    overlay.classList.add("hidden");
  }
}
