"use strict";
const $ = (id) => document.getElementById(id);

let pollTimer = null;
let startedAt = 0;

/* --------------------------------------------------------- image slots
   Same upload/preview/drag-drop behavior for both --first-frame and
   --last-frame; `prefix` selects the "first-*"/"last-*" element ids and
   the i18n key namespace (both currently share firstframe.choose/clear/
   hint/unsupported/uploadFailed - only the legend and drop-label text
   differ between the two). */
function makeImageSlot(prefix, i18nPrefix) {
  const slot = { imageId: null };
  const dropzone = $(`${prefix}-dropzone`);
  const label = $(`${prefix}-dropzone-label`);
  const input = $(`${prefix}-image-input`);
  const clearBtn = $(`${prefix}-clear-image`);
  const previewBox = $(`${prefix}-preview-box`);
  const previewImg = $(`${prefix}-preview-image`);
  const errorEl = $(`${prefix}-image-error`);

  // Fire-and-forget: the asset store doesn't need to block the UI on this,
  // and a failed delete just means the next server restart's own cleanup
  // catches it instead (see _clear_upload_dir() in server.py).
  function deleteAsset(imageId) {
    if (!imageId) return;
    fetch(`/api/assets/${imageId}`, { method: "DELETE" }).catch(() => {});
  }

  async function upload(file) {
    errorEl.classList.add("hidden");
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      errorEl.textContent = t("firstframe.unsupported");
      errorEl.classList.remove("hidden");
      return;
    }
    const res = await fetch("/api/upload-image", {
      method: "POST",
      headers: { "Content-Type": file.type },
      body: file,
    });
    const data = await res.json();
    if (!res.ok) {
      errorEl.textContent = data.error || t("firstframe.uploadFailed");
      errorEl.classList.remove("hidden");
      return;
    }
    // Replacing an already-uploaded image in this slot - drop the old file
    // rather than leaving it behind for the rest of the server's life.
    deleteAsset(slot.imageId);
    slot.imageId = data.image_id;
    label.textContent = file.name;
    clearBtn.classList.remove("hidden");
    await refreshPreview();
  }

  async function refreshPreview() {
    if (!slot.imageId) return;
    const profile = document.querySelector('input[name="profile"]:checked').value;
    previewImg.src = `/api/assets/${slot.imageId}/preview?profile=${profile}&t=${Date.now()}`;
    previewBox.classList.remove("hidden");
  }
  slot.refreshPreview = refreshPreview;

  $(`${prefix}-choose-image`).addEventListener("click", () => input.click());
  input.addEventListener("change", (event) => {
    if (event.target.files.length) upload(event.target.files[0]);
  });
  clearBtn.addEventListener("click", () => {
    deleteAsset(slot.imageId);
    slot.imageId = null;
    input.value = "";
    label.textContent = t(`${i18nPrefix}.drop`);
    clearBtn.classList.add("hidden");
    previewBox.classList.add("hidden");
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
    if (event.dataTransfer.files.length) upload(event.dataTransfer.files[0]);
  });
  return slot;
}

const firstFrameSlot = makeImageSlot("first", "firstframe");
const lastFrameSlot = makeImageSlot("last", "lastframe");
const refImageSlot = makeImageSlot("ref", "refframe");
document.querySelectorAll('input[name="profile"]').forEach((radio) =>
  radio.addEventListener("change", () => {
    firstFrameSlot.refreshPreview();
    lastFrameSlot.refreshPreview();
    refImageSlot.refreshPreview();
  }));

function showSection(name) {
  ["form", "progress", "result", "error-box"].forEach((id) =>
    $(id).classList.toggle("hidden", id !== name));
}

async function loadConfig() {
  const res = await fetch("/api/config");
  const config = await res.json();
  const banner = $("config-banner");
  const problems = [];
  if (!config.model_dir_ok) problems.push(t("banner.modelDir", { path: config.model_dir }));
  if (!config.attention_cache_ok) {
    problems.push(t(config.attention_cache_buildable ?
      "banner.cacheBuildable" : "banner.cache"));
  }
  if (problems.length) {
    banner.textContent = problems.join(" / ");
    banner.classList.remove("hidden");
  }
  $("layers").max = config.layers_max;
  $("layers").min = config.layers_min;
  $("seconds").max = config.seconds_max;
  $("seconds").min = config.seconds_min;

  if (!config.ref2va_available) {
    $("ref-choose-image").disabled = true;
    $("ref-dropzone-label").textContent = t("refframe.unavailable");
  }

  if (!config.turbo_cache_ok) {
    $("turbo").disabled = true;
    $("turbo-fieldset").title = t("turbo.unavailable");
  }
}

$("layers").addEventListener("input", () => {
  $("layers-value").textContent = $("layers").value;
});

/* The Turbo LoRA is distilled for exactly 4 steps, so Steps is forced and
   locked while it's on rather than left editable and silently overridden -
   the actual override still happens server-side in build_job() regardless
   of what reaches it, this is just so the field doesn't lie to the user. */
$("turbo").addEventListener("change", () => {
  const on = $("turbo").checked;
  $("steps").disabled = on;
  if (on) {
    $("steps").dataset.previousValue = $("steps").value;
    $("steps").value = "4";
  } else if ($("steps").dataset.previousValue) {
    $("steps").value = $("steps").dataset.previousValue;
  }
});

$("generate").addEventListener("click", async () => {
  $("form-error").classList.add("hidden");
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    $("form-error").textContent = t("alert.promptRequired");
    $("form-error").classList.remove("hidden");
    return;
  }
  const seconds = Number($("seconds").value);
  const secondsMin = Number($("seconds").min);
  const secondsMax = Number($("seconds").max);
  if (!Number.isInteger(seconds) || seconds < secondsMin || seconds > secondsMax) {
    $("form-error").textContent = t("alert.secondsRange", { min: secondsMin, max: secondsMax });
    $("form-error").classList.remove("hidden");
    return;
  }
  const body = {
    prompt,
    profile: document.querySelector('input[name="profile"]:checked').value,
    seconds,
    layers: Number($("layers").value),
    reuse: Number($("reuse").value),
    steps: Number($("steps").value),
    seed: $("seed").value.trim() || null,
    image_id: firstFrameSlot.imageId,
    last_image_id: lastFrameSlot.imageId,
    ref_image_id: refImageSlot.imageId,
    turbo: $("turbo").checked,
  };
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    $("form-error").textContent = data.error || t("alert.rejected");
    $("form-error").classList.remove("hidden");
    return;
  }
  // Fill in the seed actually used (h3 picks one itself when the field was
  // left blank) so it's ready to reuse or tweak for the next generation.
  $("seed").value = data.seed;
  startPolling(data.id);
});

function startPolling(jobId) {
  showSection("progress");
  startedAt = Date.now();
  $("phase").textContent = t("progress.queued");
  $("bar").value = 0;
  $("log").textContent = "";
  pollTimer = setInterval(() => pollJob(jobId), 1000);
  pollJob(jobId);
}

async function pollJob(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) return;
  const job = await res.json();
  const elapsedS = Math.floor((Date.now() - startedAt) / 1000);
  const time = `${Math.floor(elapsedS / 60)}:${String(elapsedS % 60).padStart(2, "0")}`;
  $("elapsed").textContent = t("progress.elapsed", { time });
  if (job.phase) {
    $("phase").textContent = job.total ? `${job.phase} (${job.completed}/${job.total})` : job.phase;
    $("bar").max = job.total || 1;
    $("bar").value = job.completed || 0;
  }
  $("log").textContent = job.log_tail.join("\n");
  $("log").scrollTop = $("log").scrollHeight;

  if (job.state === "done") {
    clearInterval(pollTimer);
    showSection("result");
    const src = `/api/jobs/${jobId}/video`;
    $("player").src = src;
    $("download").href = src;
    $("result-seed").textContent = t("result.seed", { seed: job.seed });
  } else if (job.state === "error") {
    clearInterval(pollTimer);
    showSection("error-box");
    $("error-text").textContent = job.error || t("error.default");
  }
}

$("cancel-back").addEventListener("click", () => {
  if (pollTimer) clearInterval(pollTimer);
  showSection("form");
});
$("again").addEventListener("click", () => showSection("form"));
$("error-back").addEventListener("click", () => showSection("form"));

applyStaticTranslations();
loadConfig();
