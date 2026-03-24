const state = {
  files: [],
  uploadJobId: null,
  statusPollTimer: null,
  uploadTransferPercent: 0,
};

const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#fileInput");
const pickButton = document.querySelector("#pickButton");
const uploadButton = document.querySelector("#uploadButton");
const selectedList = document.querySelector("#selectedList");
const selectionInfo = document.querySelector("#selectionInfo");
const feedback = document.querySelector("#feedback");
const loadingOverlay = document.querySelector("#loadingOverlay");
const loadingTitle = document.querySelector("#loadingTitle");
const loadingMessage = document.querySelector("#loadingMessage");
const progressBarFill = document.querySelector("#progressBarFill");
const progressPercent = document.querySelector("#progressPercent");
const progressCount = document.querySelector("#progressCount");
const elapsedTime = document.querySelector("#elapsedTime");
const estimatedTime = document.querySelector("#estimatedTime");
const remainingTime = document.querySelector("#remainingTime");
const currentTask = document.querySelector("#currentTask");

function setFeedback(message, isError = false) {
  feedback.textContent = message;
  feedback.classList.toggle("is-error", isError);
}

function openFilePicker() {
  fileInput.click();
}

function showLoadingOverlay() {
  loadingOverlay.classList.remove("is-hidden");
}

function hideLoadingOverlay() {
  loadingOverlay.classList.add("is-hidden");
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0s";
  }

  const rounded = Math.round(seconds);
  const minutes = Math.floor(rounded / 60);
  const remainSeconds = rounded % 60;

  if (minutes === 0) {
    return `${remainSeconds}s`;
  }

  return `${minutes}m ${remainSeconds}s`;
}

function setLoadingState({
  title = "Uploading and classifying photos",
  message = "Preparing the job.",
  percent = 0,
  processed = 0,
  total = 0,
  elapsed = 0,
  estimated = 0,
  remaining = 0,
  task = "Waiting",
}) {
  const normalizedPercent = Math.max(0, Math.min(percent, 100));

  loadingTitle.textContent = title;
  loadingMessage.textContent = message;
  progressBarFill.style.width = `${normalizedPercent}%`;
  progressPercent.textContent = `${Math.round(normalizedPercent)}%`;
  progressCount.textContent = `${processed} / ${total} files`;
  elapsedTime.textContent = `Elapsed ${formatDuration(elapsed)}`;
  estimatedTime.textContent = `Estimated total ${formatDuration(estimated)}`;
  remainingTime.textContent = `Remaining ${formatDuration(remaining)}`;
  currentTask.textContent = task;
}

function updateSelectionView() {
  selectedList.innerHTML = "";

  if (state.files.length === 0) {
    selectionInfo.textContent = "No files selected yet.";
    uploadButton.disabled = true;
    return;
  }

  selectionInfo.textContent = `${state.files.length} files selected`;
  uploadButton.disabled = false;

  state.files.forEach((file) => {
    const item = document.createElement("li");
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    const kind = file.type.startsWith("video/") ? "video" : "image";
    item.textContent = `${file.name} (${sizeKb} KB, ${kind})`;
    selectedList.append(item);
  });
}

function isAcceptedFile(file) {
  const lowerName = file.name.toLowerCase();
  return (
    file.type.startsWith("image/") ||
    file.type.startsWith("video/") ||
    [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"].some((ext) =>
      lowerName.endsWith(ext),
    )
  );
}

function setFiles(files) {
  state.files = files.filter((file) => isAcceptedFile(file));
  const ignoredCount = files.length - state.files.length;

  updateSelectionView();

  if (ignoredCount > 0) {
    setFeedback(`Only image and video files are allowed. ${ignoredCount} file(s) were ignored.`, true);
    return;
  }

  if (state.files.length > 0) {
    setFeedback("Ready to upload.");
  } else {
    setFeedback("");
  }
}

async function initUploadJob() {
  const response = await fetch("/upload/init", {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error("Failed to initialize the upload job.");
  }

  const status = await response.json();
  state.uploadJobId = status.job_id;
  return status;
}

async function fetchUploadStatus() {
  if (!state.uploadJobId) {
    return null;
  }

  const response = await fetch(`/upload/status/${state.uploadJobId}`);
  if (!response.ok) {
    return null;
  }

  return response.json();
}

function applyUploadStatus(status) {
  if (!status) {
    return;
  }

  const totalFiles = status.total_files || state.files.length || 0;
  const serverTransferPercent = Math.round((status.transfer_progress || 0) * 100);
  const transferPercent = Math.max(serverTransferPercent, Math.round(state.uploadTransferPercent));
  const fileProgress = totalFiles > 0 ? `${status.processed_files || 0} / ${totalFiles} files` : "Counting files";
  const currentFile = status.current_file ? `Current file: ${status.current_file}` : "Current file: preparing";
  const serverPercent = status.progress_percent || 0;
  const overallPercent = Math.max(serverPercent, transferPercent * 0.2);

  setLoadingState({
    title: "Uploading and classifying photos",
    message: status.message || "Processing files.",
    percent: overallPercent,
    processed: status.processed_files || 0,
    total: totalFiles,
    elapsed: status.elapsed_seconds || 0,
    estimated: status.estimated_total_seconds || 0,
    remaining: status.remaining_seconds || 0,
    task: `Transfer ${transferPercent}% | ${fileProgress} | ${currentFile}`,
  });
}

function startStatusPolling() {
  stopStatusPolling();
  state.statusPollTimer = window.setInterval(async () => {
    const status = await fetchUploadStatus();
    applyUploadStatus(status);
  }, 500);
}

function stopStatusPolling() {
  if (state.statusPollTimer) {
    window.clearInterval(state.statusPollTimer);
    state.statusPollTimer = null;
  }
}

function uploadWithProgress(formData) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const url = `/upload?job_id=${encodeURIComponent(state.uploadJobId)}`;

    request.open("POST", url);
    request.responseType = "json";

    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) {
        return;
      }

      const percent = (event.loaded / event.total) * 100;
      state.uploadTransferPercent = percent;
      setLoadingState({
        title: "Uploading photos to the server",
        message: "Classification starts after the transfer finishes.",
        percent: percent * 0.2,
        processed: 0,
        total: state.files.length,
        elapsed: 0,
        estimated: 0,
        remaining: 0,
        task: `Upload transfer ${Math.round(percent)}%`,
      });
    });

    request.addEventListener("load", () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(request.response);
        return;
      }

      reject(new Error("Upload request failed."));
    });

    request.addEventListener("error", () => {
      reject(new Error("A network error occurred during upload."));
    });

    request.send(formData);
  });
}

async function uploadFiles() {
  if (state.files.length === 0) {
    setFeedback("Select images first.", true);
    return;
  }

  const formData = new FormData();
  state.files.forEach((file) => formData.append("files", file));

  uploadButton.disabled = true;
  showLoadingOverlay();
  setFeedback("Uploading and classifying files.");
  setLoadingState({
    title: "Preparing the upload job",
    message: "Creating the job state.",
    percent: 0,
    processed: 0,
    total: state.files.length,
    elapsed: 0,
    estimated: 0,
    remaining: 0,
    task: "Waiting",
  });

  try {
    state.uploadTransferPercent = 0;
    const initialStatus = await initUploadJob();
    applyUploadStatus(initialStatus);
    startStatusPolling();

    const result = await uploadWithProgress(formData);
    const finalStatus = await fetchUploadStatus();
    applyUploadStatus(finalStatus);
    stopStatusPolling();

    sessionStorage.setItem("lastUploadResult", JSON.stringify(result));
    window.location.href = "/results";
  } catch (error) {
    stopStatusPolling();
    hideLoadingOverlay();
    uploadButton.disabled = false;
    setFeedback(error.message || "Upload failed.", true);
  }
}

pickButton.addEventListener("click", (event) => {
  event.stopPropagation();
  openFilePicker();
});

fileInput.addEventListener("change", (event) => {
  setFiles(Array.from(event.target.files || []));
});

uploadButton.addEventListener("click", uploadFiles);

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("is-dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("is-dragover");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("is-dragover");
  setFiles(Array.from(event.dataTransfer?.files || []));
});

dropzone.addEventListener("click", () => {
  openFilePicker();
});

dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openFilePicker();
  }
});

updateSelectionView();
hideLoadingOverlay();
