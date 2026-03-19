const state = {
  files: [],
};

const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#fileInput");
const pickButton = document.querySelector("#pickButton");
const uploadButton = document.querySelector("#uploadButton");
const selectedList = document.querySelector("#selectedList");
const selectionInfo = document.querySelector("#selectionInfo");
const feedback = document.querySelector("#feedback");
const loadingOverlay = document.querySelector("#loadingOverlay");

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

function updateSelectionView() {
  selectedList.innerHTML = "";

  if (state.files.length === 0) {
    selectionInfo.textContent = "아직 선택된 파일이 없습니다.";
    uploadButton.disabled = true;
    return;
  }

  selectionInfo.textContent = `${state.files.length}개 파일 선택됨`;
  uploadButton.disabled = false;

  state.files.forEach((file) => {
    const item = document.createElement("li");
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    item.textContent = `${file.name} (${sizeKb} KB)`;
    selectedList.append(item);
  });
}

function setFiles(files) {
  state.files = files.filter((file) => file.type.startsWith("image/"));
  const ignoredCount = files.length - state.files.length;

  updateSelectionView();

  if (ignoredCount > 0) {
    setFeedback(`이미지 파일만 선택할 수 있습니다. ${ignoredCount}개 파일은 제외되었습니다.`, true);
    return;
  }

  if (state.files.length > 0) {
    setFeedback("업로드할 준비가 되었습니다.");
  } else {
    setFeedback("");
  }
}

async function uploadFiles() {
  if (state.files.length === 0) {
    setFeedback("먼저 업로드할 이미지를 선택하세요.", true);
    return;
  }

  const formData = new FormData();
  state.files.forEach((file) => formData.append("files", file));

  uploadButton.disabled = true;
  showLoadingOverlay();
  setFeedback("업로드 및 분류 중입니다...");

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("업로드 요청이 실패했습니다.");
    }

    const result = await response.json();
    sessionStorage.setItem("lastUploadResult", JSON.stringify(result));
    window.location.href = "/results";
  } catch (error) {
    hideLoadingOverlay();
    uploadButton.disabled = false;
    setFeedback(error.message || "업로드 중 오류가 발생했습니다.", true);
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
