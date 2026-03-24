const summaryStats = document.querySelector("#summaryStats");
const uploadSummary = document.querySelector("#uploadSummary");
const classifierStatus = document.querySelector("#classifierStatus");
const warnings = document.querySelector("#warnings");
const categoryGrid = document.querySelector("#categoryGrid");
const categoryTemplate = document.querySelector("#categoryTemplate");
const imageCardTemplate = document.querySelector("#imageCardTemplate");
const resultsLoadingOverlay = document.querySelector("#resultsLoadingOverlay");
const resultsLoadingTitle = document.querySelector("#resultsLoadingTitle");
const resultsLoadingCopy = document.querySelector("#resultsLoadingCopy");
const editorStatus = document.querySelector("#editorStatus");
const trainButton = document.querySelector("#trainButton");

const state = {
  categories: [],
  groupedImages: {},
  assignments: {},
  dragFilename: null,
  hoverPreview: null,
  hoverPreviewImage: null,
  hoverPreviewName: null,
  hoverPreviewMeta: null,
};

function ensureHoverPreview() {
  if (state.hoverPreview) {
    return;
  }

  const preview = document.createElement("div");
  preview.className = "hover-preview is-hidden";

  const image = document.createElement("img");
  image.className = "hover-preview-image";
  image.alt = "";

  const info = document.createElement("div");
  info.className = "hover-preview-info";

  const name = document.createElement("p");
  name.className = "hover-preview-name";

  const meta = document.createElement("p");
  meta.className = "hover-preview-meta";

  info.append(name, meta);
  preview.append(image, info);
  document.body.append(preview);

  state.hoverPreview = preview;
  state.hoverPreviewImage = image;
  state.hoverPreviewName = name;
  state.hoverPreviewMeta = meta;
}

function updateHoverPreviewPosition(event) {
  if (!state.hoverPreview || state.hoverPreview.classList.contains("is-hidden")) {
    return;
  }

  const margin = 24;
  const previewWidth = state.hoverPreview.offsetWidth || 400;
  const previewHeight = state.hoverPreview.offsetHeight || 400;
  let left = event.clientX + 24;
  let top = event.clientY + 24;

  if (left + previewWidth + margin > window.innerWidth) {
    left = Math.max(margin, event.clientX - previewWidth - 24);
  }
  if (top + previewHeight + margin > window.innerHeight) {
    top = Math.max(margin, window.innerHeight - previewHeight - margin);
  }

  state.hoverPreview.style.left = `${left}px`;
  state.hoverPreview.style.top = `${top}px`;
}

function showHoverPreview(event, image) {
  ensureHoverPreview();

  const classifierInfo = getClassifierInfo(image);
  const sourceText = image.media_type === "video" ? "동영상 썸네일 기준" : "이미지 기준";
  const labelText = image.is_manual_label ? "수정됨" : "자동 분류";
  const confidence = `신뢰도 ${Math.round((image.confidence || 0) * 100)}%`;

  state.hoverPreviewImage.src = image.image_url;
  state.hoverPreviewImage.alt = image.filename || "";
  state.hoverPreviewName.textContent = image.filename || "";
  state.hoverPreviewMeta.textContent = `${confidence} / ${classifierInfo.label} / ${sourceText} / ${labelText}`;
  state.hoverPreview.classList.remove("is-hidden");

  updateHoverPreviewPosition(event);
}

function hideHoverPreview() {
  if (!state.hoverPreview) {
    return;
  }

  state.hoverPreview.classList.add("is-hidden");
  state.hoverPreviewImage.removeAttribute("src");
  state.hoverPreviewName.textContent = "";
  state.hoverPreviewMeta.textContent = "";
}

function showResultsLoading(
  title = "분류 결과를 불러오고 있습니다",
  message = "현재 업로드된 사진을 정리하는 중입니다.",
) {
  resultsLoadingTitle.textContent = title;
  resultsLoadingCopy.textContent = message;
  resultsLoadingOverlay.classList.remove("is-hidden");
}

function hideResultsLoading() {
  resultsLoadingOverlay.classList.add("is-hidden");
}

function setEditorStatus(message, tone = "") {
  editorStatus.textContent = message;
  editorStatus.className = "editor-status";
  if (tone) {
    editorStatus.classList.add(`is-${tone}`);
  }
}

function readLastUploadResult() {
  const raw = sessionStorage.getItem("lastUploadResult");
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function renderUploadSummary(result) {
  if (!result) {
    uploadSummary.innerHTML = "";
    return;
  }

  uploadSummary.innerHTML = `
    <div class="summary-chip">분류 완료 ${result.uploaded_count || 0}개</div>
    <div class="summary-chip">제외 ${result.ignored_count || 0}개</div>
    <div class="summary-chip">실패 ${result.failed_count || 0}개</div>
  `;
}

function renderWarnings(result) {
  warnings.innerHTML = "";
  if (!result) {
    return;
  }

  const sections = [
    {
      title: "업로드에서 제외된 파일",
      items: result.ignored_files || [],
    },
    {
      title: "분석에 실패한 파일",
      items: result.failed_files || [],
    },
  ];

  sections.forEach((section) => {
    if (section.items.length === 0) {
      return;
    }

    const wrapper = document.createElement("section");
    wrapper.className = "warning-card";

    const title = document.createElement("p");
    title.className = "warning-title";
    title.textContent = section.title;
    wrapper.append(title);

    const list = document.createElement("ul");
    list.className = "warning-list";

    section.items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = `${item.filename}: ${item.reason}`;
      list.append(li);
    });

    wrapper.append(list);
    warnings.append(wrapper);
  });
}

function getClassifierInfo(image) {
  if (image?.features?.classifier === "trained_clip") {
    return {
      label: "학습된 CLIP 분류기",
      className: "is-trained",
    };
  }

  if (image?.features?.classifier === "local_clip") {
    return {
      label: "CLIP 기본 분류",
      className: "is-clip",
    };
  }

  return {
    label: "휴리스틱 분류",
    className: "is-heuristic",
  };
}

function renderClassifierStatus() {
  const images = Object.values(state.groupedImages).flat();

  if (images.length === 0) {
    classifierStatus.innerHTML = "";
    return;
  }

  const counts = images.reduce((acc, image) => {
    const key = image?.features?.classifier || "unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  const parts = [];
  if (counts.trained_clip) {
    parts.push(`<div class="classifier-chip is-trained">학습된 모델 ${counts.trained_clip}장</div>`);
  }
  if (counts.local_clip) {
    parts.push(`<div class="classifier-chip is-clip">기본 CLIP ${counts.local_clip}장</div>`);
  }
  if (counts.heuristic_fallback) {
    parts.push(`<div class="classifier-chip is-heuristic">휴리스틱 ${counts.heuristic_fallback}장</div>`);
  }
  if (Object.keys(counts).length > 1) {
    parts.push('<div class="classifier-chip is-mixed">현재 배치에는 여러 분류기가 함께 사용되었습니다.</div>');
  }

  classifierStatus.innerHTML = parts.join("");
}

function createEmptyState() {
  const empty = document.createElement("div");
  empty.className = "category-empty";
  empty.textContent = "분류된 항목이 없습니다.";
  return empty;
}

function updateAssignmentsFromGroups() {
  const assignments = {};
  Object.entries(state.groupedImages).forEach(([categoryKey, images]) => {
    images.forEach((image) => {
      assignments[image.filename] = categoryKey;
    });
  });
  state.assignments = assignments;
}

function moveImageToCategory(filename, targetCategory) {
  if (!filename || !targetCategory || !state.groupedImages[targetCategory]) {
    return;
  }

  let movedImage = null;
  let sourceCategory = null;

  Object.entries(state.groupedImages).forEach(([categoryKey, images]) => {
    const index = images.findIndex((image) => image.filename === filename);
    if (index >= 0) {
      movedImage = images.splice(index, 1)[0];
      sourceCategory = categoryKey;
    }
  });

  if (!movedImage) {
    return;
  }

  if (sourceCategory === targetCategory) {
    state.groupedImages[sourceCategory].push(movedImage);
    return;
  }

  movedImage.category = targetCategory;
  movedImage.manual_label = targetCategory;
  movedImage.is_manual_label = true;
  state.groupedImages[targetCategory].unshift(movedImage);
  updateAssignmentsFromGroups();
  renderClassifierStatus();
  renderCategories();
  setEditorStatus(`"${movedImage.filename}" 항목을 ${targetCategory} 카테고리로 옮겼습니다. 학습 버튼을 누르면 이 수정 결과가 반영됩니다.`, "info");
}

function buildCategoryCard(category) {
  const fragment = categoryTemplate.content.cloneNode(true);
  const card = fragment.querySelector(".category-card");
  const keyElement = fragment.querySelector(".category-key");
  const titleElement = fragment.querySelector(".category-title");
  const countElement = fragment.querySelector(".category-count");
  const itemsContainer = fragment.querySelector(".category-items");
  const images = state.groupedImages[category.key] || [];

  card.dataset.category = category.key;
  keyElement.textContent = category.key;
  titleElement.textContent = category.label;
  countElement.textContent = `${images.length}`;

  card.addEventListener("dragover", (event) => {
    event.preventDefault();
    card.classList.add("is-drop-target");
  });

  card.addEventListener("dragleave", () => {
    card.classList.remove("is-drop-target");
  });

  card.addEventListener("drop", (event) => {
    event.preventDefault();
    card.classList.remove("is-drop-target");
    moveImageToCategory(state.dragFilename, category.key);
    state.dragFilename = null;
  });

  if (images.length === 0) {
    itemsContainer.append(createEmptyState());
    return card;
  }

  images.forEach((image) => {
    const imageFragment = imageCardTemplate.content.cloneNode(true);
    const cardElement = imageFragment.querySelector(".image-card");
    const imageElement = imageFragment.querySelector(".image-preview");
    const classifierElement = imageFragment.querySelector(".image-classifier");
    const originElement = imageFragment.querySelector(".image-origin");
    const confidenceElement = imageFragment.querySelector(".image-confidence");
    const nameElement = imageFragment.querySelector(".image-name");
    const classifierInfo = getClassifierInfo(image);

    cardElement.dataset.filename = image.filename;
    imageElement.src = image.image_url;
    imageElement.alt = image.filename;
    nameElement.textContent = image.filename;
    confidenceElement.textContent = `신뢰도 ${Math.round((image.confidence || 0) * 100)}%`;
    classifierElement.textContent = classifierInfo.label;
    classifierElement.className = `image-classifier ${classifierInfo.className}`;

    const sourceText = image.media_type === "video" ? "동영상 썸네일 기준 분류" : "이미지 기준 분류";
    const labelText = image.is_manual_label ? "수정됨" : "자동 분류";
    originElement.textContent = `${sourceText} / ${labelText}`;

    imageElement.addEventListener("mouseenter", (event) => {
      showHoverPreview(event, image);
    });

    imageElement.addEventListener("mousemove", (event) => {
      updateHoverPreviewPosition(event);
    });

    imageElement.addEventListener("mouseleave", () => {
      hideHoverPreview();
    });

    cardElement.addEventListener("dragstart", (event) => {
      hideHoverPreview();
      state.dragFilename = image.filename;
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", image.filename);
      cardElement.classList.add("is-dragging");
    });

    cardElement.addEventListener("dragend", () => {
      state.dragFilename = null;
      cardElement.classList.remove("is-dragging");
      document.querySelectorAll(".category-card").forEach((item) => {
        item.classList.remove("is-drop-target");
      });
    });

    itemsContainer.append(imageFragment);
  });

  return card;
}

function renderCategories() {
  categoryGrid.innerHTML = "";

  state.categories.forEach((category) => {
    categoryGrid.append(buildCategoryCard(category));
  });
}

function updateSummaryStats(totalCount) {
  summaryStats.textContent = `${totalCount}개 항목 / ${state.categories.length}개 카테고리`;
}

function waitForRenderedImages() {
  const images = Array.from(document.querySelectorAll(".image-preview"));
  if (images.length === 0) {
    return Promise.resolve();
  }

  return Promise.all(
    images.map((image) => {
      if (image.complete) {
        return Promise.resolve();
      }

      return new Promise((resolve) => {
        image.addEventListener("load", resolve, { once: true });
        image.addEventListener("error", resolve, { once: true });
      });
    }),
  );
}

async function saveAssignments() {
  const response = await fetch("/labels", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      assignments: state.assignments,
    }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "라벨 저장에 실패했습니다.");
  }

  state.assignments = payload.assignments || {};
}

async function trainCurrentAssignments() {
  showResultsLoading("학습 중입니다", "현재 화면의 분류 결과를 저장하고 모델에 반영하는 중입니다.");
  trainButton.disabled = true;

  try {
    await saveAssignments();

    const response = await fetch("/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        assignments: state.assignments,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "학습에 실패했습니다.");
    }

    setEditorStatus(`학습이 완료되었습니다. ${payload.trained_count}개 항목이 반영되었습니다.`, "success");
    await loadResults();
  } catch (error) {
    setEditorStatus(error.message || "학습 중 오류가 발생했습니다.", "error");
  } finally {
    trainButton.disabled = false;
    hideResultsLoading();
  }
}

async function loadResults() {
  const response = await fetch("/images/categorized");
  if (!response.ok) {
    throw new Error("분류 결과를 불러오지 못했습니다.");
  }

  const result = await response.json();
  state.categories = result.categories || [];
  state.groupedImages = result.grouped_images || {};
  state.assignments = result.assignments || {};
  updateAssignmentsFromGroups();
  updateSummaryStats(result.total_count || 0);
  renderClassifierStatus();
  renderCategories();
  await waitForRenderedImages();
}

async function init() {
  ensureHoverPreview();
  showResultsLoading();

  const lastUploadResult = readLastUploadResult();
  renderUploadSummary(lastUploadResult);
  renderWarnings(lastUploadResult);
  setEditorStatus("카드를 다른 카테고리로 옮긴 뒤 학습 버튼을 누르면 수정한 결과가 저장되고 다음 분류에 반영됩니다.", "info");

  try {
    await loadResults();
    if (lastUploadResult) {
      sessionStorage.removeItem("lastUploadResult");
    }
  } catch (error) {
    uploadSummary.innerHTML =
      '<div class="summary-chip summary-chip-error">결과를 불러오지 못했습니다.</div>';
    setEditorStatus(error.message || "결과 화면을 초기화하지 못했습니다.", "error");
  } finally {
    hideResultsLoading();
  }
}

trainButton.addEventListener("click", trainCurrentAssignments);
window.addEventListener("scroll", hideHoverPreview, { passive: true });

init();
