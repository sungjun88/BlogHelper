const summaryStats = document.querySelector("#summaryStats");
const uploadSummary = document.querySelector("#uploadSummary");
const warnings = document.querySelector("#warnings");
const categoryGrid = document.querySelector("#categoryGrid");
const categoryTemplate = document.querySelector("#categoryTemplate");
const imageCardTemplate = document.querySelector("#imageCardTemplate");
const resultsLoadingOverlay = document.querySelector("#resultsLoadingOverlay");

const state = {
  categories: [],
  groupedImages: {},
};

function showResultsLoading() {
  resultsLoadingOverlay.classList.remove("is-hidden");
}

function hideResultsLoading() {
  resultsLoadingOverlay.classList.add("is-hidden");
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

function createEmptyState() {
  const empty = document.createElement("div");
  empty.className = "category-empty";
  empty.textContent = "분류된 사진이 없습니다.";
  return empty;
}

function renderCategories() {
  categoryGrid.innerHTML = "";

  state.categories
    .filter((category) => category.enabled)
    .forEach((category) => {
      const fragment = categoryTemplate.content.cloneNode(true);
      const card = fragment.querySelector(".category-card");
      const key = fragment.querySelector(".category-key");
      const title = fragment.querySelector(".category-title");
      const count = fragment.querySelector(".category-count");
      const itemsContainer = fragment.querySelector(".category-items");
      const images = state.groupedImages[category.key] || [];

      key.textContent = category.key;
      title.textContent = category.label;
      count.textContent = `${images.length}`;

      if (images.length === 0) {
        itemsContainer.append(createEmptyState());
      } else {
        images.forEach((image) => {
          const imageFragment = imageCardTemplate.content.cloneNode(true);
          const imageElement = imageFragment.querySelector(".image-preview");
          const nameElement = imageFragment.querySelector(".image-name");
          const confidenceElement = imageFragment.querySelector(".image-confidence");

          imageElement.src = image.image_url;
          imageElement.alt = image.filename;
          nameElement.textContent = image.filename;
          confidenceElement.textContent = `신뢰도 ${Math.round((image.confidence || 0) * 100)}%`;

          itemsContainer.append(imageFragment);
        });
      }

      categoryGrid.append(card);
    });
}

function updateSummaryStats(totalCount) {
  const enabledCount = state.categories.filter((category) => category.enabled).length;
  summaryStats.textContent = `${totalCount}개 이미지 / ${enabledCount}개 활성 카테고리`;
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

async function loadResults() {
  const response = await fetch("/images/categorized");
  if (!response.ok) {
    throw new Error("분류 결과를 불러오지 못했습니다.");
  }

  const result = await response.json();
  state.categories = result.categories || [];
  state.groupedImages = result.grouped_images || {};
  updateSummaryStats(result.total_count || 0);
  renderCategories();
  await waitForRenderedImages();
}

async function init() {
  showResultsLoading();

  const lastUploadResult = readLastUploadResult();
  renderUploadSummary(lastUploadResult);
  renderWarnings(lastUploadResult);

  try {
    await loadResults();
    if (lastUploadResult) {
      sessionStorage.removeItem("lastUploadResult");
    }
  } catch (error) {
    uploadSummary.innerHTML =
      '<div class="summary-chip summary-chip-error">결과를 불러오지 못했습니다.</div>';
  } finally {
    hideResultsLoading();
  }
}

init();
