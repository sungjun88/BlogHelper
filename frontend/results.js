const summaryStats = document.querySelector("#summaryStats");
const uploadSummary = document.querySelector("#uploadSummary");
const classifierStatus = document.querySelector("#classifierStatus");
const warnings = document.querySelector("#warnings");
const matrixBoard = document.querySelector("#matrixBoard");
const imageCardTemplate = document.querySelector("#imageCardTemplate");
const resultsLoadingOverlay = document.querySelector("#resultsLoadingOverlay");
const resultsLoadingTitle = document.querySelector("#resultsLoadingTitle");
const resultsLoadingCopy = document.querySelector("#resultsLoadingCopy");
const editorStatus = document.querySelector("#editorStatus");
const trainButton = document.querySelector("#trainButton");

const state = {
  categories: [],
  groupedImages: {},
  locationGroups: [],
  selectedPlacesByGroup: {},
  customPlacesByGroup: {},
  assignments: {},
  dragFilename: null,
  dragRowId: null,
  hoverPreview: null,
  hoverPreviewImage: null,
  hoverPreviewName: null,
  hoverPreviewMeta: null,
  hoverPreviewPlace: null,
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

  const place = document.createElement("p");
  place.className = "hover-preview-place";

  info.append(name, meta, place);
  preview.append(image, info);
  document.body.append(preview);

  state.hoverPreview = preview;
  state.hoverPreviewImage = image;
  state.hoverPreviewName = name;
  state.hoverPreviewMeta = meta;
  state.hoverPreviewPlace = place;
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

function getClassifierInfo(image) {
  if (image?.features?.classifier === "trained_clip") {
    return { label: "Trained CLIP", className: "is-trained" };
  }

  if (image?.features?.classifier === "local_clip") {
    return { label: "Base CLIP", className: "is-clip" };
  }

  return { label: "Heuristic", className: "is-heuristic" };
}

function showHoverPreview(event, image) {
  ensureHoverPreview();

  const classifierInfo = getClassifierInfo(image);
  const sourceText = image.media_type === "video" ? "video thumbnail" : "image";
  const labelText = image.is_manual_label ? "edited" : "auto";
  const confidence = `Confidence ${Math.round((image.confidence || 0) * 100)}%`;
  const placeInfo = image.place_info || {};
  const gps = placeInfo.gps;
  const nearest = placeInfo.nearest_place;
  const nearbyPlaces = Array.isArray(placeInfo.nearby_places) ? placeInfo.nearby_places : [];
  const reverse = placeInfo.reverse_geocode;
  const inferredPlaceName = placeInfo.inferred_place_name;
  const inferredAddress = placeInfo.inferred_address;
  const inferredFromNeighbors = placeInfo.inferred_from_neighbors;
  const inferredFromTimeWindow = placeInfo.inferred_from_time_window;
  const inferredSourceFilename = placeInfo.inferred_source_filename;
  const inferredTimeDeltaMinutes = placeInfo.inferred_time_delta_minutes;

  state.hoverPreviewImage.src = image.image_url;
  state.hoverPreviewImage.alt = image.filename || "";
  state.hoverPreviewName.textContent = image.filename || "";
  state.hoverPreviewMeta.textContent = `${confidence} / ${classifierInfo.label} / ${sourceText} / ${labelText}`;

  if (inferredFromTimeWindow) {
    const placeLabel = inferredPlaceName || nearest?.name || "Unknown";
    const deltaLabel = Number.isFinite(inferredTimeDeltaMinutes)
      ? ` / ${Math.round(inferredTimeDeltaMinutes)} min window`
      : "";
    state.hoverPreviewPlace.textContent =
      `Inferred from ${inferredSourceFilename || "nearby capture time"} / ${placeLabel}` +
      `${inferredAddress ? ` / ${inferredAddress}` : ""}${deltaLabel}`;
  } else if (gps) {
    const gpsText = `GPS ${gps.latitude}, ${gps.longitude}`;
    const candidatesText = nearbyPlaces.length
      ? nearbyPlaces
          .map((place, index) => `${index + 1}. ${place.name}${place.distance_meters ? ` (${place.distance_meters}m)` : ""}`)
          .join(" / ")
      : nearest?.name
        ? `1. ${nearest.name}${nearest.distance_meters ? ` (${nearest.distance_meters}m)` : ""}`
        : "";
    const fallbackText = reverse?.display_name
      ? `Address: ${reverse.display_name}`
      : "GPS found, but no nearby place matched.";
    state.hoverPreviewPlace.textContent = candidatesText
      ? `${gpsText} / Candidates: ${candidatesText}`
      : `${gpsText} / ${fallbackText}`;
  } else {
    state.hoverPreviewPlace.textContent = inferredFromNeighbors
      ? `No GPS metadata / Inferred place: ${inferredPlaceName || "Unknown"}${inferredAddress ? ` / ${inferredAddress}` : ""}`
      : "No GPS metadata";
  }

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
  state.hoverPreviewPlace.textContent = "";
}

function showResultsLoading(title = "Loading results", message = "Organizing uploaded media.") {
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
    <div class="summary-chip">Uploaded ${result.uploaded_count || 0}</div>
    <div class="summary-chip">Ignored ${result.ignored_count || 0}</div>
    <div class="summary-chip">Failed ${result.failed_count || 0}</div>
  `;
}

function renderWarnings(result) {
  warnings.innerHTML = "";
  if (!result) {
    return;
  }

  const sections = [
    { title: "Ignored files", items: result.ignored_files || [] },
    { title: "Failed files", items: result.failed_files || [] },
  ];

  sections.forEach((section) => {
    if (!section.items.length) {
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

function renderClassifierStatus() {
  const images = Object.values(state.groupedImages).flat();

  if (!images.length) {
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
    parts.push(`<div class="classifier-chip is-trained">Trained ${counts.trained_clip}</div>`);
  }
  if (counts.local_clip) {
    parts.push(`<div class="classifier-chip is-clip">Base CLIP ${counts.local_clip}</div>`);
  }
  if (counts.heuristic_fallback) {
    parts.push(`<div class="classifier-chip is-heuristic">Heuristic ${counts.heuristic_fallback}</div>`);
  }
  if (Object.keys(counts).length > 1) {
    parts.push('<div class="classifier-chip is-mixed">Multiple classifiers were used.</div>');
  }

  classifierStatus.innerHTML = parts.join("");
}

function createEmptyState(message = "No items") {
  const empty = document.createElement("div");
  empty.className = "category-empty";
  empty.textContent = message;
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

function findImageRowId(filename) {
  for (const group of state.locationGroups) {
    if (group.items.some((item) => item.filename === filename)) {
      return group.group_id;
    }
  }
  return "no-gps";
}

function syncLocationGroupCategory(filename, targetCategory) {
  state.locationGroups.forEach((group) => {
    group.items.forEach((item) => {
      if (item.filename === filename) {
        item.category = targetCategory;
        item.manual_label = targetCategory;
        item.is_manual_label = true;
      }
    });
  });
}

function moveImageToCategory(filename, targetCategory, targetRowId) {
  if (!filename || !targetCategory || !state.groupedImages[targetCategory]) {
    return;
  }

  const sourceRowId = findImageRowId(filename);
  if (sourceRowId !== targetRowId) {
    setEditorStatus("GPS row is fixed. Drag thumbnails only across columns inside the same row.", "error");
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
  syncLocationGroupCategory(filename, targetCategory);
  updateAssignmentsFromGroups();
  renderClassifierStatus();
  renderMatrixBoard();
  setEditorStatus(`Moved "${movedImage.filename}" to ${targetCategory}. Click Train to save this edit.`, "info");
}

function buildImageCard(image, rowId) {
  const imageFragment = imageCardTemplate.content.cloneNode(true);
  const cardElement = imageFragment.querySelector(".image-card");
  const imageElement = imageFragment.querySelector(".image-preview");
  const classifierElement = imageFragment.querySelector(".image-classifier");
  const originElement = imageFragment.querySelector(".image-origin");
  const confidenceElement = imageFragment.querySelector(".image-confidence");
  const nameElement = imageFragment.querySelector(".image-name");
  const classifierInfo = getClassifierInfo(image);

  cardElement.dataset.filename = image.filename;
  cardElement.dataset.rowId = rowId;
  imageElement.src = image.image_url;
  imageElement.alt = image.filename;
  nameElement.textContent = image.filename;
  confidenceElement.textContent = `Confidence ${Math.round((image.confidence || 0) * 100)}%`;
  classifierElement.textContent = classifierInfo.label;
  classifierElement.className = `image-classifier ${classifierInfo.className}`;

  const sourceText = image.media_type === "video" ? "Video thumbnail" : "Image";
  const labelText = image.is_manual_label ? "edited" : "auto";
  originElement.textContent = `${sourceText} / ${labelText}`;

  imageElement.addEventListener("mouseenter", (event) => {
    showHoverPreview(event, image);
  });

  imageElement.addEventListener("mousemove", (event) => {
    updateHoverPreviewPosition(event);
  });

  imageElement.addEventListener("mouseleave", hideHoverPreview);

  cardElement.addEventListener("dragstart", (event) => {
    hideHoverPreview();
    state.dragFilename = image.filename;
    state.dragRowId = rowId;
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", image.filename);
    cardElement.classList.add("is-dragging");
  });

  cardElement.addEventListener("dragend", () => {
    state.dragFilename = null;
    state.dragRowId = null;
    cardElement.classList.remove("is-dragging");
    document.querySelectorAll(".matrix-cell").forEach((item) => {
      item.classList.remove("is-drop-target");
      item.classList.remove("is-drop-blocked");
    });
  });

  return imageFragment;
}

function createMatrixCell(categoryKey, rowId, images) {
  const cell = document.createElement("div");
  cell.className = "matrix-cell";
  cell.dataset.category = categoryKey;
  cell.dataset.rowId = rowId;

  const items = document.createElement("div");
  items.className = "matrix-cell-items";

  cell.addEventListener("dragover", (event) => {
    event.preventDefault();
    const isAllowed = !state.dragRowId || state.dragRowId === rowId;
    cell.classList.toggle("is-drop-target", isAllowed);
    cell.classList.toggle("is-drop-blocked", !isAllowed);
  });

  cell.addEventListener("dragleave", () => {
    cell.classList.remove("is-drop-target");
    cell.classList.remove("is-drop-blocked");
  });

  cell.addEventListener("drop", (event) => {
    event.preventDefault();
    cell.classList.remove("is-drop-target");
    cell.classList.remove("is-drop-blocked");
    moveImageToCategory(state.dragFilename, categoryKey, rowId);
    state.dragFilename = null;
    state.dragRowId = null;
  });

  if (!images.length) {
    items.append(createEmptyState());
  } else {
    images.forEach((image) => {
      items.append(buildImageCard(image, rowId));
    });
  }

  cell.append(items);
  return cell;
}

function buildNoGpsRow() {
  const groupedNames = new Set();
  state.locationGroups.forEach((group) => {
    group.items.forEach((item) => groupedNames.add(item.filename));
  });

  const items = Object.values(state.groupedImages)
    .flat()
    .filter((item) => !groupedNames.has(item.filename));

  return {
    group_id: "no-gps",
    count: items.length,
    place_name: "No GPS",
    address: "Items without readable GPS metadata",
    center: null,
    items,
  };
}

function groupItemsByCategory(items) {
  const grouped = {};
  state.categories.forEach((category) => {
    grouped[category.key] = [];
  });

  items.forEach((item) => {
    if (!grouped[item.category]) {
      grouped[item.category] = [];
    }
    grouped[item.category].push(item);
  });

  return grouped;
}

function createHeaderRow() {
  const headerRow = document.createElement("div");
  headerRow.className = "matrix-row matrix-row-header";
  headerRow.style.gridTemplateColumns = `240px repeat(${state.categories.length}, minmax(140px, 1fr))`;

  const corner = document.createElement("div");
  corner.className = "matrix-corner";
  corner.innerHTML = `
    <p class="matrix-corner-label">Place</p>
    <p class="matrix-corner-copy">Rows are GPS groups within 10 meters.</p>
  `;
  headerRow.append(corner);

  state.categories.forEach((category) => {
    const cell = document.createElement("div");
    cell.className = "matrix-header-cell";

    const key = document.createElement("p");
    key.className = "matrix-header-key";
    key.textContent = category.key;

    const title = document.createElement("h3");
    title.className = "matrix-header-title";
    title.textContent = category.label;

    cell.append(key, title);
    headerRow.append(cell);
  });

  return headerRow;
}

function getGroupPlaceCandidates(group) {
  const candidateMap = new Map();
  const items = Array.isArray(group.items) ? group.items : [];

  if (group.place_name || group.address) {
    candidateMap.set(`group::${group.place_name || ""}::${group.address || ""}`, {
      id: `group::${group.group_id}`,
      name: group.place_name || "Unknown place",
      address: group.address || "No matched place name",
      distance_meters: 0,
      kind: "group",
    });
  }

  items.forEach((item) => {
    const placeInfo = item.place_info || {};
    const nearbyPlaces = Array.isArray(placeInfo.nearby_places) ? placeInfo.nearby_places : [];
    const reverse = placeInfo.reverse_geocode || {};
    const inferredAddress = placeInfo.inferred_address;
    const candidateAddress = reverse.display_name || inferredAddress || group.address || "";

    nearbyPlaces.forEach((place) => {
      if (!place?.name) {
        return;
      }

      const key = `${place.name}::${place.brand || ""}::${candidateAddress}`;
      const distance = Number.isFinite(place.distance_meters) ? place.distance_meters : Number.POSITIVE_INFINITY;
      const current = candidateMap.get(key);

      if (!current || distance < current.distance_meters) {
        candidateMap.set(key, {
          id: key,
          name: place.name,
          address: candidateAddress || "No matched place name",
          distance_meters: distance,
          kind: place.kind || "",
        });
      }
    });
  });

  return Array.from(candidateMap.values())
    .sort((left, right) => left.distance_meters - right.distance_meters)
    .slice(0, 5);
}

function getSelectedPlaceCandidate(group, index) {
  const candidates = getGroupPlaceCandidates(group);
  const selectedId = state.selectedPlacesByGroup[group.group_id];
  const selectedCandidate = candidates.find((candidate) => candidate.id === selectedId);
  const customPlaceName = (state.customPlacesByGroup[group.group_id] || "").trim();

  return {
    selectedCandidate: selectedCandidate || candidates[0] || null,
    candidates,
    customPlaceName,
    fallbackName: group.place_name || `Place ${index + 1}`,
    fallbackAddress: group.address || "No matched place name",
  };
}

function createPlaceLabel(group, index) {
  const label = document.createElement("aside");
  label.className = "matrix-row-label";

  const { selectedCandidate, candidates, customPlaceName, fallbackName, fallbackAddress } = getSelectedPlaceCandidate(group, index);

  const name = document.createElement("h3");
  name.className = "matrix-place-title";
  name.textContent = customPlaceName || selectedCandidate?.name || fallbackName;

  const address = document.createElement("p");
  address.className = "matrix-place-address";
  address.textContent = selectedCandidate?.address || fallbackAddress;

  const candidatePanel = document.createElement("div");
  candidatePanel.className = "matrix-place-candidate-panel";

  const candidatePanelLabel = document.createElement("p");
  candidatePanelLabel.className = "matrix-place-candidate-label";
  candidatePanelLabel.textContent = "Place candidates";
  candidatePanel.append(candidatePanelLabel);

  const candidateList = document.createElement("div");
  candidateList.className = "matrix-place-candidate-list";

  if (candidates.length) {
    candidates.forEach((candidate, candidateIndex) => {
      const candidateButton = document.createElement("button");
      candidateButton.type = "button";
      candidateButton.className = "matrix-place-candidate";
      if (selectedCandidate?.id === candidate.id) {
        candidateButton.classList.add("is-selected");
      }

      candidateButton.innerHTML = `
        <span class="matrix-place-candidate-rank">${candidateIndex + 1}</span>
        <span class="matrix-place-candidate-text">
          <span class="matrix-place-candidate-name">${candidate.name}</span>
          <span class="matrix-place-candidate-meta">${Number.isFinite(candidate.distance_meters) ? `${Math.round(candidate.distance_meters)}m` : "distance unknown"}</span>
        </span>
      `;
      candidateButton.addEventListener("click", () => {
        state.selectedPlacesByGroup[group.group_id] = candidate.id;
        state.customPlacesByGroup[group.group_id] = "";
        renderMatrixBoard();
      });
      candidateList.append(candidateButton);
    });
  } else {
    const empty = document.createElement("p");
    empty.className = "matrix-place-candidates";
    empty.textContent = "No place candidates";
    candidateList.append(empty);
  }

  const customPlaceInput = document.createElement("input");
  customPlaceInput.type = "text";
  customPlaceInput.className = "matrix-place-custom-input";
  customPlaceInput.placeholder = "찾으시는 업체가 없다면 직접 적어주세요";
  customPlaceInput.value = state.customPlacesByGroup[group.group_id] || "";
  customPlaceInput.addEventListener("input", (event) => {
    state.customPlacesByGroup[group.group_id] = event.target.value;
    name.textContent = event.target.value.trim() || selectedCandidate?.name || fallbackName;
  });
  customPlaceInput.addEventListener("blur", () => {
    renderMatrixBoard();
  });
  customPlaceInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      customPlaceInput.blur();
    }
  });

  candidatePanel.append(candidateList);
  candidatePanel.append(customPlaceInput);
  label.append(name, address, candidatePanel);
  return label;
}

function renderMatrixBoard() {
  matrixBoard.innerHTML = "";

  if (!state.categories.length) {
    matrixBoard.append(createEmptyState("No categories"));
    return;
  }

  const board = document.createElement("div");
  board.className = "matrix-grid";
  board.append(createHeaderRow());

  const rows = [...state.locationGroups];
  const noGpsRow = buildNoGpsRow();
  if (noGpsRow.count > 0) {
    rows.push(noGpsRow);
  }

  if (!rows.length) {
    matrixBoard.append(createEmptyState("No uploaded items"));
    return;
  }

  rows.forEach((group, index) => {
    const row = document.createElement("div");
    row.className = "matrix-row";
    row.dataset.rowId = group.group_id;
    row.style.gridTemplateColumns = `240px repeat(${state.categories.length}, minmax(140px, 1fr))`;
    row.append(createPlaceLabel(group, index));

    const grouped = groupItemsByCategory(group.items || []);
    state.categories.forEach((category) => {
      row.append(createMatrixCell(category.key, group.group_id, grouped[category.key] || []));
    });

    board.append(row);
  });

  matrixBoard.append(board);
}

function updateSummaryStats(totalCount) {
  const gpsCount = state.locationGroups.length;
  summaryStats.textContent = `${totalCount} items / ${state.categories.length} categories / ${gpsCount} GPS groups`;
}

function waitForRenderedImages() {
  const images = Array.from(document.querySelectorAll(".image-preview"));
  if (!images.length) {
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
    body: JSON.stringify({ assignments: state.assignments }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to save labels.");
  }

  state.assignments = payload.assignments || {};
}

async function trainCurrentAssignments() {
  showResultsLoading("Training", "Saving current labels and retraining the classifier.");
  trainButton.disabled = true;

  try {
    await saveAssignments();

    const response = await fetch("/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ assignments: state.assignments }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Training failed.");
    }

    setEditorStatus(`Training completed. ${payload.trained_count} items were used.`, "success");
    await loadResults();
  } catch (error) {
    setEditorStatus(error.message || "Training failed.", "error");
  } finally {
    trainButton.disabled = false;
    hideResultsLoading();
  }
}

async function loadResults() {
  const response = await fetch("/images/categorized");
  if (!response.ok) {
    throw new Error("Failed to load categorized results.");
  }

  const result = await response.json();
  state.categories = result.categories || [];
  state.groupedImages = result.grouped_images || {};
  state.locationGroups = result.location_groups || [];
  state.selectedPlacesByGroup = {};
  state.customPlacesByGroup = {};
  state.assignments = result.assignments || {};

  updateAssignmentsFromGroups();
  updateSummaryStats(result.total_count || 0);
  renderClassifierStatus();
  renderMatrixBoard();
  await waitForRenderedImages();
}

async function init() {
  ensureHoverPreview();
  showResultsLoading();

  const lastUploadResult = readLastUploadResult();
  renderUploadSummary(lastUploadResult);
  renderWarnings(lastUploadResult);
  setEditorStatus("Drag thumbnails across columns inside the same GPS row, then click Train.", "info");

  try {
    await loadResults();
    if (lastUploadResult) {
      sessionStorage.removeItem("lastUploadResult");
    }
  } catch (error) {
    uploadSummary.innerHTML = '<div class="summary-chip summary-chip-error">Failed to load results.</div>';
    setEditorStatus(error.message || "Failed to initialize results.", "error");
  } finally {
    hideResultsLoading();
  }
}

trainButton.addEventListener("click", trainCurrentAssignments);
window.addEventListener("scroll", hideHoverPreview, { passive: true });

init();
