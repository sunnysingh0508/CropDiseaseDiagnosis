const cropOptions = document.querySelectorAll(".crop-option");
const cropError = document.querySelector("#crop-error");
const dropZone = document.querySelector("#drop-zone");
const imageInput = document.querySelector("#image-input");
const dropContent = document.querySelector("#drop-content");
const imagePreview = document.querySelector("#image-preview");
const previewImage = document.querySelector("#preview-image");
const previewName = document.querySelector("#preview-name");
const previewSize = document.querySelector("#preview-size");
const imageError = document.querySelector("#image-error");
const uploadFeedback = document.querySelector("#upload-feedback");
const removeImageButton = document.querySelector("#remove-image");
const changeImageButton = document.querySelector("#change-image");
const analyzeButton = document.querySelector("#analyze-button");
const analyzeButtonLabel = document.querySelector("#analyze-button-label");
const analyzeSpinner = document.querySelector("#analyze-spinner");
const diagnosisError = document.querySelector("#diagnosis-error");
const diagnosisLoading = document.querySelector("#diagnosis-loading");
const diagnosisMessage = document.querySelector("#diagnosis-message");
const diagnosisTitle = document.querySelector("#diagnosis-title");
const diagnosisStatus = document.querySelector("#diagnosis-status");
const diagnosisConfidence = document.querySelector("#diagnosis-confidence");
const topPredictionsList = document.querySelector("#top-predictions-list");
const menuToggle = document.querySelector(".menu-toggle");
const navMenu = document.querySelector(".nav-menu");

const diagnosisEndpoint = "http://127.0.0.1:8000/api/diagnose";
const allowedExtensions = new Set(["jpg", "jpeg", "png"]);
const allowedMimeTypes = new Set(["image/jpeg", "image/png"]);
const maxFileSize = 10 * 1024 * 1024;
let selectedCrop = "";
let selectedImage = null;
let previewUrl = "";
let activeRequestController = null;

function openFilePicker() {
  imageInput.click();
}

function clearDiagnosis() {
  if (activeRequestController) {
    activeRequestController.abort();
    activeRequestController = null;
  }

  diagnosisLoading.hidden = true;
  diagnosisMessage.hidden = true;
  diagnosisMessage.removeAttribute("data-status");
  diagnosisError.hidden = true;
  diagnosisError.textContent = "";
  analyzeButton.disabled = false;
  analyzeButtonLabel.textContent = "Analyze Crop Disease";
  analyzeSpinner.hidden = true;
  topPredictionsList.replaceChildren();
}

function clearImage() {
  clearDiagnosis();
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = "";
  }

  selectedImage = null;
  imageInput.value = "";
  previewImage.removeAttribute("src");
  previewName.textContent = "";
  previewSize.textContent = "";
  imagePreview.hidden = true;
  dropContent.hidden = false;
  imageError.textContent = "";
  uploadFeedback.textContent = "";
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function validateFile(file) {
  const extension = file.name.split(".").pop().toLowerCase();
  const hasAllowedExtension = allowedExtensions.has(extension);
  const hasAllowedMimeType = !file.type || allowedMimeTypes.has(file.type);

  if (!hasAllowedExtension || !hasAllowedMimeType) {
    return "Choose an image in JPG, JPEG, or PNG format.";
  }

  if (file.size > maxFileSize) {
    return "This image is larger than 10 MB. Choose a smaller file.";
  }

  if (file.size === 0) {
    return "This image file is empty. Choose another image.";
  }

  return "";
}

function showImage(file) {
  const validationMessage = validateFile(file);
  clearImage();

  if (validationMessage) {
    imageError.textContent = validationMessage;
    return;
  }

  selectedImage = file;
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewName.textContent = file.name;
  previewSize.textContent = formatFileSize(file.size);
  imagePreview.hidden = false;
  dropContent.hidden = true;
  uploadFeedback.textContent = "Image ready to review.";
}

function isValidConfidence(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;
}

function validateDiagnosisResponse(data) {
  return (
    data !== null &&
    typeof data === "object" &&
    typeof data.predicted_class === "string" &&
    data.predicted_class.length > 0 &&
    isValidConfidence(data.confidence_percentage) &&
    (data.status === "healthy" || data.status === "diseased") &&
    Array.isArray(data.top_3_predictions) &&
    data.top_3_predictions.length === 3 &&
    data.top_3_predictions.every(
      (prediction) =>
        prediction !== null &&
        typeof prediction === "object" &&
        typeof prediction.class_name === "string" &&
        prediction.class_name.length > 0 &&
        isValidConfidence(prediction.confidence_percentage),
    )
  );
}

function formatClassName(className) {
  return className.replace(/_+/g, " ");
}

function renderDiagnosis(data) {
  diagnosisTitle.textContent = formatClassName(data.predicted_class);
  diagnosisStatus.textContent = `Status: ${data.status}`;
  diagnosisConfidence.textContent = `Confidence: ${data.confidence_percentage.toFixed(2)}%`;
  diagnosisMessage.dataset.status = data.status;
  topPredictionsList.replaceChildren(
    ...data.top_3_predictions.map((prediction) => {
      const item = document.createElement("li");
      item.textContent =
        `${formatClassName(prediction.class_name)} — ` +
        `${prediction.confidence_percentage.toFixed(2)}%`;
      return item;
    }),
  );
  diagnosisMessage.hidden = false;
}

async function readErrorMessage(response) {
  let responseBody;
  try {
    responseBody = await response.json();
  } catch {
    return `Diagnosis failed (HTTP ${response.status}). Please try again.`;
  }

  if (typeof responseBody.detail === "string") {
    return responseBody.detail;
  }

  return `Diagnosis failed (HTTP ${response.status}). Please try again.`;
}

async function analyzeImage() {
  cropError.textContent = selectedCrop ? "" : "Please select a crop to continue.";
  imageError.textContent = selectedImage ? "" : "Please upload a crop leaf image to continue.";
  clearDiagnosis();

  if (!selectedCrop || !selectedImage) {
    return;
  }

  const controller = new AbortController();
  activeRequestController = controller;
  const formData = new FormData();
  formData.append("image", selectedImage);

  analyzeButton.disabled = true;
  analyzeButtonLabel.textContent = "Analyzing...";
  analyzeSpinner.hidden = false;
  diagnosisLoading.hidden = false;

  try {
    const response = await fetch(diagnosisEndpoint, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error("The diagnosis service returned an unreadable response.");
    }

    if (!validateDiagnosisResponse(result)) {
      throw new Error("The diagnosis service returned an unexpected response.");
    }

    renderDiagnosis(result);
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }

    diagnosisError.textContent =
      error instanceof TypeError
        ? "Could not connect to the diagnosis service. Check that the FastAPI backend is running at 127.0.0.1:8000."
        : error.message || "Diagnosis failed. Please try again.";
    diagnosisError.hidden = false;
  } finally {
    if (activeRequestController === controller) {
      activeRequestController = null;
      diagnosisLoading.hidden = true;
      analyzeButton.disabled = false;
      analyzeButtonLabel.textContent = "Analyze Crop Disease";
      analyzeSpinner.hidden = true;
    }
  }
}

cropOptions.forEach((option) => {
  option.addEventListener("click", () => {
    selectedCrop = option.dataset.crop;
    cropOptions.forEach((cropOption) => {
      cropOption.setAttribute("aria-pressed", String(cropOption === option));
    });
    cropError.textContent = "";
    clearDiagnosis();
  });
});

dropZone.addEventListener("click", (event) => {
  if (!event.target.closest("button")) {
    openFilePicker();
  }
});

dropZone.addEventListener("keydown", (event) => {
  if (event.target === dropZone && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    openFilePicker();
  }
});

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  if (file) {
    showImage(file);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) {
    showImage(file);
  }
});

removeImageButton.addEventListener("click", clearImage);
changeImageButton.addEventListener("click", () => {
  imageInput.value = "";
  openFilePicker();
});
analyzeButton.addEventListener("click", analyzeImage);

function closeMenu() {
  menuToggle.setAttribute("aria-expanded", "false");
  menuToggle.setAttribute("aria-label", "Open navigation menu");
  navMenu.classList.remove("is-open");
}

menuToggle.addEventListener("click", () => {
  const isExpanded = menuToggle.getAttribute("aria-expanded") === "true";
  menuToggle.setAttribute("aria-expanded", String(!isExpanded));
  menuToggle.setAttribute("aria-label", isExpanded ? "Open navigation menu" : "Close navigation menu");
  navMenu.classList.toggle("is-open", !isExpanded);
});

navMenu.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", closeMenu);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeMenu();
  }
});

window.addEventListener("beforeunload", () => {
  if (activeRequestController) {
    activeRequestController.abort();
  }
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
});
