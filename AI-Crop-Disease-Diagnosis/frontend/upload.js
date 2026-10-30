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
const diagnosisMessage = document.querySelector("#diagnosis-message");
const menuToggle = document.querySelector(".menu-toggle");
const navMenu = document.querySelector(".nav-menu");

const allowedExtensions = new Set(["jpg", "jpeg", "png", "webp"]);
const allowedMimeTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
const maxFileSize = 10 * 1024 * 1024;
let selectedCrop = "";
let previewUrl = "";

function openFilePicker() {
  imageInput.click();
}

function clearImage() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = "";
  }

  imageInput.value = "";
  previewImage.removeAttribute("src");
  previewName.textContent = "";
  previewSize.textContent = "";
  imagePreview.hidden = true;
  dropContent.hidden = false;
  imageError.textContent = "";
  uploadFeedback.textContent = "";
  diagnosisMessage.hidden = true;
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
    return "Choose an image in JPG, JPEG, PNG, or WEBP format.";
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
  diagnosisMessage.hidden = true;
  uploadFeedback.textContent = "";

  if (validationMessage) {
    clearImage();
    imageError.textContent = validationMessage;
    return;
  }

  clearImage();
  imageError.textContent = "";
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewName.textContent = file.name;
  previewSize.textContent = formatFileSize(file.size);
  imagePreview.hidden = false;
  dropContent.hidden = true;
  uploadFeedback.textContent = "Image ready to review.";
}

cropOptions.forEach((option) => {
  option.addEventListener("click", () => {
    selectedCrop = option.dataset.crop;
    cropOptions.forEach((cropOption) => {
      cropOption.setAttribute("aria-pressed", String(cropOption === option));
    });
    cropError.textContent = "";
    diagnosisMessage.hidden = true;
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

analyzeButton.addEventListener("click", () => {
  cropError.textContent = selectedCrop ? "" : "Please select a crop to continue.";
  imageError.textContent = previewUrl ? "" : "Please upload a crop leaf image to continue.";
  diagnosisMessage.hidden = true;

  if (!selectedCrop || !previewUrl) {
    return;
  }

  diagnosisMessage.hidden = false;
});

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
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
});
