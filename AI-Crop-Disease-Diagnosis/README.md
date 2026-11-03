# CropCare AI — AI Crop Disease Diagnosis System

CropCare AI is a crop disease diagnosis project. The frontend currently provides a responsive project landing page and an interactive crop image upload experience. The backend provides API health checks and validates submitted crop images, but **AI model integration is pending**: it does not identify diseases or return real predictions.

## Project structure

```text
AI-Crop-Disease-Diagnosis/
├── frontend/
│   ├── index.html       # Landing page
│   ├── style.css        # Shared landing page design system
│   ├── script.js        # Landing page navigation and reveal animations
│   ├── upload.html      # Crop selection and image upload page
│   ├── upload.css       # Upload page layout and responsive styles
│   └── upload.js        # Crop selection, local image preview, and validation
├── backend/
│   ├── main.py          # FastAPI app, CORS configuration, root health response
│   ├── config.py        # Environment-based backend settings
│   ├── requirements.txt # Python backend dependencies
│   ├── .env             # Local backend configuration (ignored by Git)
│   ├── .gitignore       # Ignores local environment and temporary files
│   ├── routes/
│   │   ├── __init__.py
│   │   └── diagnosis.py # Health and image-validation API endpoints
│   └── services/
│       └── __init__.py  # Placeholder for backend services
├── scripts/
│   ├── inspect_dataset.py          # Read-only PlantVillage inspection
│   ├── prepare_dataset.py          # Deduplicate and create stratified manifests
│   ├── verify_preprocessing.py     # Verify manifests and MobileNetV2 inputs
│   ├── train_model.py              # Explicit opt-in MobileNetV2 training script
│   ├── training_config.json        # Seed, epochs, batch size, learning rate, etc.
│   ├── requirements-preprocessing.txt
│   └── requirements-training.txt   # Stable TensorFlow training environment
├── notebooks/
│   └── cropcare_mobilenetv2_training.ipynb # Google Colab GPU training workflow
├── dataset/
│   ├── raw/             # Original dataset; preprocessing never modifies it
│   └── processed/       # Generated manifests, class metadata, and report
├── model/               # Reserved for a future trained AI model
└── README.md
```

## Frontend

The frontend uses **HTML5, CSS3, and vanilla JavaScript**. It does not use React, Bootstrap, Tailwind, npm, or a `package.json`.

### Landing page

Open `frontend/index.html` in a browser. The landing page includes:

- CropCare AI branding and responsive navigation.
- Hero section, how-it-works steps, feature cards, supported crop cards, and project information.
- A sample diagnosis panel explicitly labeled as a demonstration, not a real diagnosis.
- Diagnosis and get-started links to `upload.html`.
- Smooth anchor scrolling, responsive mobile navigation, hover states, and scroll reveal effects.

### Upload and diagnosis preparation page

Open `frontend/upload.html` directly or navigate there using a diagnosis button on the landing page. The page includes:

- Responsive navigation back to the landing page.
- Selectable crop cards for Tomato, Potato, Apple, Grape, Corn, and Bell Pepper.
- Image selection through the file picker or drag and drop.
- JPG, JPEG, PNG, and WEBP file validation with a 10 MB maximum size.
- A local image preview showing the filename and file size, with remove and change controls.
- Photo-taking guidelines and a notice that AI diagnosis is not available yet.
- Form validation and the message “AI diagnosis integration coming soon.” after valid crop and image selections.

The upload page uses `URL.createObjectURL` to preview a selected image in the browser. The current frontend does **not** upload that image to the backend or perform disease detection. It does not generate predictions.

### Opening the frontend

You can open `frontend/index.html` or `frontend/upload.html` directly in a modern browser. Alternatively, serve the `frontend` folder with a static development server such as VS Code Live Server. No npm installation or build step is required.

The page loads its fonts and crop photographs from external services, so those assets require an internet connection.

## Backend API

The backend is a Python **FastAPI** application, compatible with Python 3.11 or newer. The current `backend/requirements.txt` lists:

- `fastapi`
- `uvicorn`
- `python-dotenv`
- `python-multipart`
- `pillow`

### Run the backend (optional)

From a terminal, change to the `backend` directory, create and activate a Python virtual environment, install the listed requirements, and start the development server:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

If Python 3.11 is not installed, replace `-3.11` with an installed Python version that is 3.11 or newer. The included `.env` contains local development defaults; it is ignored by Git. Adjust `CORS_ORIGINS` if your static frontend server uses a different address.

The development server uses `http://127.0.0.1:8000` by default. FastAPI's interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Confirms the backend is running. |
| `GET` | `/api/health` | Reports that the diagnosis API is ready. |
| `POST` | `/api/diagnose` | Accepts a crop name and image, validates the supported crop, file size, and image format, then returns a pending response. **No disease prediction is made.** |

The `/api/diagnose` endpoint expects multipart form data with:

- `crop_name`: Tomato, Potato, Apple, Grape, Corn, or Bell Pepper.
- `image`: a valid JPEG, PNG, or WEBP image within the configured maximum size.

### Backend configuration

Backend settings can be supplied through environment variables or a `backend/.env` file:

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://127.0.0.1:5500,http://localhost:5500` | Comma-separated frontend origins allowed by CORS. |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum accepted image size in megabytes; must be a positive integer. |
| `ALLOWED_IMAGE_FORMATS` | `JPEG,PNG,WEBP` | Comma-separated image formats accepted by the backend. |

The static upload page currently validates and previews images in the browser only; connecting its form to the backend remains a future integration step.

## Current status and next steps

- The landing page and upload page are implemented.
- Frontend crop selection, image preview, and input validation are implemented.
- The backend API and image-validation endpoint are present.
- The frontend is not yet connected to the backend.
- No disease-detection model or real AI diagnosis is implemented.
- `dataset/` and `model/` are reserved for future data and model work.

### Prepare and verify PlantVillage data (no model training)

Install the preprocessing dependencies into a Python virtual environment:

```powershell
python -m pip install -r scripts\requirements-preprocessing.txt
```

Create reproducible, deduplicated train/validation/test manifests and training-only class weights:

```powershell
python scripts\prepare_dataset.py
python scripts\verify_preprocessing.py
```

The scripts discover classes from `dataset/raw/PlantVillage`, preserve original image files, and write metadata only under `dataset/processed/`. Split assignments use random seed `42`; exact duplicate images are represented once. The verification script checks the manifests and TensorFlow MobileNetV2 preprocessing but does not train a model.

### Prepare the local MobileNetV2 environment (do not train locally)

The stable local training environment uses Python 3.13 and TensorFlow `2.21.0`. On native Windows, TensorFlow 2.11 and newer does not use NVIDIA CUDA GPUs. Use the Colab notebook below for GPU training; do not start a long training run on the local CPU.

```powershell
py -3.13 -m venv .venv-training
.\.venv-training\Scripts\Activate.ps1
python -m pip install -r scripts\requirements-training.txt
python scripts\train_model.py
```

The final command only checks the environment and manifests. It does not build or train a model. Training requires an explicit `--train` option; do not use it for Colab preparation.

### Train MobileNetV2 in Google Colab

The notebook at `notebooks/cropcare_mobilenetv2_training.ipynb` trains on a Colab GPU, using the existing training and validation manifests, class mapping, training-only class weights, and `scripts/training_config.json`. It checks that a GPU is available and stops with an explanation if the runtime has none. It does not load the test manifest or modify the dataset.

1. In Google Drive, use the project root `/content/drive/MyDrive/CropCare-AI` and place the raw class folders under `dataset/raw/PlantVillage/PlantVillage/`. Copy `scripts/training_config.json` and only these processed files: `train_manifest.csv`, `validation_manifest.csv`, `class_mapping.json`, and `class_weights.json`. Preserve their directory structure and do not copy or open the test manifest.
2. Upload `notebooks/cropcare_mobilenetv2_training.ipynb` to Google Colab and open it.
3. In Colab, select **Runtime > Change runtime type > T4 GPU** (or another available GPU), then save the setting. GPU availability varies; the notebook verifies the selected runtime before training.
4. Run the Drive setup cell and authorize Drive. The notebook uses the exact project and raw-data paths above. Run the training-input preflight first; it checks class folders, original-versus-Drive image counts, every training/validation image path, and duplicate hashes. It stops on missing images or mismatched counts.
5. Review the reported resolved counts (14,436 training and 3,094 validation), class mapping, and training configuration. Run the GPU check and training cell only after the preflight succeeds and you are ready to train.
6. The notebook saves `best_model.keras`, `class_mapping.json`, `training_history.json`, and per-epoch `training_history.csv` under `model/training/` in the Drive project folder. It prints the actual completed epochs and best validation metrics.
7. Run the export cell to create and download `cropcare_mobilenetv2_artifacts.zip`, containing the best model, mapping, config, and history. Keep the Colab session open until the download completes.

The notebook uses ImageNet MobileNetV2 with the base frozen and the configured batch size, epoch limit, and callbacks. Early stopping may finish before the maximum epoch count. No training metrics are assumed in advance; they are reported only after a completed run. The test split remains reserved for a separately planned evaluation.

## Important note

This project is not a substitute for professional agricultural advice. Do not rely on the demonstration content or the current upload page for crop treatment decisions. Consult a qualified agricultural expert when needed.
