# CropCare AI — AI Crop Disease Diagnosis System

CropCare AI is a crop disease diagnosis project. The frontend currently provides a responsive project landing page and an interactive crop image upload experience. The FastAPI backend loads the trained MobileNetV2 classifier and exposes an image diagnosis endpoint; the static frontend is not connected to the backend yet.

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
│   │   └── diagnosis.py # Health and model-backed image diagnosis endpoints
│   └── services/
│       ├── __init__.py
│       └── model_service.py # Model loading, preprocessing, and inference
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
├── model/
│   └── training/
│       └── best_model.keras # Trained 15-class classifier used by the backend
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

The backend is a Python **FastAPI** application, compatible with Python 3.11–3.13. `backend/requirements.txt` includes TensorFlow 2.21.0 and NumPy for local model inference. The trained model must be present at `model/training/best_model.keras`; startup fails explicitly if the model or the 15-class mapping is missing or incompatible.

- `fastapi`
- `uvicorn`
- `python-dotenv`
- `python-multipart`
- `pillow`
- `numpy`
- `tensorflow`

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
| `POST` | `/api/diagnose` | Accepts a JPEG or PNG image, validates its size and content, then returns model-backed predictions. |

The `/api/diagnose` endpoint expects multipart form data with:

- `image`: a valid JPG/JPEG or PNG leaf image no larger than 10 MB.

### Backend configuration

Backend settings can be supplied through environment variables or a `backend/.env` file:

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://127.0.0.1:5500,http://localhost:5500` | Comma-separated frontend origins allowed by CORS. |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum accepted image size in megabytes; must be a positive integer. |
| `ALLOWED_IMAGE_FORMATS` | `JPEG,PNG` | Image formats accepted by the backend (restricted to JPEG and PNG). |

The response includes the predicted class, its confidence percentage, the top three class scores, and a `healthy` or `diseased` status. Input images are converted to RGB, resized to 224x224 with TensorFlow's image resize operation, and processed with the same `tensorflow.keras.applications.mobilenet_v2.preprocess_input` function used during training. Predictions run only on the local trained model; no external AI service is used. The static upload page remains unconnected to the backend.

## Dataset Information

### Dataset Overview

CropCare AI uses the **PlantVillage dataset** for training and developing its crop disease classification system.

The dataset contains images of healthy and diseased plant leaves across 15 classification categories covering three crop groups:

- Bell Pepper
- Potato
- Tomato

The dataset is organised into class-specific directories, where each directory represents a particular crop disease or healthy condition.

### Dataset Statistics

| Specification | Value |
|---|---:|
| Dataset | PlantVillage |
| Original source image files | 20,638 |
| Unique readable images after deduplication | 20,624 |
| Number of classification classes | 15 |
| Corrupted images excluded | 0 |
| Exact duplicate copies excluded | 14 |
| Random seed | 42 |

### Dataset Class Distribution

The following counts represent the original dataset inspection, before duplicate removal.

| Crop | Disease / Condition | Original Images |
|---|---|---:|
| Bell Pepper | Bacterial Spot | 997 |
| Bell Pepper | Healthy | 1,478 |
| Potato | Early Blight | 1,000 |
| Potato | Healthy | 152 |
| Potato | Late Blight | 1,000 |
| Tomato | Target Spot | 1,404 |
| Tomato | Mosaic Virus | 373 |
| Tomato | Yellow Leaf Curl Virus | 3,208 |
| Tomato | Bacterial Spot | 2,127 |
| Tomato | Early Blight | 1,000 |
| Tomato | Healthy | 1,591 |
| Tomato | Late Blight | 1,909 |
| Tomato | Leaf Mold | 952 |
| Tomato | Septoria Leaf Spot | 1,771 |
| Tomato | Spider Mites | 1,676 |

### Dataset Splitting

The dataset is divided into three subsets using a stratified splitting strategy and random seed `42`.

| Dataset Split | Percentage | Images |
|---|---:|---:|
| Training | 70% | 14,436 |
| Validation | 15% | 3,094 |
| Testing | 15% | 3,094 |
| **Total** | **100%** | **20,624** |

- **Training set:** Used to train the classification model.
- **Validation set:** Used to monitor model performance and support training decisions.
- **Testing set:** Reserved for final evaluation on previously unseen images.

The test dataset is kept separate from model training and validation to help prevent data leakage during model development.

### Data Cleaning and Preprocessing

The preprocessing pipeline performs the following operations:

1. Scans the original PlantVillage class directories.
2. Checks image readability and validates image content.
3. Identifies exact duplicate images using SHA-256 hashes.
4. Excludes 14 duplicate copies while retaining one representative image from each duplicate group.
5. Ignores non-image files and preserves the original source images.
6. Creates reproducible training, validation and testing manifests.
7. Generates class mappings and class weights using training data only.

### Generated Dataset Files

The preprocessing scripts generate the following files under `dataset/processed/`:

| File | Purpose |
|---|---|
| `train_manifest.csv` | Training image paths and labels |
| `validation_manifest.csv` | Validation image paths and labels |
| `test_manifest.csv` | Reserved testing image paths and labels |
| `class_mapping.json` | Maps class names to numerical indices |
| `class_weights.json` | Stores training-only class weights |
| `preprocessing_report.txt` | Records dataset inspection and preprocessing information |

### Dataset Limitations

- The model is currently designed for 15 classes across Bell Pepper, Potato and Tomato.
- Class distribution is uneven, with some categories containing considerably more images than others.
- The dataset's performance may not represent real-world field conditions.
- Exact duplicate removal does not eliminate all possible similarities between images.
- The current preprocessing pipeline does not include random data augmentation.

The reserved test dataset should be used for a separate final model evaluation after training.

## Deep Learning Model — MobileNetV2

### Model Overview

CropCare AI uses **MobileNetV2**, a lightweight convolutional neural network (CNN), for crop disease classification.

The project uses **Transfer Learning**, where a MobileNetV2 model pretrained on the ImageNet dataset serves as the feature extraction backbone. The original classification head is replaced with a custom classification head designed to recognise 15 crop disease and healthy-leaf categories.

### Model Architecture

The model consists of the following layers:

| Layer | Purpose |
|---|---|
| Input Layer | Accepts RGB images of size 224 × 224 × 3 |
| MobileNetV2 Backbone | Extracts visual features from leaf images |
| Global Average Pooling 2D | Converts feature maps into a compact feature vector |
| Dropout | Applies a dropout rate of 0.3 to help reduce overfitting |
| Dense Output Layer | Produces predictions for 15 classes using Softmax activation |

**Architecture flow:**

```text
Input Image (224 × 224 × 3)
          |
          v
   MobileNetV2 Backbone
   (ImageNet pretrained)
   (Frozen feature extractor)
          |
          v
Global Average Pooling 2D
          |
          v
     Dropout (0.3)
          |
          v
 Dense Layer (15 Units)
   Softmax Activation
          |
          v
   15-Class Prediction
```

### Model Specifications

| Specification | Value |
|---|---|
| Model Architecture | MobileNetV2 |
| Learning Approach | Transfer Learning |
| Pretrained Weights | ImageNet |
| Input Image Size | 224 × 224 pixels |
| Input Channels | 3 (RGB) |
| Number of Classes | 15 |
| Total Parameters | 2,277,199 |
| Trainable Parameters | 19,215 |
| Non-trainable Parameters | 2,257,984 |
| Dropout Rate | 0.3 |
| Output Activation | Softmax |
| Loss Function | Sparse Categorical Crossentropy |
| Optimizer | Adam |

### Image Preprocessing

Before classification, the backend performs the following preprocessing operations:

1. Validates the uploaded image.
2. Converts the image into RGB format.
3. Resizes it to 224 × 224 pixels.
4. Converts the image into a TensorFlow-compatible numerical array.
5. Applies MobileNetV2's `preprocess_input` function.
6. Passes the processed image to the trained model.

The same preprocessing approach is used during training and inference to maintain consistency.

### Training Strategy

The model uses a transfer-learning approach:

- MobileNetV2 is initialised with ImageNet pretrained weights.
- The pretrained backbone is frozen during training.
- A custom classification head is trained for the 15 target classes.
- Class weights are used to account for differences in class distribution.
- EarlyStopping monitors validation loss.
- ModelCheckpoint saves the best model based on validation loss.
- ReduceLROnPlateau reduces the learning rate when validation performance stops improving.

Training is designed to run in Google Colab using an available GPU.

### Prediction and Classification

The final Dense layer uses Softmax activation to generate a score for each of the 15 classes.

The backend identifies the class with the highest output score and returns:

- Predicted disease or healthy-leaf class.
- Prediction score expressed as a percentage.
- Top three predicted classes and their scores.
- Classification status: `healthy` or `diseased`.

**Note:** Softmax scores represent model outputs and should not automatically be interpreted as calibrated probabilities or guaranteed prediction accuracy.

### Model Limitations

- Predictions depend on image quality and similarity to the training dataset.
- Performance on real-world field images may differ from performance on PlantVillage images.
- The model is designed for the 15 classes included in its class mapping.
- Predictions should not replace professional agricultural diagnosis.

## Current status and next steps

- The landing page and upload page are implemented.
- Frontend crop selection, image preview, and input validation are implemented.
- The backend loads and validates the trained model and provides real model-backed predictions.
- The frontend is not yet connected to the backend.
- The frontend still needs a separate integration to submit images and display API results.

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
