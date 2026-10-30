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
│   ├── routes/
│   │   ├── __init__.py
│   │   └── diagnosis.py # Health and image-validation API endpoints
│   └── services/
│       └── __init__.py  # Placeholder for backend services
├── dataset/             # Reserved for crop image datasets
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

The backend is a Python **FastAPI** application. The current `backend/requirements.txt` lists:

- `fastapi`
- `uvicorn`
- `python-dotenv`
- `python-multipart`
- `pillow`

### Run the backend (optional)

From a terminal, change to the `backend` directory, create and activate a Python virtual environment, install the listed requirements, and start the development server:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload
```

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

## Important note

This project is not a substitute for professional agricultural advice. Do not rely on the demonstration content or the current upload page for crop treatment decisions. Consult a qualified agricultural expert when needed.
