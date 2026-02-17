# apmain.py

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
import os

from main import (
    load_models,
    process_image_to_fen
)

# -------------------------
# FastAPI Setup
# -------------------------
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# Load models once (global)
# -------------------------
detector, classifier = load_models(
    "models/detect/board_train/weights/best.pt",
    "models/classify/piece_classifier_v1/weights/best.pt"
)


# -------------------------
# PAGE 1: Upload Page
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------
# PAGE 2: Process Image
# -------------------------
@app.post("/process", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...)):
    # Save uploaded file
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model pipeline
    fen_code = process_image_to_fen(file_path, detector, classifier)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "fen": fen_code,
            "image_path": "/" + file_path
        }
    )
