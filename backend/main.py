import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT_DIR / "backend" / "uploads"
OUTPUT_DIR = ROOT_DIR / "backend" / "outputs"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT_DIR /"weights" / "best.pt")))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YOLOv8 Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

_model = None
video_executor = ThreadPoolExecutor(max_workers=1)
video_tasks = {}


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def parse_detections(result):
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append(
            {
                "class_id": cls_id,
                "class_name": result.names.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            }
        )
    return detections


def find_first_file(directory: Path):
    candidates = sorted(directory.glob("*"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def run_video_task(task_id: str, upload_path: Path):
    try:
        video_tasks[task_id]["status"] = "running"
        model = get_model()
        model.predict(
            source=str(upload_path),
            conf=0.25,
            imgsz=640,
            save=True,
            project=str(OUTPUT_DIR),
            name=task_id,
            exist_ok=True,
            verbose=False,
        )

        output_subdir = OUTPUT_DIR / task_id
        output_video = output_subdir / upload_path.name
        if not output_video.exists():
            guessed_file = find_first_file(output_subdir)
            if guessed_file is None:
                raise FileNotFoundError("No output video generated.")
            output_video = guessed_file

        video_tasks[task_id]["status"] = "completed"
        video_tasks[task_id]["video_url"] = f"/outputs/{task_id}/{output_video.name}"
    except Exception as exc:
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(exc)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "video_worker_busy": any(task["status"] == "running" for task in video_tasks.values()),
    }


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    req_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{req_id}{suffix}"
    output_subdir = OUTPUT_DIR / req_id
    output_subdir.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        model = get_model()
        results = model.predict(
            source=str(upload_path),
            conf=0.25,
            imgsz=640,
            save=True,
            project=str(OUTPUT_DIR),
            name=req_id,
            exist_ok=True,
            verbose=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    result = results[0]
    boxes = parse_detections(result)

    output_image_name = upload_path.name
    output_image_path = output_subdir / output_image_name
    if not output_image_path.exists():
        candidates = sorted(output_subdir.glob("*"))
        if not candidates:
            raise HTTPException(status_code=500, detail="Inference finished but no output image was generated.")
        output_image_path = candidates[0]

    image_url = f"/outputs/{req_id}/{output_image_path.name}"
    return JSONResponse(
        {
            "message": "Detection success",
            "request_id": req_id,
            "image_url": image_url,
            "detections": boxes,
        }
    )


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video file.")

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    task_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{task_id}{suffix}"
    output_subdir = OUTPUT_DIR / task_id
    output_subdir.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    video_tasks[task_id] = {"status": "queued", "video_url": None, "error": None}
    video_executor.submit(run_video_task, task_id, upload_path)
    return {"message": "Video task submitted", "task_id": task_id, "status": "queued"}


@app.get("/api/tasks/{task_id}")
def get_task_status(task_id: str):
    task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"task_id": task_id, **task}
