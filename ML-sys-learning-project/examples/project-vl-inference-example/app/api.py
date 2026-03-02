import json
import os
import shutil
import time
import uuid
from pathlib import Path

import redis
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


DATA_DIR = Path(os.getenv("DATA_DIR", "/data/workspace/platform_data"))
JOBS_DIR = DATA_DIR / "jobs"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "queue:infer")
ALLOWED_EXTS = {".jpg", ".jpeg"}

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def job_key(job_id: str) -> str:
    return "job:%s" % job_id


def write_status_mirror(job_dir: Path, data: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "status.json"
    status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_status(job_id: str, job_dir: Path, **fields) -> dict:
    key = job_key(job_id)
    if fields:
        redis_client.hset(key, mapping=fields)
    data = redis_client.hgetall(key)
    if data:
        write_status_mirror(job_dir, data)
    return data


def ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def on_startup() -> None:
    ensure_dirs()


@app.post("/api/infer")
def create_infer_job(file: UploadFile = File(...)) -> dict:
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Only .jpg/.jpeg files are supported.")

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / "input.jpg"
    with image_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    created_at = now_ms()
    jsonl_path = job_dir / "infer_out.jsonl"
    result_path = job_dir / "result.json"
    status = {
        "status": "queued",
        "created_at": str(created_at),
        "started_at": "",
        "finished_at": "",
        "duration_ms": "",
        "image_path": str(image_path),
        "jsonl_path": str(jsonl_path),
        "result_path": str(result_path),
        "error": "",
    }
    update_status(job_id, job_dir, **status)
    redis_client.lpush(QUEUE_NAME, job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/infer/{job_id}")
def get_infer_status(job_id: str) -> dict:
    key = job_key(job_id)
    data = redis_client.hgetall(key)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found.")
    return data


@app.get("/api/result/{job_id}")
def get_result(job_id: str) -> JSONResponse:
    job_dir = JOBS_DIR / job_id
    result_path = job_dir / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="result.json not found.")
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read result.json: %s" % exc)
    return JSONResponse(content=data)


@app.get("/api/image/{job_id}")
def get_image(job_id: str) -> FileResponse:
    job_dir = JOBS_DIR / job_id
    image_path = job_dir / "input.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="input.jpg not found.")
    return FileResponse(path=str(image_path))


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(path=str(index_path))
