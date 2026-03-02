import json
import os
import subprocess
import sys
import time
from pathlib import Path

import redis


DATA_DIR = Path(os.getenv("DATA_DIR", "/data/workspace/platform_data"))
JOBS_DIR = DATA_DIR / "jobs"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "queue:infer")

INFER_PY = os.getenv("INFER_PY", "/data/workspace/batches/infer.py")
PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://127.0.0.1:9000/api/vllm")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen3-VL-30B-A3B-Instruct")
INFER_CONCURRENCY = int(os.getenv("INFER_CONCURRENCY", "1"))
INFER_TIMEOUT_SEC = int(os.getenv("INFER_TIMEOUT_SEC", "180"))

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


def build_command(job_dir: Path, jsonl_path: Path) -> list:
    return [
        PYTHON_BIN,
        INFER_PY,
        "--input-dir",
        str(job_dir),
        "--output-jsonl",
        str(jsonl_path),
        "--api-base",
        VLLM_API_BASE,
        "--model",
        VLLM_MODEL,
        "--concurrency",
        str(INFER_CONCURRENCY),
        "--timeout",
        str(INFER_TIMEOUT_SEC),
    ]


def select_record(jsonl_path: Path, image_id: str, image_path: str) -> dict | None:
    if not jsonl_path.exists():
        return None
    first = None
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if first is None:
                first = obj
            if obj.get("image_id") == image_id or obj.get("image") == image_path:
                return obj
    return first


def run_job(job_id: str) -> None:
    job_dir = JOBS_DIR / job_id
    image_path = job_dir / "input.jpg"
    jsonl_path = job_dir / "infer_out.jsonl"
    result_path = job_dir / "result.json"
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"

    if not image_path.exists():
        update_status(
            job_id,
            job_dir,
            status="failed",
            finished_at=str(now_ms()),
            error="input.jpg not found.",
        )
        return

    if jsonl_path.exists():
        jsonl_path.unlink()

    started_at = now_ms()
    update_status(
        job_id,
        job_dir,
        status="running",
        started_at=str(started_at),
        image_path=str(image_path),
        jsonl_path=str(jsonl_path),
        result_path=str(result_path),
        error="",
    )

    cmd = build_command(job_dir, jsonl_path)
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open(
            "w", encoding="utf-8"
        ) as err_f:
            proc = subprocess.run(
                cmd,
                stdout=out_f,
                stderr=err_f,
                timeout=INFER_TIMEOUT_SEC,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError("infer.py failed with return code %s" % proc.returncode)
        record = select_record(jsonl_path, "input", str(image_path))
        if record is None:
            raise RuntimeError("No record found in infer_out.jsonl.")

        result_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        finished_at = now_ms()
        update_status(
            job_id,
            job_dir,
            status="done",
            finished_at=str(finished_at),
            duration_ms=str(finished_at - started_at),
            error="",
        )
    except subprocess.TimeoutExpired:
        finished_at = now_ms()
        update_status(
            job_id,
            job_dir,
            status="failed",
            finished_at=str(finished_at),
            duration_ms=str(finished_at - started_at),
            error="infer.py timed out after %s seconds." % INFER_TIMEOUT_SEC,
        )
    except Exception as exc:
        finished_at = now_ms()
        update_status(
            job_id,
            job_dir,
            status="failed",
            finished_at=str(finished_at),
            duration_ms=str(finished_at - started_at),
            error=str(exc),
        )


def main() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    print("Worker started. Waiting for jobs on %s..." % QUEUE_NAME, flush=True)
    while True:
        try:
            item = redis_client.brpop(QUEUE_NAME, timeout=5)
            if not item:
                continue
            job_id = item[1]
            run_job(job_id)
        except KeyboardInterrupt:
            print("Worker stopped.", flush=True)
            return
        except Exception as exc:
            print("Worker loop error: %s" % exc, flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
