import io
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import BackgroundTasks, FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from omnivoice import OmniVoice, VoiceClonePrompt

DATA_DIR = Path(os.environ.get("OMNISPEAK_DATA_DIR", "/content/omnispeak_data"))
PROFILES_DIR = DATA_DIR / "profiles"
INDEX_PATH = DATA_DIR / "profiles.json"
FRONTEND_DIR = os.environ.get("OMNISPEAK_FRONTEND_DIR", "/content/omnispeak_frontend")
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# ---- Log có timestamp/level rõ ràng (in ra stdout, cell 7 đã redirect vào omnispeak_backend.log) ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("omnispeak")

# Giới hạn văn bản: người dùng thường dùng ~1000-2000 từ, chặn cứng ở 3000 từ.
HARD_WORD_LIMIT = 3000
CHUNK_MAX_CHARS = 400  # mỗi lần gọi model.generate() xử lý tối đa ~400 ký tự
SR = 24000
GAP_SECONDS = 0.25  # khoảng lặng chèn giữa các đoạn khi ghép lại

# Giới hạn file mẫu giọng khi upload
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
MIN_REF_SECONDS = 0.5
MAX_REF_SECONDS = 120
PREVIEW_SECONDS = 8       # đoạn mẫu ngắn giữ lại để nghe thử trong thư viện
MAX_PROFILES = 20         # giới hạn số giọng lưu trong thư viện
NAME_MAX_LEN = 80         # giới hạn độ dài tên giọng nói
REF_TEXT_MAX_LEN = 500    # giới hạn độ dài ref_text (transcript của mẫu giọng)

# Dọn job cũ khỏi bộ nhớ
JOB_RESULT_TTL = 30 * 60     # job đã xong nhưng không ai lấy audio -> xoá sau 30 phút
JOB_STALE_TTL = 2 * 60 * 60  # job kẹt bất thường quá 2 tiếng -> xoá luôn (an toàn)

# profile_id / job_id đều là uuid4().hex[:12] -> chỉ gồm hex 12 ký tự.
# Bắt buộc khớp định dạng này trước khi ghép vào đường dẫn file, tránh path traversal
# (vd ai đó gửi profile_id="../../etc/passwd" qua URL).
ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _validate_id(id_: str):
    if not ID_RE.fullmatch(id_ or ""):
        raise HTTPException(400, "ID không hợp lệ")


app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
logger.info(f"Đang tải model OmniVoice lên {device}...")
MODEL = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype, load_asr=True)
logger.info("Model đã sẵn sàng.")

GEN_LOCK = threading.Lock()  # chỉ chạy 1 job trên GPU tại một thời điểm
JOBS: dict = {}  # job_id -> {status, total, done, audio_bytes, gen_time, error, created}


def _cleanup_jobs():
    """Xoá job đã xong lâu mà không ai lấy kết quả, và job kẹt bất thường quá lâu."""
    now = time.time()
    stale = []
    for jid, job in JOBS.items():
        age = now - job.get("created", now)
        if job["status"] in ("done", "error") and age > JOB_RESULT_TTL:
            stale.append(jid)
        elif age > JOB_STALE_TTL:
            stale.append(jid)
    for jid in stale:
        JOBS.pop(jid, None)
    if stale:
        logger.info(f"Đã dọn {len(stale)} job cũ khỏi bộ nhớ.")


def _has_active_job():
    return any(j["status"] in ("pending", "running") for j in JOBS.values())


def _load_index():
    return json.loads(INDEX_PATH.read_text()) if INDEX_PATH.exists() else []


def _save_index(items):
    INDEX_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2))


def _split_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS):
    """Tách văn bản thành các đoạn nhỏ theo câu, mỗi đoạn tối đa ~max_chars ký tự."""
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks, buf = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            # câu quá dài — cắt theo khoảng trắng
            words = s.split(" ")
            piece = ""
            for w in words:
                if len(piece) + len(w) + 1 > max_chars:
                    if piece:
                        chunks.append(piece.strip())
                    piece = w
                else:
                    piece = (piece + " " + w).strip()
            if piece:
                s = piece
            else:
                continue
        if len(buf) + len(s) + 1 <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks or [text.strip()]


def _run_job(job_id: str, text: str, profile_id: Optional[str]):
    job = JOBS[job_id]
    job["status"] = "running"
    logger.info(f"[{job_id}] Bắt đầu tạo giọng nói — {len(text)} ký tự, profile={profile_id or 'mặc định'}")
    try:
        kwargs = {}
        if profile_id:
            p = PROFILES_DIR / f"{profile_id}.pt"
            if not p.exists():
                raise RuntimeError(f"Profile {profile_id} not found")
            kwargs["voice_clone_prompt"] = VoiceClonePrompt.load(str(p))

        chunks = _split_into_chunks(text)
        job["total"] = len(chunks)

        t0 = time.time()
        pieces = []
        gap = np.zeros(int(GAP_SECONDS * SR), dtype=np.float32)

        with GEN_LOCK:
            for i, chunk in enumerate(chunks):
                audio = MODEL.generate(text=chunk, **kwargs)
                wav = audio[0]
                if isinstance(wav, torch.Tensor):
                    wav = wav.detach().cpu().float().numpy()
                wav = np.asarray(wav, dtype=np.float32)
                if wav.ndim == 2:
                    wav = wav.T
                    if wav.shape[1] == 1:
                        wav = wav[:, 0]
                pieces.append(wav)
                if i < len(chunks) - 1:
                    pieces.append(gap)
                job["done"] = i + 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        full = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        buf = io.BytesIO()
        sf.write(buf, full, SR, format="WAV", subtype="PCM_16")
        job["audio_bytes"] = buf.getvalue()
        job["gen_time"] = time.time() - t0
        job["status"] = "done"
        logger.info(f"[{job_id}] Xong trong {job['gen_time']:.2f}s ({len(chunks)} đoạn).")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        logger.error(f"[{job_id}] Lỗi khi tạo giọng nói: {e}\n{traceback.format_exc()}")


@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.post("/generate")
async def generate(background_tasks: BackgroundTasks, text: str = Form(...), profile_id: Optional[str] = Form(None)):
    _cleanup_jobs()
    text = text.strip()
    if not text:
        raise HTTPException(400, "Văn bản trống")
    word_count = len(text.split())
    if word_count > HARD_WORD_LIMIT:
        raise HTTPException(413, f"Văn bản vượt quá {HARD_WORD_LIMIT} từ (hiện tại: {word_count} từ)")
    if profile_id:
        _validate_id(profile_id)
        p = PROFILES_DIR / f"{profile_id}.pt"
        if not p.exists():
            raise HTTPException(404, f"Profile {profile_id} not found")
    if _has_active_job():
        raise HTTPException(429, "Đang có một yêu cầu tạo giọng khác được xử lý, vui lòng đợi rồi thử lại.")

    chunks = _split_into_chunks(text)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "pending", "total": len(chunks), "done": 0,
                     "audio_bytes": None, "gen_time": None, "error": None,
                     "created": time.time()}
    logger.info(f"[{job_id}] Job mới — {word_count} từ, {len(chunks)} đoạn.")
    background_tasks.add_task(_run_job, job_id, text, profile_id)
    return {"job_id": job_id, "total_chunks": len(chunks)}


@app.get("/generate/{job_id}/status")
def generate_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"status": job["status"], "total": job["total"], "done": job["done"], "error": job["error"]}


@app.get("/generate/{job_id}/audio")
def generate_audio(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "error":
        raise HTTPException(500, job["error"] or "Lỗi tạo giọng nói")
    if job["status"] != "done":
        raise HTTPException(409, "Job chưa xong")
    data = job["audio_bytes"]
    gen_time = job["gen_time"]
    JOBS.pop(job_id, None)  # dọn bộ nhớ sau khi trả kết quả
    return Response(content=data, media_type="audio/wav",
                     headers={"X-Gen-Time": f"{gen_time:.2f}"})


@app.get("/profiles")
def list_profiles():
    return _load_index()


@app.get("/profiles/{profile_id}/preview")
def profile_preview(profile_id: str):
    _validate_id(profile_id)
    path = PROFILES_DIR / f"{profile_id}_preview.wav"
    if not path.exists():
        raise HTTPException(404, "Giọng này chưa có bản nghe thử (được tạo trước khi tính năng này ra mắt).")
    return Response(content=path.read_bytes(), media_type="audio/wav")


@app.post("/profiles")
async def create_profile(name: str = Form(...), kind: str = Form("clone"),
                          ref_audio: UploadFile = File(...), ref_text: Optional[str] = Form(None)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Tên giọng nói không được để trống")
    if len(name) > NAME_MAX_LEN:
        raise HTTPException(400, f"Tên giọng nói vượt quá {NAME_MAX_LEN} ký tự")
    if ref_text and len(ref_text) > REF_TEXT_MAX_LEN:
        raise HTTPException(400, f"ref_text vượt quá {REF_TEXT_MAX_LEN} ký tự")

    items = _load_index()
    existing = next((p for p in items if p["name"] == name), None)
    if existing:
        return existing
    if len(items) >= MAX_PROFILES:
        raise HTTPException(429, f"Thư viện đã đạt giới hạn {MAX_PROFILES} giọng nói — hãy xoá bớt giọng cũ trước khi thêm mới.")

    raw = await ref_audio.read()
    if not raw:
        raise HTTPException(400, "File mẫu giọng trống")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File mẫu giọng vượt quá {MAX_UPLOAD_BYTES // (1024*1024)}MB")

    profile_id = uuid.uuid4().hex[:12]
    tmp = DATA_DIR / f"_upload_{profile_id}.wav"
    tmp.write_bytes(raw)

    try:
        # Kiểm tra file có phải audio hợp lệ + thời lượng hợp lý trước khi đưa vào model
        try:
            info = sf.info(str(tmp))
        except Exception:
            raise HTTPException(400, "File mẫu không phải audio hợp lệ (thử .wav, .mp3, .m4a...)")
        if info.duration < MIN_REF_SECONDS:
            raise HTTPException(400, f"Mẫu giọng quá ngắn ({info.duration:.1f}s) — cần ít nhất {MIN_REF_SECONDS}s")
        if info.duration > MAX_REF_SECONDS:
            raise HTTPException(400, f"Mẫu giọng quá dài ({info.duration:.0f}s) — tối đa {MAX_REF_SECONDS}s")

        # Lưu lại một đoạn mẫu ngắn để nghe thử trong thư viện (không lưu toàn bộ file gốc)
        try:
            data, sr = sf.read(str(tmp))
            preview = data[: int(PREVIEW_SECONDS * sr)]
            sf.write(str(PROFILES_DIR / f"{profile_id}_preview.wav"), preview, sr, subtype="PCM_16")
        except Exception as e:
            logger.error(f"Không tạo được bản xem trước cho '{name}': {e}")

        try:
            prompt = MODEL.create_voice_clone_prompt(ref_audio=str(tmp), ref_text=ref_text or None)
            prompt.save(str(PROFILES_DIR / f"{profile_id}.pt"))
        except Exception as e:
            logger.error(f"Lỗi tạo voice clone prompt cho '{name}': {e}\n{traceback.format_exc()}")
            (PROFILES_DIR / f"{profile_id}_preview.wav").unlink(missing_ok=True)
            raise HTTPException(500, f"Không tạo được giọng nói từ mẫu này: {e}")
    finally:
        tmp.unlink(missing_ok=True)

    entry = {"id": profile_id, "name": name, "kind": kind}
    items.append(entry)
    _save_index(items)
    logger.info(f"Đã tạo giọng mới: {name} ({profile_id})")
    return entry


@app.delete("/profiles/{profile_id}")
def delete_profile(profile_id: str):
    _validate_id(profile_id)
    items = _load_index()
    remaining = [p for p in items if p["id"] != profile_id]
    if len(remaining) == len(items):
        raise HTTPException(404, "Not found")
    _save_index(remaining)
    (PROFILES_DIR / f"{profile_id}.pt").unlink(missing_ok=True)
    (PROFILES_DIR / f"{profile_id}_preview.wav").unlink(missing_ok=True)
    logger.info(f"Đã xoá giọng: {profile_id}")
    return {"deleted": profile_id}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
