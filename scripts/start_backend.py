"""Khởi động backend OmniSpeak (uvicorn) trong Colab."""

import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = 3900
LOG_PATH = "/content/omnispeak_backend.log"


def health(port=PORT, timeout=5):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def start(app_dir, frontend_dir, data_dir=None, port=PORT,
          force_restart=True, log_path=LOG_PATH, timeout=300):
    if not force_restart:
        info = health(port)
        if info:
            print("Backend đang chạy —", info)
            return info

    subprocess.run(f"kill -9 $(lsof -t -i:{port}) 2>/dev/null || true", shell=True)
    time.sleep(1)

    env = os.environ.copy()
    env["OMNISPEAK_DATA_DIR"] = data_dir or os.environ.get(
        "OMNISPEAK_DATA_DIR", "/content/omnispeak_data")
    env["OMNISPEAK_FRONTEND_DIR"] = frontend_dir
    env["PYTHONUNBUFFERED"] = "1"

    # Mỗi lần backend thực sự khởi động lại (không phải chỉ kiểm tra health),
    # ghi log mới — tránh log cũ tích luỹ qua nhiều lần chạy lại cell trong cùng phiên.
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend:app", "--app-dir", app_dir,
         "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"Đang khởi động (PID {proc.pid})...")

    info = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        info = health(port)
        if info:
            break
        print(".", end="", flush=True)
        time.sleep(3)
    print()

    if info:
        print("Backend đã sẵn sàng —", info)
        return info

    try:
        tail = "".join(open(log_path, errors="replace").readlines()[-40:])
    except OSError:
        tail = "(không có log)"
    raise SystemExit(f"Backend không lên được sau {timeout // 60} phút.\n--- log ---\n{tail}")
