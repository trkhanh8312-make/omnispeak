"""Cài hệ thống + Python package cho OmniSpeak."""

import hashlib
import sys

from colab_utils import run

APT_PACKAGES = ["ffmpeg", "libsndfile1"]
HASH_MARKER = "/content/.omnispeak_install_hash"


def _imports_ok():
    try:
        import fastapi  # noqa: F401
        import soundfile  # noqa: F401
        import omnivoice  # noqa: F401
        return True
    except ImportError:
        return False


def _file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _marker_matches(current_hash):
    try:
        with open(HASH_MARKER) as f:
            return f.read().strip() == current_hash
    except FileNotFoundError:
        return False


def _write_marker(current_hash):
    with open(HASH_MARKER, "w") as f:
        f.write(current_hash)


def main(requirements_path=None, force=False):
    if not requirements_path:
        raise SystemExit(
            "Thiếu requirements_path — truyền đường dẫn tới backend/requirements.txt."
        )

    current_hash = _file_hash(requirements_path)

    if not force and _imports_ok() and _marker_matches(current_hash):
        print("requirements.txt không đổi và đã cài đủ package — bỏ qua bước cài đặt.")
        print("(Muốn cài lại từ đầu: gọi main(force=True))")
        return

    run(f"apt-get -qq update && apt-get -qq install -y {' '.join(APT_PACKAGES)}",
        "cài gói hệ thống")

    run([sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
        "cài Python package từ requirements.txt")

    run([sys.executable, "-c",
         "from omnivoice import OmniVoice, VoiceClonePrompt; import fastapi, soundfile; print('OK')"],
        "kiểm tra import")

    _write_marker(current_hash)


if __name__ == "__main__":
    main()
