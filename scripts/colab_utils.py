"""Tiện ích dùng chung cho các script chạy trong Colab."""

import subprocess


def run(cmd, what=""):
    print("\n$", cmd if isinstance(cmd, str) else " ".join(cmd))
    if subprocess.run(cmd, shell=isinstance(cmd, str)).returncode != 0:
        raise SystemExit(f"Lỗi: {what or cmd}. Xem log phía trên rồi chạy lại cell này.")
