# OmniSpeak (Colab)

Đọc văn bản (TTS) và nhân bản giọng nói, chạy trên Google Colab (T4 GPU), dùng model [OmniVoice](https://github.com/k2-fsa/OmniVoice) (`k2-fsa/OmniVoice`, Apache-2.0).

## Cấu trúc repo

- `frontend/` — giao diện web
  - `index.html` — khung giao diện
  - `style.css` — toàn bộ CSS
  - `app.js` — toàn bộ logic JS (gọi API, ghi âm, phát lại...)
- `backend/` — server xử lý
  - `backend.py` — FastAPI server: TTS, thư viện giọng, lịch sử
  - `requirements.txt` — danh sách package cần cài
- `scripts/` — các bước chạy trong Colab, tách riêng để notebook gọn
  - `colab_utils.py` — tiện ích dùng chung (chạy lệnh, báo lỗi)
  - `install_deps.py` — cài apt + pip package
  - `download_model.py` — tải trước model OmniVoice
  - `start_backend.py` — khởi động uvicorn, theo dõi health
- `.github/workflows/` — kiểm tra tự động khi commit (JSON notebook, cú pháp Python)
- `omnispeak.ipynb` — notebook chạy trên Colab, chỉ đồng bộ code rồi gọi các script trên

Notebook không nhúng code app — mọi logic nằm trên GitHub, notebook chỉ `git clone`/`git pull` code mới nhất về rồi gọi.

## Cách chạy

1. Mở `omnispeak.ipynb` bằng Google Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Chạy các cell theo thứ tự từ trên xuống (hoặc `Runtime → Run all`).
4. Cell 2 có form để chỉnh `GITHUB_USER` / `GITHUB_REPO` / `GITHUB_BRANCH` nếu bạn dùng bản fork riêng.
5. Cell 4 (mount Google Drive) là tuỳ chọn — bật lên nếu muốn giữ giọng nói + cache model qua các phiên sau. Mỗi phiên chạy mới, Google sẽ hỏi quyền truy cập Drive lại — đây là giới hạn bảo mật của Colab, không có cách bỏ qua.
6. Cell 7 mở giao diện web.

## Bảo mật

Giao diện được bảo vệ bằng mật khẩu qua header `X-Omnispeak-Key`. Đặt mật khẩu trong Colab Secrets với tên `OMNISPEAK_SECRET`. Trình duyệt sẽ hỏi mật khẩu 1 lần đầu rồi lưu lại (`localStorage`) cho các lần mở sau.

## Xử lý sự cố

Xem mục "Xử lý sự cố" ở cuối `omnispeak.ipynb`.

## Giấy phép

Xem [LICENSE](LICENSE). Model OmniVoice thuộc giấy phép Apache-2.0 riêng — xem [repo gốc](https://github.com/k2-fsa/OmniVoice).
