# OmniSpeak

App đọc văn bản và nhân bản giọng nói (Việt/Anh), chạy trên Google Colab, dùng model [OmniVoice](https://github.com/k2-fsa/OmniVoice) (`k2-fsa/OmniVoice`, Apache-2.0).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trkhanh8312-make/omnispeak/blob/main/omnispeak.ipynb)

## Cấu trúc repo

```
omnispeak/
├── omnispeak.ipynb        # notebook chạy trên Google Colab
├── frontend/
│   └── index.html         # giao diện web
└── backend/
    └── backend.py         # FastAPI backend gọi thẳng OmniVoice
```

Notebook không nhúng code app — mỗi lần chạy sẽ `git clone`/`git pull` toàn bộ repo này về Colab rồi chạy trực tiếp từ đó.

## Cách chạy

1. Mở `omnispeak.ipynb` trên Google Colab (bấm badge phía trên).
2. `Runtime → Change runtime type → T4 GPU`.
3. Chạy lần lượt các cell từ trên xuống. Mọi cell đều an toàn khi chạy lại.
4. Cell cuối cùng ("Mở giao diện web") sẽ mở tab giao diện để dùng.

## Sửa code / cập nhật

1. Sửa `frontend/index.html` hoặc `backend/backend.py` → commit & push lên GitHub.
2. Trong notebook đang chạy: chạy lại cell **"Lấy code"** (đồng bộ code mới nhất từ repo), rồi cell **"Khởi động backend"** — `FORCE_RESTART = True` sẽ tự nạp code mới, không cần tự kill process hay khởi động lại Colab.

## Lưu dữ liệu lâu dài

Nếu muốn giữ giọng nói đã nhân bản và cache model qua các phiên Colab sau, chạy cell **"Mount Google Drive"** trước cell tải model — dữ liệu sẽ lưu vào `MyDrive/omnispeak_data` và `MyDrive/omnispeak_hf_cache`.

## Model & License

- Model: [OmniVoice](https://github.com/k2-fsa/OmniVoice) — `k2-fsa/OmniVoice`, giấy phép Apache-2.0.
- Backend: FastAPI, gọi thẳng OmniVoice, không qua thư viện app trung gian nào khác.
