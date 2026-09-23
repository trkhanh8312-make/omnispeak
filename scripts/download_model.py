"""Tải trước model OmniVoice (bỏ qua cũng được — backend sẽ tự tải)."""

MODEL_ID = "k2-fsa/OmniVoice"


def main(model_id=MODEL_ID):
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id)
    print("Model tại:", path)
    return path


if __name__ == "__main__":
    main()
