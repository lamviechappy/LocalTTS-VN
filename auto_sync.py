import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# --- CẤU HÌNH ĐƯỜNG DẪN ---
REPO_ID = "dolly-vn/viterbox"
# Đường dẫn SSD của bạn (Nơi lưu trữ cố định)
DEST_DIR = "/Volumes/SSD256/ai-models/TTS/Viterbox/snapshots/6ddcc1430e2c1c67c5cf8e5c30b2c6952e6191db"

# Các file quan trọng cần đồng bộ
PATTERNS = [
    "ve.pt",
    "t3_ml24ls_v2.safetensors",
    "s3gen.pt",
    "tokenizer_vi_expanded.json",
    "conds.pt",
]

def sync_model():
    print(f"🌐 Đang kiểm tra phiên bản mới từ Hugging Face: {REPO_ID}...")
    try:
        # 1. Tải về cache tạm thời của hệ thống
        cache_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            allow_patterns=PATTERNS,
            resume_download=True
        )
        
        print(f"✅ Đã tải/kiểm tra xong tại Cache: {cache_dir}")
        
        # 2. Tạo thư mục đích trên SSD nếu chưa có
        dest_path = Path(DEST_DIR)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # 3. Đồng bộ từng file vào SSD
        print(f"🔄 Đang đồng bộ vào SSD: {DEST_DIR}...")
        files_updated = 0
        for file_name in PATTERNS:
            src = os.path.join(cache_dir, file_name)
            dst = os.path.join(DEST_DIR, file_name)
            
            if os.path.exists(src):
                # Chỉ copy nếu file chưa có hoặc có sự khác biệt (tối ưu hóa tốc độ)
                if not os.path.exists(dst) or os.path.getsize(src) != os.path.getsize(dst):
                    # Sử dụng shutil.copy2 để giữ nguyên thuộc tính file, đảm bảo tính toàn vẹn cho các trọng số AI (weights).
                    shutil.copy2(src, dst)
                    print(f"  + Đã cập nhật: {file_name}")
                    files_updated += 1
                else:
                    print(f"  - Đã trùng khớp (bỏ qua): {file_name}")
        
        if files_updated > 0:
            print(f"✨ Thành công! Đã cập nhật {files_updated} file mới vào SSD.")
        else:
            print("✨ Tuyệt vời! Dữ liệu trên SSD của bạn đã là bản mới nhất.")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình đồng bộ: {e}")

if __name__ == "__main__":
    sync_model()
