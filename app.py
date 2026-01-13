"""
Viterbox - Gradio Web Interface
"""
# --- CONFIGURATION SECTION ---
import torch
import os
import sys
import signal
import shutil
import numpy as np
import time
import librosa
from pathlib import Path
import warnings
import random
import gradio as gr
import tempfile
from viterbox import Viterbox
from viterbox.tts import vad_trim
from datetime import datetime
import config

from config import OFFLINE_MODE, REPO_ID, LOCAL_MODEL_PATH, USE_VAD, VAD_MIN_SILENCE_MS, VAD_MARGIN_MS, VAD_THRESHOLD, VAD_MIN_SPEECH_MS, TEXT_LINE, SAMPLES, WAVS_DIR, VAD_TOP_DB

# GET OPTIMAL DEVICE
'''
1. Ý nghĩa kỹ thuật
hasattr(object, "name"): Là một hàm của Python dùng để kiểm tra xem một đối tượng (object) có sở hữu một thuộc tính hoặc phương thức mang tên ("name") hay không.
torch.backends.mps: Đây là thành phần được thêm vào từ các phiên bản PyTorch mới (khoảng 1.12 trở đi) để làm việc với chip Apple Silicon (M1, M2, M3, M4).
2. Sự khác biệt giữa hasattr và is_available
Thông thường, trong code chúng ta kết hợp cả hai:
hasattr: Kiểm tra xem phần mềm (thư viện PyTorch) có biết "mps" là cái gì không.
is_available(): Kiểm tra xem phần cứng (Chip M4 của bạn) có thực sự hỗ trợ và sẵn sàng chạy nó không.
'''

# --- LICENSE CHECKING FUNCTION ---
def check_license():
    if config.EXPIRY_DATE is None:
        return True
        
    try:
        expiry = datetime.strptime(config.EXPIRY_DATE, "%Y-%m-%d")
        current_date = datetime.now()
        
        if current_date > expiry:
            print(f"\n" + "!"*50)
            print("❌ PHẦN MỀM ĐÃ HẾT HẠN DÙNG THỬ!")
            print(f"Ngày hết hạn: {config.EXPIRY_DATE}")
            print("Vui lòng liên hệ tác giả để gia hạn.")
            print("!"*50 + "\n")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Lỗi kiểm tra bản quyền: {e}")
        return False

# --- END LICENSE CHECKING FUNCTION ---

# Thực hiện kiểm tra ngay khi chạy script
if not check_license():
    sys.exit()


def get_optimal_device():
    # 1. Kiểm tra NVIDIA (Windows/Linux)
    if torch.cuda.is_available():
        return "cuda"
    
    # 2. Kiểm tra Apple Silicon (Mac M4 của bạn)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    # 3. Kiểm tra AMD (Chỉ dành cho Windows)
    try:
        import torch_directml # không cần quan tâm đến cảnh báo này
        # Nếu cài thư viện này, nó sẽ trả về thiết bị 'dml'
        return torch_directml.device()
    except (ImportError, AttributeError):
        # Nếu không có thư viện torch-directml, bỏ qua
        pass

    # 4. Mặc định dùng CPU
    return "cpu"
DEVICE = get_optimal_device()
#  END GETTING OPTIMAL DEVICE

if OFFLINE_MODE:
    print("📦 Running in OFFLINE MODE...")
    # Gọi hàm from_local vì nó nhận ckpt_dir trực tiếp
    MODEL = Viterbox.from_local(LOCAL_MODEL_PATH, DEVICE)
else:
    print("🌐 Checking for UPDATES from Internet...")
    try:
        # Chỉ truyền DEVICE vì hàm gốc của bạn chỉ nhận đúng cái này
        MODEL = Viterbox.from_pretrained(DEVICE,REPO_ID) 
        print("✅ System updated and loaded!")
    except Exception as e:
        print(f"❌ Update failed ({e}). Falling back to Local...")
        MODEL = Viterbox.from_local(LOCAL_MODEL_PATH, DEVICE)

# --- END CONFIGURATION SECTION --- 

warnings.filterwarnings('ignore')
# Thiết lập thư mục tạm thời cho Gradio
os.environ["GRADIO_TEMP_DIR"] = tempfile.gettempdir() + "/my_gradio_tmp"
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

# Load model
print("=" * 50)
print("🚀 Loading Local TTS System...")
print("=" * 50)
print(f"Device: {DEVICE}")


def exit_app():
    """Giải phóng bộ nhớ và thoát ứng dụng hoàn toàn"""
    print("🚀 Đang giải phóng bộ nhớ và thoát App...")
    # Gửi tín hiệu ngắt để đóng server Gradio và giải phóng MPS (GPU)
    os.kill(os.getpid(), signal.SIGINT)
    return "Đã đóng ứng dụng."


def list_voices() -> list[str]:
    """List available voice files"""
    # from config import WAVS_DIR
    wav_dir = Path(WAVS_DIR)
    if wav_dir.exists():
        return sorted([str(f) for f in wav_dir.glob("*.wav")])
    return []


def get_random_voice() -> str:
    """Get a random voice file from voice_samples folder"""
    voices = list_voices()
    if voices:
        return random.choice(voices)
    return None

def reset_vad_defaults():
    # Trả về các giá trị mặc định từ file config
    return (
        True,         # use_vad
        VAD_THRESHOLD,
        # 0.35,         # vad_threshold
        VAD_MARGIN_MS,
        # 15,           # vad_margin
        VAD_TOP_DB,
        # 30,           # vad_top_db
        "♻️ Đã khôi phục mặc định"
    )

def stop_generation():
    # Trong Gradio, nút STOP mặc định sẽ ngắt kết nối API
    return "🛑 Đã dừng tiến trình!"




def generate_speech(
    text: str,
    language: str,
    ref_audio,
    ref_dropdown,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    sentence_pause: float,
):
    """Generate speech from text - Optimized for Mac M4 (Jan 2026)"""
    if not text.strip():
        return None, "❌ Nhập vào văn bản để tạo giọng nói"
    
    # LOGIC ƯU TIÊN MỚI (Jan 2026):
    if ref_audio:
        ref_path = ref_audio      # 1. Nếu có file upload/ghi âm -> Dùng ngay
    elif ref_dropdown:
        ref_path = ref_dropdown   # 2. Nếu không upload nhưng có chọn dropdown -> Dùng dropdown
    else:
        ref_path = get_random_voice() # 3. Cuối cùng mới chọn ngẫu nhiên
    
    if not ref_path:
        return None, "❌ Không tìm thấy giọng mẫu nào!"
    
    try:
        import config
        from pathlib import Path
        
        # 1. Sinh Audio từ Model (Dạng Tensor)
        wav = MODEL.generate(
            text=text.strip(),
            language=language,
            audio_prompt=ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            sentence_pause_ms=int(sentence_pause * 1000),
        )

        # 2. LƯU FILE VÀO SSD (Thực hiện trước khi Trim để giữ bản gốc)
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = "audio"
        extension = ".wav"
        # file_path = output_dir / f"{base_name}{extension}"

        # 2.1 Logic kiểm tra trùng tên và thêm hậu tố -1, -2...
        # counter = 1
        # while file_path.exists():
        #     # Nếu file tồn tại, tạo tên mới: generated_speech-1.wav
        #     file_path = output_dir / f"{base_name}-{counter}{extension}"
        #     counter += 1
        counter = 0
        while True:
            suffix = f"-{counter}" if counter > 0 else ""
            original_filename = f"{base_name}{suffix}{extension}"
            trimmed_filename = f"trim-{base_name}{suffix}{extension}"
            
            # Kiểm tra: Chỉ dùng counter này nếu CẢ HAI tên file đều chưa tồn tại
            if not (output_dir / original_filename).exists() and \
               not (output_dir / trimmed_filename).exists():
                break
            counter += 1
        
        original_path = output_dir / original_filename
        trimmed_path = output_dir / trimmed_filename
        # 2.2 Lưu file
        MODEL.save_audio(wav, str(original_path))
        # 3. CHUYỂN SANG NUMPY VÀ ÉP KIỂU ĐỂ XỬ LÝ (An toàn cho 2026)
        audio_np = wav[0].cpu().numpy().astype(np.float32)

        # 4. LOGIC LỌC ÂM NÂNG CAO (Đã đưa lên trên lệnh return)
        if config.USE_VAD:
            # Sử dụng VAD thông minh dựa trên Sliders UI
            audio_np = vad_trim(audio_np, MODEL.sr)
            vad_status_msg = "VAD ON"
        else:
            # Lọc năng lượng cơ bản theo top_db từ UI
            import librosa
            audio_np, _ = librosa.effects.trim(audio_np, top_db=config.VAD_TOP_DB)
            vad_status_msg = f"VAD OFF (top_db={config.VAD_TOP_DB})"
        
        # 5. LƯU FILE ĐÃ TRIM (Sử dụng soundfile vì đã là numpy)
        import soundfile as sf
        sf.write(str(trimmed_path), audio_np, MODEL.sr)
        duration = len(audio_np) / MODEL.sr
        status = f"✅ {vad_status_msg} |  Đã lưu: {original_filename} & {trimmed_filename} | {duration:.2f}s"
        status = f"✅ Xong! | Giọng mẫu đã sử dụng: {ref_path} | {duration:.2f}s"
        return (MODEL.sr, audio_np), status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"




# # CSS
# CSS = """
# body, .gradio-container { background: #0f172a !important; }
# .gradio-container { max-width: 100% !important; padding: 1rem 2rem !important; }
# .status-badge { 
#     display: inline-flex; align-items: center; padding: 4px 12px;
#     border-radius: 999px; font-size: 0.8rem; font-weight: 500;
#     background: #4f46e5; color: #fff;
# }
# #main-row { gap: 1rem !important; }
# #main-row > div { flex: 1 !important; min-width: 0 !important; }
# .card { 
#     background: #1e293b !important; border-radius: 0.75rem;
#     border: 1px solid #334155 !important; padding: 1rem 1.25rem; height: 100%;
# }
# .section-title { 
#     font-size: 0.85rem; font-weight: 600; color: #e5e7eb;
#     margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;
# }
# .generate-btn { 
#     background: #88e072 !important; border-radius: 0.5rem !important;
#     font-size: 1rem !important; padding: 10px 24px !important; margin-top: 0.75rem !important;
# }
# .output-card { 
#     background: #1e293b !important; border-radius: 0.75rem;
#     border: 1px solid #334155 !important; padding: 1rem 1.25rem; margin-top: 0.75rem;
# }
# """

# # CSS MỚI

# # 1. Khai báo các hằng số thiết kế (Design Tokens)
# # Bạn chỉ cần chỉnh sửa các giá trị ở đây để thay đổi toàn bộ giao diện
# UI_CONFIG = {
#     "primary": "#6366f1",          # Tím Indigo hiện đại
#     "secondary": "#22c55e",        # Xanh lá Emerald (cho nút bấm)
#     "bg_main": "#0f172a",          # Nền tối sâu (Slate 950)
#     "bg_card": "#1e293b",          # Nền thẻ (Slate 800)
#     "border_color": "#334155",     # Màu viền (Slate 700)
#     "text_main": "#f8fafc",        # Chữ trắng xám
#     "text_muted": "#94a3b8",       # Chữ xám nhạt cho tiêu đề phụ
#     "radius_lg": "1rem",           # Bo góc lớn cho Card
#     "radius_md": "0.75rem",        # Bo góc vừa cho Button
#     "font_main": "'Inter', system-ui, -apple-system, sans-serif"
# }

# # 2. Sử dụng f-string để truyền các hằng số vào chuỗi CSS
# CSS = f"""
# /* Tổng thể giao diện */
# body, .gradio-container {{ 
#     background: {UI_CONFIG['bg_main']} !important; 
#     font-family: {UI_CONFIG['font_main']};
# }}

# .gradio-container {{ 
#     max-width: 1200px !important; 
#     padding: 1.5rem 2rem !important; 
# }}

# /* Badge trạng thái - Phong cách tối giản */
# .status-badge {{ 
#     display: inline-flex; align-items: center; 
#     padding: 6px 14px; border-radius: 999px;
#     font-size: 0.75rem; font-weight: 600;
#     background: rgba(99, 102, 241, 0.15);
#     color: {UI_CONFIG['primary']};
#     border: 1px solid {UI_CONFIG['primary']};
# }}

# /* Cấu trúc Layout */
# #main-row {{ gap: 1.5rem !important; }}

# /* Thẻ Card nội dung */
# .card, .output-card {{ 
#     background: {UI_CONFIG['bg_card']} !important; 
#     border-radius: {UI_CONFIG['radius_lg']};
#     border: 1px solid {UI_CONFIG['border_color']} !important; 
#     padding: 1.5rem; 
#     transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
# }}

# .card:hover {{ 
#     border-color: {UI_CONFIG['primary']} !important;
#     box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
# }}

# /* Tiêu đề mục - Chữ in hoa nhỏ hiện đại */
# .section-title {{ 
#     font-size: 0.8rem; font-weight: 700; color: {UI_CONFIG['text_muted']};
#     text-transform: uppercase; letter-spacing: 0.05em;
#     margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;
# }}

# /* Nút Generate - Hiệu ứng Gradient & Phóng to khi Hover */
# .generate-btn {{ 
#     background: linear-gradient(135deg, {UI_CONFIG['secondary']}, #16a34a) !important; 
#     color: white !important;
#     border-radius: {UI_CONFIG['radius_md']} !important;
#     font-size: 1rem !important; font-weight: 700 !important;
#     padding: 12px 24px !important; margin-top: 1rem !important;
#     border: none !important; cursor: pointer;
#     transition: transform 0.2s, box-shadow 0.2s !important;
# }}

# .generate-btn:hover {{ 
#     transform: translateY(-2px);
#     box-shadow: 0 8px 20px rgba(34, 197, 94, 0.4) !important;
# }}

# .generate-btn:active {{ transform: scale(0.98); }}

# /* Card kết quả - Có đường kẻ nhấn mạnh bên trái */
# .output-card {{ 
#     margin-top: 1rem;
#     border-left: 4px solid {UI_CONFIG['primary']} !important;
# }}
# """

# # END CSS MỚI


# CSS MỚI -1
# Cập nhật bộ màu phong cách Futuristic 2026
UI_CONFIG = {
    # Màu chủ đạo: Chuyển sang Electric Cyan để tạo cảm giác công nghệ cao
    "primary": "#06b6d4",          
    
    # Màu hành động: Chuyển sang Soft Violet/Pink để tạo sự tương phản mạnh với Cyan
    "secondary": "#a855f7",        
    
    # Nền tổng thể: Sử dụng màu Midnight Blue cực sâu (giúp mắt thư giãn hơn Slate)
    "bg_main": "#020617",          
    
    # Nền thẻ: Hiệu ứng kính mờ (Glassmorphism) nhẹ
    "bg_card": "#0f172a",          
    
    # Màu viền: Sử dụng màu trung tính nhưng có độ sáng cao hơn để tách biệt card
    "border_color": "#1e293b",     
    
    # Văn bản: Sử dụng trắng tinh khiết cho nội dung chính và bạc cho nội dung phụ
    "text_main": "#ffffff",        
    "text_muted": "#64748b",       
    
    # Hình khối: Bo góc lớn hơn (Soft UI) đang là xu hướng 2026
    "radius_lg": "1.25rem",        
    "radius_md": "0.85rem",
    "radius_sm": "0.5rem",
    
    # Font chữ: Ưu tiên font Geometric Sans-Serif hiện đại
    "font_main": "'Plus Jakarta Sans', 'Inter', sans-serif"
}

# CSS với hiệu ứng ánh sáng (Glow Effect) cho năm 2026
CSS = f"""
body, .gradio-container {{ 
    background: radial-gradient(circle at 50% 0%, #1e293b 0%, {UI_CONFIG['bg_main']} 100%) !important;
    font-family: {UI_CONFIG['font_main']};
}}

/* Card với hiệu ứng viền phát sáng nhẹ khi hover */
.card {{ 
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 200px; /* Điều chỉnh chiều cao tối thiểu bạn muốn */
    background: {UI_CONFIG['bg_card']} !important;
    border: 1px solid {UI_CONFIG['border_color']} !important;
    border-radius: {UI_CONFIG['radius_md']};
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}}

.card:hover {{
    border-color: {UI_CONFIG['primary']}66 !important; /* Thêm 66 để tạo độ trong suốt cho hex */
    box-shadow: 0 0 20px {UI_CONFIG['primary']}22;
}}

.vad-card {{ 
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 10px; /* Điều chỉnh chiều cao tối thiểu bạn muốn */
    background: {UI_CONFIG['bg_card']} !important;
    border: 1px solid {UI_CONFIG['border_color']} !important;
    border-radius: {UI_CONFIG['radius_md']};
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}}

.vad-card:hover {{
    border-color: {UI_CONFIG['primary']}66 !important; /* Thêm 66 để tạo độ trong suốt cho hex */
    box-shadow: 0 0 20px {UI_CONFIG['primary']}22;
}}

.output-card {{ 
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 10px; /* Điều chỉnh chiều cao tối thiểu bạn muốn */
    background: {UI_CONFIG['bg_card']} !important;
    border: 1px solid {UI_CONFIG['border_color']} !important;
    border-radius: {UI_CONFIG['radius_md']};
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}}

.output-card:hover {{
    border-color: {UI_CONFIG['primary']}66 !important; /* Thêm 66 để tạo độ trong suốt cho hex */
    box-shadow: 0 0 20px {UI_CONFIG['primary']}22;
}}

/* Ép Textbox lấp đầy không gian trống còn lại trong Card */
.card > .gr-form {{
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}}

.card textarea {{
    flex-grow: 1 !important;
    height: 100% !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.6 !important;
    resize: none !important; /* Tắt nút kéo giãn thủ công cho đẹp */
}}

/* Nút bấm với hiệu ứng Gradient đa sắc */
.generate-btn {{
    background: linear-gradient(135deg, {UI_CONFIG['primary']} 0%, {UI_CONFIG['secondary']} 100%) !important;
    color: white !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: none !important;
    border-radius: {UI_CONFIG['radius_sm']} !important;
    transition: all 0.4s ease !important;
}}

.generate-btn:hover {{
    filter: brightness(1.2);
    box-shadow: 0 0 25px {UI_CONFIG['primary']}66 !important;
    transform: translateY(-3px);
}}

/* Kiểu dáng cho 2 nút phụ (Random & Clear) */
.secondary-btn {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: {UI_CONFIG['primary']} !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}}

.secondary-btn:hover {{
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: {UI_CONFIG['primary']} !important;
    transform: translateY(-2px);
}}

#main-textbox {{
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important;
}}

#main-textbox > label {{
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}}

#main-textbox textarea {{
    flex-grow: 1 !important;
    height: 100% !important; /* Ép giãn theo container */
    min-height: 150px !important; /* Đảm bảo đủ cao để bằng cột bên cạnh */
}}

button.gr-button-variant-secondary {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
}}

button.gr-button-variant-secondary:hover {{
    background: #444 !important;
    color: #ff4b2b !important; /* Đổi màu chữ khi hover để cảnh báo thoát */
    border-color: #ff4b2b !important;
}}


#main-row {{ gap: 1rem !important; }}
#main-row > div {{ flex: 1 !important; min-width: 0 !important; }}


"""


UI_CUSTOM_CSS = f"""
/* Ép Column Card trở thành Flexbox container */
.card {{
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important; 
    min-height: 300px; /* Điều chỉnh chiều cao tối thiểu bạn muốn */
}}

/* Ép Textbox lấp đầy không gian trống còn lại trong Card */
.card > .gr-form {{
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}}

.card textarea {{
    flex-grow: 1 !important;
    height: 100% !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.6 !important;
    resize: none !important; /* Tắt nút kéo giãn thủ công cho đẹp */
}}

/* Kiểu dáng cho 2 nút phụ (Random & Clear) */
.secondary-btn {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: {UI_CONFIG['primary']} !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}}

.secondary-btn:hover {{
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: {UI_CONFIG['primary']} !important;
    transform: translateY(-2px);
}}
.secondary-card {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: {UI_CONFIG['primary']} !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}}

.secondary-card:hover {{
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: {UI_CONFIG['primary']} !important;
    transform: translateY(-2px);
}}
"""




# END CSS MỚI -1

# Build UI
# with gr.Blocks(
#     title="🎙️ Local TTS - Vietnamese Support - Donald 0986168163",
#     theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate"),
#     css=CSS
# ) as demo:
with gr.Blocks(title="🎙️ Local TTS - Vietnamese Support - Donald 0986168163", css=CSS) as demo:
    def get_license_info():
        if config.EXPIRY_DATE is None:
            return "♾️ Phiên bản vĩnh viễn"
        expiry = datetime.strptime(config.EXPIRY_DATE, "%Y-%m-%d")
        days_left = (expiry - datetime.now()).days
        if days_left > 0:
            return f"⏳ Còn {days_left} ngày (Hết hạn: {config.EXPIRY_DATE})"
        return "❌ Đã hết hạn"

    gr.Markdown(f"<center>{get_license_info()}</center>")

    # NEW CODE -2
    # CHÈN ĐOẠN NÀY VÀO DƯỚI gr.HTML
    with gr.Accordion("🛠️ Cấu hình Cắt/Lọc âm nâng cao (VAD)", elem_classes=["vad-card"], open=True): # Mặc định mở ra
        with gr.Row():
            use_vad = gr.Checkbox(label="Kích hoạt VAD", value=config.USE_VAD)
            vad_threshold = gr.Slider(0.1, 0.9, step=0.05, value=config.VAD_THRESHOLD, label="Độ nhạy")

            vad_margin = gr.Slider(0, 100, step=5, value=config.VAD_MARGIN_MS, label="Đệm đuôi")

            vad_top_db = gr.Slider(10, 60, step=1, value=config.VAD_TOP_DB, label="Ngưỡng cắt")
        with gr.Row():
            # Nút Reset
            # reset_btn = gr.Button("♻️ Reset VAD", size="md", variant="secondary", elem_classes=["secondary-btn"])
            reset_btn = gr.Button("♻️ Reset VAD")

        def update_params(u, t, m, db):
            config.USE_VAD = u
            config.VAD_THRESHOLD = t
            config.VAD_MARGIN_MS = m
            config.VAD_TOP_DB = db
            # Trả về Markdown để nhẹ hơn Textbox
            return f"⚙️ **Cấu hình:** VAD={'Bật' if u else 'Tắt'} | Sens={t} | Margin={m}ms | DB={db}"
        # ĐƯA DÒNG NÀY LÊN TRƯỚC VÒNG LẶP FOR
        vad_status = gr.Markdown(f"Trạng thái hiện tại: {'Bật' if config.USE_VAD else 'Tắt'}")
        # Kết nối sự kiện với chế độ tối ưu
        for ctrl in [use_vad, vad_threshold, vad_margin, vad_top_db]:
            ctrl.change(
                update_params, 
                inputs=[use_vad, vad_threshold, vad_margin, vad_top_db], 
                outputs=vad_status,
                show_progress="hidden" # Tắt xoay vòng load mỗi khi kéo slider
            )
        # Sự kiện Reset
        reset_btn.click(
            fn=reset_vad_defaults,
            outputs=[use_vad, vad_threshold, vad_margin, vad_top_db, vad_status]
        )
    # END NEW CODE -2

    with gr.Row(equal_height=True): # Tạo một hàng ngang
        # gr.HTML('<div class="section-title" style="margin-top: 0.75rem;">⚙️ Settings</div>')
        with gr.Column(scale=1, elem_classes=["secondary-card"]): # Cột 1
            exaggeration = gr.Slider(0.0, 2.0, 0.6, step=0.05, label="Exaggeration", info="Expression")
        with gr.Column(scale=1, elem_classes=["secondary-card"]): # Cột 2
            cfg_weight = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="CFG Weight", info="Voice adherence")
        with gr.Column(scale=1, elem_classes=["secondary-card"]): # Cột 3 
            temperature = gr.Slider(0.1, 1.0, 0.6, step=0.05, label="Temperature", info="Variation")
        with gr.Column(scale=1, elem_classes=["secondary-card"]): # Cột 4
            sentence_pause = gr.Slider(0.0, 2.0, 0.2, step=0.1, label="Sentence Pause (s)", info="Pause between sentences")

    with gr.Row(equal_height=True, elem_id="main-row"):
        # Left - Text Input
        with gr.Column(scale=1, elem_classes=["card"]):
            # gr.HTML('<div class="section-title">📝 Text Input</div>')
            language = gr.Radio(
                choices=[("🇻🇳 Tiếng Việt", "vi"), ("🇺🇸 English", "en")],
                value="vi", label="Language"
            )
            text_input = gr.Textbox(
                # label="Văn bản cần tạo voice",
                label = None,
                placeholder="Nhập văn bản hoặc click Sample Text để tạo voice...",
                lines=TEXT_LINE, #CSS sẽ ghi đè để giãn theo card
                elem_id="main-textbox"
            )
            
            with gr.Row():
                sample_btn = gr.Button(
                    "🔀 Sample Text", 
                    variant="secondary", 
                    # size="md", 
                    elem_classes=["secondary-btn"]
                )
                clear_btn = gr.Button(
                    "🧹 Clear Text", 
                    variant="secondary", 
                    # size="md", 
                    elem_classes=["secondary-btn"]
                )
        
        # Right - Voice & Settings
        with gr.Column(scale=1, elem_classes=["card"]):
            # gr.HTML('<div class="section-title">🎤 Select Voice to Clone</div>')
            
            wav_files = list_voices()
            if wav_files:
                ref_dropdown = gr.Dropdown(
                    choices=[(Path(f).stem, f) for f in wav_files],
                    label="Please select a voice from the list below",
                    value=wav_files[0] if wav_files else None,
                    # elem_classes=["dropdown"],
                )
            else:
                ref_dropdown = gr.Dropdown(choices=[], label="Không có giọng mẫu")
            
            ref_audio = gr.Audio(label="Or Upload/Record", type="filepath", sources=["upload", "microphone"])
    with gr.Row():        
        # Generate button
        # generate_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg", elem_classes=["generate-btn"])
        
        # --- PHẦN NÚT GENERATE & STOP/PAUSE ---
        generate_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg", elem_classes=["generate-btn"])
    
    with gr.Row():
        stop_btn = gr.Button("🛑 STOP", variant="stop", scale=1)
        exit_btn = gr.Button("🚪 EXIT APP", variant="secondary", scale=1)

    

    
    # Output
    with gr.Column(elem_classes=["output-card"]):
        gr.HTML('<div class="section-title">🔈 Output/LOG</div>')
        with gr.Row():
            output_audio = gr.Audio(label="Generated Speech", type="numpy", scale=2)
            status_text = gr.Textbox(label="Status", lines=2, scale=1)
    
    # Handlers
    sample_btn.click(
        fn=lambda lang: random.choice(SAMPLES.get(lang, SAMPLES["vi"])),
        inputs=[language],
        outputs=[text_input]
    )
    clear_btn.click(fn=lambda: "", outputs=[text_input])
    ref_dropdown.change(fn=lambda x: x, inputs=[ref_dropdown], outputs=[ref_audio])
    
    click_event=generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, language, ref_audio, ref_dropdown, exaggeration, cfg_weight, temperature, sentence_pause],
        outputs=[output_audio, status_text]
    )

    # # Kết nối sự kiện STOP (Gradio cung cấp cơ chế hủy tiến trình)
    # click_event = generate_btn.click(
    #     fn=generate_speech, 
    #     inputs=[
    #         text_input, 
    #         language, 
    #         ref_audio, # Nếu bạn có dùng ref audio
    #         exaggeration, 
    #         cfg_weight, 
    #         temperature, 
    #         sentence_pause
    #     ], 
    #     outputs=[output_audio, status_text] # Thay bằng tên biến audio và status của bạn
    # )
    
    
    # Sự kiện Generate (Sử dụng wrapper để khóa UI)
    # gen_event = generate_btn.click(
    #     fn=tts_wrapper,
    #     inputs=UI_COMPONENTS, # Truyền tất cả đầu vào
    #     outputs=[output_audio, status_text, generate_btn] + UI_COMPONENTS
    # )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event])
    # Nút STOP: Hủy tiến trình
    # stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

    # Nút EXIT: Thoát App
    exit_btn.click(fn=exit_app, inputs=None, outputs=None)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
