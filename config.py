# config.py
from anyio import Path
import torch

OFFLINE_MODE = True # Chuyển thành False nếu muốn check update/tải model mới


# --- VAD SETTINGS ---
VAD_LOCAL_PATH = "/Volumes/SSD256/ai-models/TTS/Viterbox/VAD/silero_vad.jit"
USE_VAD = True  # Chuyển thành False nếu muốn tắt hoàn toàn bộ lọc VAD
VAD_THRESHOLD = 0.35              # Độ nhạy (0.1 -> 0.9). Càng thấp càng nhạy.
VAD_MIN_SPEECH_MS = 250           # Độ dài tối thiểu để coi là tiếng người.
VAD_MIN_SILENCE_MS = 100          # Khoảng lặng tối thiểu để bắt đầu cắt.
VAD_MARGIN_MS = 15                # Khoảng đệm (ms) sau khi nói để tránh mất âm đuôi.
VAD_TOP_DB = 30  # Ngưỡng decibel để coi là im lặng (mặc định 30)

# ---------------------

LOCAL_MODEL_DIR = "/Volumes/SSD256/ai-models/TTS/Viterbox/snapshots/6ddcc1430e2c1c67c5cf8e5c30b2c6952e6191db"
LOCAL_MODEL_PATH = "/Volumes/SSD256/ai-models/TTS/Viterbox/snapshots/6ddcc1430e2c1c67c5cf8e5c30b2c6952e6191db" 
REPO_ID = "dolly-vn/viterbox"  # Repo ID trên Hugging Face để update
WAVS_DIR = "voice_samples"
# -----------------------------
TEXT_LINE = 10 # Số dòng nhập văn bản trong giao diện app.py





#  MẪU THỬ GIỌNG VOICE

SAMPLES = {
    "vi": [
        # 1. Quảng cáo (Commercial - Tươi trẻ):
        "Một ngày mới khởi đầu bằng hương vị cà phê nguyên bản. Đậm đà, nồng nàn và đầy hứng khởi. Cà phê Mộc – không chỉ là thức uống, mà là lời đánh thức mọi giác quan. Thử một ngụm, bừng tỉnh đam mê. Mộc: Gói trọn tinh hoa từ cao nguyên.", 
        # 2. Truyền cảm hứng (Inspirational - Nhẹ nhàng):
        "Có những hành trình không đo bằng số km, mà bằng những kỷ niệm ta lưu giữ. Đừng ngại bước chậm lại để lắng nghe hơi thở của thiên nhiên. Vì hạnh phúc đôi khi chỉ là một sớm mai bình yên, thấy tâm hồn mình thực sự tự do.", 
        # 3. Phim điện ảnh (Movie Trailer - Kịch tính):
        "Trong thành phố không ngủ này, ranh giới giữa trắng và đen chưa bao giờ mong manh đến thế. Khi sự thật bị che lấp bởi những lời dối trá, liệu ai sẽ dám đứng ra bảo vệ công lý? Cuộc đấu trí nghẹt thở bắt đầu từ đây. Đừng tin bất kỳ ai!",
        # 4. Giới thiệu sản phẩm (Product Intro - Sang trọng):
        "Tinh tế trong từng đường nét, đẳng cấp trong mọi góc nhìn. Chiếc đồng hồ không chỉ xem giờ, mà là tuyên ngôn của sự thành công. Chế tác thủ công từ những vật liệu quý hiếm nhất. Khẳng định vị thế, dẫn lối tương lai cùng Aura Luxury.",
        # 5. Kể chuyện/Audiobook (Storytelling - Truyền cảm):
        "Đêm ấy, cơn mưa rào bất chợt phủ kín lối về. Bà ngồi bên hiên nhà, đôi mắt mờ đục nhìn xa xăm vào khoảng không tĩnh mịch. Tiếng radio cũ kỹ phát ra bản nhạc xưa, đưa ông bà trở lại những ngày tuổi trẻ đầy nắng và gió. Thời gian có thể trôi, nhưng tình yêu thì ở lại.",
        #6. Thời sự
        "Phòng CSĐT tội phạm về tham nhũng, kinh tế, buôn lậu, môi trường CATP Đà Nẵng đấu tranh thành công chuyên án, làm rõ các đối tượng vận chuyển trái phép 90.350 cá thể tôm hùm xanh giống, trị giá gần 6.8 tỷ đồng qua đường hàng không."
        
    ],
    "en": [
        # 1. Thương mại (Commercial - Năng động):
        "Step into the future of sound. With the all-new Sonic-X headphones, every beat is a heartbeat, and every note is a story. Wireless, weightless, and purely cinematic. Experience the sound that moves you. Sonic-X: Feel the pulse of the city.",
        # 2. Tường thuật (Narrative - Trầm ấm):
        "In the heart of the ancient forest, time stood still. The silver mist clung to the oaks like a forgotten secret. Here, every shadow whispered of legends past, waiting for a soul brave enough to listen. Adventure isn't calling—it’s waiting.",
        # 3. Thông báo (Announcement - Trang trọng):
        "Ladies and gentlemen, welcome aboard Flight 772 to London. Please ensure your seatbelts are securely fastened and all electronic devices are in airplane mode. Your comfort and safety are our highest priorities. Sit back, relax, and enjoy the journey.",
        # 4. Kịch tính (Dramatic - Mạnh mẽ):
        "They thought the war was over. They thought the silence meant peace. But in the darkness, a spark remained. One choice will change everything. One hero will rise. This summer, witness the end of the beginning. Justice has a new face.",
        # 5. Công nghệ (Tech/Corporate - Hiện đại):
        "Innovation isn't just about building gadgets; it's about bridging worlds. At Nexus Corp, we’re redefining connectivity with AI-driven solutions that think ahead. Faster. Smarter. Human-centric. Nexus: Creating the tomorrow you’ve always imagined.",
    ],
}

# --- LICENSE SETTINGS ---
# Định dạng: YYYY-MM-DD. Để None nếu muốn dùng vĩnh viễn (không giới hạn)
EXPIRY_DATE = "2026-03-01" 


# # 5 Mẫu Tiếng Anh (English Scripts)
# 1. Thương mại (Commercial - Năng động):
# "Step into the future of sound. With the all-new Sonic-X headphones, every beat is a heartbeat, and every note is a story. Wireless, weightless, and purely cinematic. Experience the sound that moves you. Sonic-X: Feel the pulse of the city."
# 2. Tường thuật (Narrative - Trầm ấm):
# "In the heart of the ancient forest, time stood still. The silver mist clung to the oaks like a forgotten secret. Here, every shadow whispered of legends past, waiting for a soul brave enough to listen. Adventure isn't calling—it’s waiting."
# 3. Thông báo (Announcement - Trang trọng):
# "Ladies and gentlemen, welcome aboard Flight 772 to London. Please ensure your seatbelts are securely fastened and all electronic devices are in airplane mode. Your comfort and safety are our highest priorities. Sit back, relax, and enjoy the journey."
# 4. Kịch tính (Dramatic - Mạnh mẽ):
# "They thought the war was over. They thought the silence meant peace. But in the darkness, a spark remained. One choice will change everything. One hero will rise. This summer, witness the end of the beginning. Justice has a new face."
# 5. Công nghệ (Tech/Corporate - Hiện đại):
# "Innovation isn't just about building gadgets; it's about bridging worlds. At Nexus Corp, we’re redefining connectivity with AI-driven solutions that think ahead. Faster. Smarter. Human-centric. Nexus: Creating the tomorrow you’ve always imagined."

# # 5 Mẫu Tiếng Việt (Vietnamese Scripts)
# 1. Quảng cáo (Commercial - Tươi trẻ):
# "Một ngày mới khởi đầu bằng hương vị cà phê nguyên bản. Đậm đà, nồng nàn và đầy hứng khởi. Cà phê Mộc – không chỉ là thức uống, mà là lời đánh thức mọi giác quan. Thử một ngụm, bừng tỉnh đam mê. Mộc: Gói trọn tinh hoa từ cao nguyên."
# 2. Truyền cảm hứng (Inspirational - Nhẹ nhàng):
# "Có những hành trình không đo bằng số km, mà bằng những kỷ niệm ta lưu giữ. Đừng ngại bước chậm lại để lắng nghe hơi thở của thiên nhiên. Vì hạnh phúc đôi khi chỉ là một sớm mai bình yên, thấy tâm hồn mình thực sự tự do."
# 3. Phim điện ảnh (Movie Trailer - Kịch tính):
# "Trong thành phố không ngủ này, ranh giới giữa trắng và đen chưa bao giờ mong manh đến thế. Khi sự thật bị che lấp bởi những lời dối trá, liệu ai sẽ dám đứng ra bảo vệ công lý? Cuộc đấu trí nghẹt thở bắt đầu từ đây. Đừng tin bất kỳ ai!"
# 4. Giới thiệu sản phẩm (Product Intro - Sang trọng):
# "Tinh tế trong từng đường nét, đẳng cấp trong mọi góc nhìn. Chiếc đồng hồ không chỉ xem giờ, mà là tuyên ngôn của sự thành công. Chế tác thủ công từ những vật liệu quý hiếm nhất. Khẳng định vị thế, dẫn lối tương lai cùng Aura Luxury."
# 5. Kể chuyện/Audiobook (Storytelling - Truyền cảm):
# "Đêm ấy, cơn mưa rào bất chợt phủ kín lối về. Bà ngồi bên hiên nhà, đôi mắt mờ đục nhìn xa xăm vào khoảng không tĩnh mịch. Tiếng radio cũ kỹ phát ra bản nhạc xưa, đưa ông bà trở lại những ngày tuổi trẻ đầy nắng và gió. Thời gian có thể trôi, nhưng tình yêu thì ở lại."