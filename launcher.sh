#!/bin/bash

# Cấu hình
PROJECT_DIR="/Volumes/SSD256/dev-projects/viterbox-tts"
PORT=7861
URL="http://localhost:$PORT"

cd "$PROJECT_DIR"

# Kiểm tra xem cổng 7860 có đang mở không (App đã chạy chưa?)
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "App đang chạy, tiến hành mở trình duyệt..."
    open "$URL"
else
    echo "App đang khởi động..."
    # Kích hoạt môi trường ảo
    source .venv/bin/activate
    
    # Khởi động app ở chế độ chạy ngầm (nohup) để không cần mở Terminal
    # Log sẽ được ghi vào file app_log.txt để bạn kiểm tra nếu cần
    # nohup python app.py > app_log.txt 2>&1 &
    python app.py > app_log.txt 2>&1 &
    
    # Đợi một chút để server kịp khởi động (Mac M4 rất nhanh, 3-5s là đủ)
    sleep 4
    
    # Mở trình duyệt
    open "$URL"
fi

