import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from ultralytics import YOLO
import google.generativeai as genai

app = FastAPI(title="Braille Translation API")

# Cấu hình CORS để cho phép Frontend (từ Vercel) gọi sang API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lấy API Key từ biến môi trường của hệ thống
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel('gemini-3.5-flash')

# Tải mô hình YOLO khi ứng dụng bắt đầu
MODEL_PATH = './weights/yolov8_braille.pt'
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Lỗi tải mô hình YOLO: {e}")

# --- BẮT ĐẦU TỪ ĐIỂN VÀ HÀM LOGIC DỊCH (GIỮ NGUYÊN TỪ CODE CỦA BẠN) ---
braille_to_vietnamese = {
    '100000': 'a', '001110': 'ă', '100001': 'â', '110000': 'b', '100100': 'c',
    '100110': 'd', '011101': 'đ', '100010': 'e', '110001': 'ê', '110100': 'f',
    '110110': 'g', '110010': 'h', '010100': 'i', '010110': 'j', '101000': 'k',
    '111000': 'l', '101100': 'm', '101110': 'n', '101010': 'o', '100111': 'ô',
    '010101': 'ơ', '111100': 'p', '111110': 'q', '111010': 'r', '011100': 's',
    '011110': 't', '101001': 'u', '110011': 'ư', '111001': 'v', '010111': 'w',
    '101101': 'x', '101111': 'y', '101011': 'z', '110101': 'r',
    '001010': '´', '000011': '`', '010001': 'ˀ', '001001': '~', '000001': '.',
    '001111': 'NUMBER_SIGN', '000101': 'UPPER_CASE', '000000': ' ',
    '010000': ',', '011000': ';', '010010': ':', '010011': '.',
    '111101': 'y', '111111': 'WRONG_CHAR'
}
braille_to_numbers = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'}
tone_map = {
    ('´', 'a'): 'á', ('`', 'a'): 'à', ('ˀ', 'a'): 'ả', ('~', 'a'): 'ã', ('.', 'a'): 'ạ',
    ('´', 'ă'): 'ắ', ('`', 'ă'): 'ằ', ('ˀ', 'ă'): 'ẳ', ('~', 'ă'): 'ẵ', ('.', 'ă'): 'ặ',
    ('´', 'â'): 'ấ', ('`', 'â'): 'ầ', ('ˀ', 'â'): 'ẩ', ('~', 'â'): 'ẫ', ('.', 'â'): 'ậ',
    ('´', 'e'): 'é', ('`', 'e'): 'è', ('ˀ', 'e'): 'ẻ', ('~', 'e'): 'ẽ', ('.', 'e'): 'ẹ',
    ('´', 'ê'): 'ế', ('`', 'ê'): 'ề', ('ˀ', 'ê'): 'ể', ('~', 'ê'): 'ễ', ('.', 'ê'): 'ệ',
    ('´', 'i'): 'í', ('`', 'i'): 'ì', ('ˀ', 'i'): 'ỉ', ('~', 'i'): 'ĩ', ('.', 'i'): 'ị',
    ('´', 'o'): 'ó', ('`', 'o'): 'ò', ('ˀ', 'o'): 'ỏ', ('~', 'o'): 'õ', ('.', 'o'): 'ọ',
    ('´', 'ô'): 'ố', ('`', 'ô'): 'ồ', ('ˀ', 'ô'): 'ổ', ('~', 'ô'): 'ỗ', ('.', 'ô'): 'ộ',
    ('´', 'ơ'): 'ớ', ('`', 'ơ'): 'ờ', ('ˀ', 'ơ'): 'ở', ('~', 'ơ'): 'ỡ', ('.', 'ơ'): 'ợ',
    ('´', 'u'): 'ú', ('`', 'u'): 'ù', ('ˀ', 'u'): 'ủ', ('~', 'u'): 'ũ', ('.', 'u'): 'ụ',
    ('´', 'ư'): 'ứ', ('`', 'ư'): 'ừ', ('ˀ', 'ư'): 'ử', ('~', 'ư'): 'ữ', ('.', 'ư'): 'ự',
    ('´', 'y'): 'ý', ('`', 'y'): 'ỳ', ('ˀ', 'y'): 'ỷ', ('~', 'y'): 'ỹ', ('.', 'y'): 'ỵ',
}

def perform_translation(matrix):
    translated_result = []
    for row in matrix:
        line_text = ""
        is_number_mode = False
        pending_tone = None
        is_upper_mode = False
        for label in row:
            val = braille_to_vietnamese.get(label, "?")
            if val == 'WRONG_CHAR': continue
            if val == ' ' or label == '000000':
                is_number_mode = False; pending_tone = None; line_text += " "; continue
            if val == 'NUMBER_SIGN': is_number_mode = True; continue
            if val == 'UPPER_CASE': is_upper_mode = True; continue
            if val in ['´', '`', 'ˀ', '~', '.']: pending_tone = val; continue

            final_char = val
            if is_number_mode and val in braille_to_numbers:
                final_char = braille_to_numbers[val]
            elif not is_number_mode:
                if pending_tone and (pending_tone, val) in tone_map:
                    final_char = tone_map[(pending_tone, val)]
                    pending_tone = None
                if is_upper_mode:
                    final_char = final_char.upper(); is_upper_mode = False
            line_text += final_char
        translated_result.append(line_text)
    return "\n".join(translated_result)
# --- KẾT THÚC LOGIC DỊCH ---

# ENDPOINT API TẠO MỚI
@app.post("/api/translate")
async def translate_braille(files: List[UploadFile] = File(...)):
    if len(files) < 1 or len(files) > 5:
        raise HTTPException(status_code=400, detail="Vui lòng gửi từ 1 đến 5 file ảnh.")

    raw_texts = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model.predict(source=img, conf=0.15, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names

        detected = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            detected.append({"x": (x1+x2)/2, "y": (y1+y2)/2, "label": names[classes[i]]})

        detected.sort(key=lambda item: item["y"])
        y_thresh = 20
        lines = []
        curr_line = []
        if detected:
            for i, item in enumerate(detected):
                if i == 0: curr_line.append(item)
                else:
                    if abs(item["y"] - curr_line[-1]["y"]) < y_thresh: curr_line.append(item)
                    else:
                        lines.append(curr_line); curr_line = [item]
            if curr_line: lines.append(curr_line)

        matrix = []
        for line in lines:
            line.sort(key=lambda item: item["x"])
            matrix.append([item["label"] for item in line])

        raw_texts.append(perform_translation(matrix))

    # Gọi Gemini xử lý lại văn bản
    combined_raw = "\n---\n".join([f"Bản {i+1}: {t}" for i, t in enumerate(raw_texts)])
    prompt = f"""
    Dưới đây là {len(raw_texts)} bản dịch thô từ ảnh quét chữ nổi Braille của cùng một văn bản.
    Hãy đối chiếu, sửa các lỗi nhận diện sai dấu hoặc sai ký tự để tạo ra một văn bản tiếng Việt chuẩn nhất.

    Dữ liệu thô:
    {combined_raw}

    Chỉ trả về văn bản kết quả cuối cùng, không giải thích.
    """

    try:
        response = llm_model.generate_content(prompt)
        return {
            "status": "success",
            "translated_text": response.text.strip(),
            "raw_texts": raw_texts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi AI: {str(e)}")
