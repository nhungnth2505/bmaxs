# MAXA Render Minimal

## Deploy lên Render

### Cách 1: Deploy từ GitHub
1. Giải nén project này.
2. Đẩy toàn bộ file lên GitHub.
3. Vào Render -> New + -> Web Service.
4. Chọn repo GitHub.
5. Render sẽ tự đọc `render.yaml`.
6. Nhấn Deploy.

### Cách 2: Tạo Web Service thủ công
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`

## Chạy local
- on Mac:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
- on Window:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
## API
- `GET /health`
- `POST /predict_stress`
- `POST /predict_emotion`
- `POST /predict_combined`

## Lưu ý
- Nếu label mô hình thật khác thứ tự hiện tại, sửa trong `app.py`:
  - `STRESS_LABELS`
  - `EMOTION_LABELS`
- Render free có thể mất vài phút lần đầu vì cài `torch` và `torchvision`.
