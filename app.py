from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STRESS_MODEL_PATH = BASE_DIR / 'lam_model.h5'
EMOTION_MODEL_PATH = BASE_DIR / 'model_weights.pth'
TEMPLATE_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

# -----------------------------------------------------------------------------
# Configurable label maps
# If your training label order is different, only change these two lists.
# -----------------------------------------------------------------------------
STRESS_LABELS = ['Normal', 'Stress level 1', 'Stress level 2']
EMOTION_LABELS = ['Happy', 'Neutral', 'Sad']

# 20 survey items for frontend rendering
FEATURE_CONFIG: list[dict[str, Any]] = [
    {'name': 'q1', 'label': 'Câu 1', 'question': 'Trong 1 tuần gần đây, bạn có cảm thấy áp lực vì bài tập hoặc kiểm tra không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q2', 'label': 'Câu 2', 'question': 'Bạn có thấy khó tập trung khi học không?', 'type': 'range', 'min': 0, 'max': 4, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q3', 'label': 'Câu 3', 'question': 'Bạn có cảm thấy mệt mỏi dù không làm việc nặng không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q4', 'label': 'Câu 4', 'question': 'Bạn có khó ngủ hoặc ngủ không ngon không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q5', 'label': 'Câu 5', 'question': 'Bạn có thấy lo lắng trước các kỳ thi hoặc bài thuyết trình không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q6', 'label': 'Câu 6', 'question': 'Bạn có thấy chán học hoặc mất hứng thú học tập không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q7', 'label': 'Câu 7', 'question': 'Bạn có cảm thấy mình đang phải cố gắng quá sức để theo kịp việc học không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q8', 'label': 'Câu 8', 'question': 'Bạn có hay suy nghĩ tiêu cực về kết quả học tập của mình không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q9', 'label': 'Câu 9', 'question': 'Bạn có cảm thấy áp lực từ kỳ vọng của gia đình hoặc thầy cô không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q10', 'label': 'Câu 10', 'question': 'Bạn có thấy khó cân bằng giữa học tập và nghỉ ngơi không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q11', 'label': 'Câu 11', 'question': 'Bạn có thấy cơ thể căng cứng, đau đầu hoặc hồi hộp khi học không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q12', 'label': 'Câu 12', 'question': 'Bạn có dễ cáu gắt hoặc khó chịu trong thời gian gần đây không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q13', 'label': 'Câu 13', 'question': 'Bạn có thấy mình học lâu nhưng hiệu quả không cao không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q14', 'label': 'Câu 14', 'question': 'Bạn có cảm thấy sợ mắc lỗi hoặc sợ bị đánh giá kém không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q15', 'label': 'Câu 15', 'question': 'Bạn có hay trì hoãn việc học vì cảm thấy quá tải không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q16', 'label': 'Câu 16', 'question': 'Bạn có cảm thấy khó thư giãn ngay cả khi đã nghỉ không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q17', 'label': 'Câu 17', 'question': 'Bạn có thấy khối lượng việc học mỗi ngày là quá nhiều không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q18', 'label': 'Câu 18', 'question': 'Bạn có cảm thấy áp lực khi so sánh bản thân với bạn bè không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q19', 'label': 'Câu 19', 'question': 'Bạn có thấy dễ nản khi gặp bài khó hoặc điểm chưa tốt không?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Không', 'right_label': 'Rất nhiều'},
    {'name': 'q20', 'label': 'Câu 20', 'question': 'Nhìn chung, bạn đánh giá mức căng thẳng học tập hiện tại của mình như thế nào?', 'type': 'range', 'min': 0, 'max': 5, 'step': 1, 'left_label': 'Rất thấp', 'right_label': 'Rất cao'},
]

# -----------------------------------------------------------------------------
# Stress model loaded from Keras .h5 without needing TensorFlow runtime
# -----------------------------------------------------------------------------
class H5DenseSurveyModel:
    def __init__(self, path: Path):
        self.path = path
        self.weights, self.biases = self._load_parameters(path)

    @staticmethod
    def _load_parameters(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
        with h5py.File(path, 'r') as f:
            root = f['model_weights']
            dense_names = sorted([name for name in root.keys() if name.startswith('dense_')], key=lambda x: int(x.split('_')[1]))
            weights: list[np.ndarray] = []
            biases: list[np.ndarray] = []
            for name in dense_names:
                # keras 3 saved layout in this file: model_weights/dense_x/sequential_x/dense_x/{kernel,bias}
                layer_group = root[name]
                subgroups = list(layer_group.keys())
                if not subgroups:
                    continue
                seq_group = layer_group[subgroups[0]]
                final_group = seq_group[name]
                weights.append(np.array(final_group['kernel'], dtype=np.float32))
                biases.append(np.array(final_group['bias'], dtype=np.float32))
        if not weights:
            raise RuntimeError('Không đọc được trọng số từ file lam_model.h5')
        return weights, biases

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        output = x.astype(np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            output = output @ w + b
            if i < len(self.weights) - 1:
                output = np.maximum(output, 0.0)  # ReLU
        output = output - np.max(output, axis=1, keepdims=True)
        exp = np.exp(output)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs


# -----------------------------------------------------------------------------
# Emotion model - EfficientNet-B0 architecture compatible with torchvision
# This code expects a normal environment with torch + torchvision installed.
# -----------------------------------------------------------------------------
class EmotionModelWrapper:
    def __init__(self, path: Path):
        self.path = path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(path)
        self.available = self.model is not None

    def _load_model(self, path: Path) -> nn.Module | None:
        if not path.exists():
            return None
        try:
            # Imported lazily so the rest of the app still works even if torchvision
            # is missing or mismatched in the current environment.
            from torchvision import models  # type: ignore

            model = models.efficientnet_b0(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, 3),
            )
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model.to(self.device)
            return model
        except Exception as exc:  # pragma: no cover - safe fallback path
            print(f'[WARN] Không load được mô hình ảnh: {exc}')
            return None

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not self.available or self.model is None:
            raise RuntimeError(
                'Mô hình ảnh chưa sẵn sàng. Hãy cài torch + torchvision tương thích rồi chạy lại. '
                'Ví dụ: pip install torch torchvision'
            )

        image = image.convert('RGB').resize((224, 224))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return {
            'label': EMOTION_LABELS[idx],
            'confidence': round(float(probs[idx]) * 100, 2),
            'probs': {label: round(float(prob) * 100, 2) for label, prob in zip(EMOTION_LABELS, probs)}
        }


stress_model = H5DenseSurveyModel(STRESS_MODEL_PATH)
emotion_model = EmotionModelWrapper(EMOTION_MODEL_PATH)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def survey_recommendations(label: str) -> list[str]:
    if label == 'Normal':
        return [
            'Tiếp tục duy trì nhịp học hiện tại và nghỉ giải lao đều đặn.',
            'Ngủ đủ giấc và vận động nhẹ mỗi ngày.',
            'Theo dõi định kỳ để phát hiện sớm thay đổi bất thường.'
        ]
    if label == 'Stress level 1':
        return [
            'Giảm tải một phần các nhiệm vụ học tập trong ngày.',
            'Ưu tiên bài quan trọng trước, tránh dồn việc quá nhiều.',
            'Tăng thời gian nghỉ giữa các phiên học.'
        ]
    return [
        'Tạm thời giảm cường độ học và tránh thức khuya kéo dài.',
        'Trao đổi với phụ huynh, giáo viên hoặc người tin cậy.',
        'Nếu tình trạng kéo dài, nên tìm hỗ trợ chuyên môn phù hợp.'
    ]


def survey_analysis(label: str) -> tuple[str, str, str]:
    if label == 'Normal':
        return (
            'green', '😊',
            'Kết quả khảo sát cho thấy trạng thái hiện tại tương đối ổn định. Hệ thống chưa phát hiện dấu hiệu stress học đường rõ rệt.'
        )
    if label == 'Stress level 1':
        return (
            'yellow', '🙂',
            'Mô hình khảo sát ghi nhận một số dấu hiệu căng thẳng. Bạn nên điều chỉnh nhịp học và nghỉ ngơi sớm để tránh tăng áp lực.'
        )
    return (
        'red', '😟',
        'Mô hình khảo sát cho thấy mức áp lực cao. Bạn nên giảm tải việc học ngắn hạn và tìm hỗ trợ sớm nếu cảm giác căng thẳng kéo dài.'
    )


def emotion_analysis(label: str) -> str:
    if label == 'Happy':
        return 'Ảnh khuôn mặt cho thấy cảm xúc tích cực. Đây là tín hiệu thuận lợi khi kết hợp với kết quả khảo sát.'
    if label == 'Neutral':
        return 'Ảnh khuôn mặt thể hiện trạng thái trung tính. Cần kết hợp thêm với kết quả khảo sát để đánh giá đầy đủ hơn.'
    return 'Ảnh khuôn mặt cho thấy cảm xúc buồn hoặc thiếu năng lượng. Đây là tín hiệu cần chú ý khi kết hợp với kết quả khảo sát.'


def normalize_survey_input(payload: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    for item in FEATURE_CONFIG:
        name = item['name']
        raw = payload.get(name, 0)
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(0.0)
    arr = np.array(values, dtype=np.float32).reshape(1, -1)
    if arr.shape[1] != 20:
        raise ValueError('Mô hình stress cần đúng 20 đầu vào.')
    return arr


def combine_rule(survey_label: str, image_label: str) -> dict[str, Any]:
    if survey_label == 'Normal' and image_label in ('Happy', 'Neutral'):
        return {
            'level': 'Mức I – Ổn định',
            'meaning': 'Trạng thái ổn định',
            'emoji': '😊',
            'color': 'emerald',
            'explanation': 'Kết quả khảo sát bình thường và ảnh không cho thấy dấu hiệu cảm xúc tiêu cực rõ rệt.',
            'recommendations': [
                'Duy trì kế hoạch học hiện tại.',
                'Tiếp tục ngủ nghỉ hợp lý và vận động đều đặn.',
                'Theo dõi định kỳ để phát hiện sớm thay đổi.'
            ]
        }
    if survey_label == 'Normal' and image_label == 'Sad':
        return {
            'level': 'Mức II – Cần chú ý',
            'meaning': 'Cần chú ý cảm xúc',
            'emoji': '🙂',
            'color': 'yellow',
            'explanation': 'Khảo sát chưa thấy stress rõ ràng nhưng ảnh thể hiện cảm xúc buồn, nên theo dõi thêm.',
            'recommendations': [
                'Giảm tải nhẹ nếu đang mệt.',
                'Nghỉ giữa giờ và ngủ đúng giờ.',
                'Chia sẻ với người thân nếu cảm xúc buồn lặp lại.'
            ]
        }
    if ((survey_label == 'Stress level 1' and image_label in ('Happy', 'Neutral')) or
            (survey_label == 'Stress level 2' and image_label == 'Neutral')):
        return {
            'level': 'Mức III – Có stress',
            'meaning': 'Có stress, cần điều chỉnh',
            'emoji': '😐',
            'color': 'orange',
            'explanation': 'Khảo sát đã cho thấy áp lực đáng kể, dù cảm xúc ảnh chưa biểu lộ buồn rõ ràng.',
            'recommendations': [
                'Giảm số việc học nặng trong cùng một ngày.',
                'Ưu tiên môn yếu hoặc việc quan trọng trước.',
                'Theo dõi lại sau vài ngày.'
            ]
        }
    if survey_label in ('Stress level 1', 'Stress level 2') and image_label == 'Sad':
        return {
            'level': 'Mức IV – Đáng lo ngại',
            'meaning': 'Nguy cơ cao, cần hỗ trợ sớm',
            'emoji': '😟',
            'color': 'red',
            'explanation': 'Cả khảo sát và ảnh đều cho thấy tín hiệu đáng lo ngại, cần hỗ trợ sớm.',
            'recommendations': [
                'Giảm ngay cường độ học tập trong ngắn hạn.',
                'Trao đổi với phụ huynh, giáo viên hoặc cố vấn học tập.',
                'Nếu cảm giác buồn chán kéo dài, nên tìm hỗ trợ chuyên môn.'
            ]
        }
    return {
        'level': 'Chưa xác định',
        'meaning': 'Đầu vào chưa hợp lệ',
        'emoji': '❔',
        'color': 'gray',
        'explanation': 'Cặp nhãn đầu vào chưa nằm trong bảng luật kết hợp.',
        'recommendations': ['Kiểm tra lại nhãn từ mô hình khảo sát và mô hình ảnh.']
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get('/')
def home():
    return render_template('index.html', feature_config=FEATURE_CONFIG, model_accuracy=90)


@app.post('/predict_stress')
def predict_stress():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        x = normalize_survey_input(payload)
        probs = stress_model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        label = STRESS_LABELS[idx]
        color, icon, analysis = survey_analysis(label)
        return jsonify({
            'ok': True,
            'raw_label': label,
            'level': label,
            'confidence': round(float(probs[idx]) * 100, 2),
            'analysis': analysis,
            'recommendations': survey_recommendations(label),
            'icon': icon,
            'color': color,
            'probabilities': {name: round(float(prob) * 100, 2) for name, prob in zip(STRESS_LABELS, probs)}
        })
    except Exception as exc:
        return jsonify({'ok': False, 'message': f'Lỗi dự đoán stress: {exc}'}), 400


@app.post('/predict_emotion')
def predict_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({'ok': False, 'message': 'Không tìm thấy file ảnh trong request.'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'ok': False, 'message': 'Bạn chưa chọn ảnh.'}), 400
        image = Image.open(io.BytesIO(file.read()))
        result = emotion_model.predict(image)
        return jsonify({
            'ok': True,
            'raw_label': result['label'],
            'label': result['label'],
            'confidence': result['confidence'],
            'analysis': emotion_analysis(result['label']),
            'probabilities': result['probs']
        })
    except Exception as exc:
        return jsonify({'ok': False, 'message': f'Lỗi dự đoán cảm xúc: {exc}'}), 400


@app.post('/predict_combined')
def predict_combined():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        survey_label = str(payload.get('survey_label', '')).strip()
        image_label = str(payload.get('image_label', '')).strip()
        if not survey_label or not image_label:
            return jsonify({'ok': False, 'message': 'Cần đủ survey_label và image_label.'}), 400
        data = combine_rule(survey_label, image_label)
        return jsonify({'ok': True, **data})
    except Exception as exc:
        return jsonify({'ok': False, 'message': f'Lỗi kết hợp kết quả: {exc}'}), 400


@app.get('/health')
def health():
    return jsonify({
        'ok': True,
        'stress_model_loaded': STRESS_MODEL_PATH.exists(),
        'emotion_model_loaded': EMOTION_MODEL_PATH.exists(),
        'emotion_runtime_ready': emotion_model.available,
        'stress_labels': STRESS_LABELS,
        'emotion_labels': EMOTION_LABELS,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
