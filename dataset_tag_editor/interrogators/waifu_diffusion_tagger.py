"""
WaifuDiffusion Tagger 推理器（dataset-editor 版）

适配说明：
- 移除 modules.shared / launch / modules.images 依赖
- 使用 onnxruntime 直接推理
- 使用 OpenCV 替代 webui 的 images.resize_image
- 纯本地模型路径

实例：
```python
from dataset_tag_editor.interrogators.waifu_diffusion_tagger import WaifuDiffusionTagger
tagger = WaifuDiffusionTagger("E:/models/wd-tagger/wd-v1-4-vit-tagger-v2")
tagger.load()
labels = tagger.apply(image)
```
"""
from PIL import Image
import numpy as np
from typing import List, Tuple
from pathlib import Path
import cv2


class WaifuDiffusionTagger():
    """
    WaifuDiffusion ONNX 推理器（dataset-editor 集成用）
    
    参数 model_dir: 包含 model.onnx 和 selected_tags.csv 的本地目录
    """
    MODEL_FILENAME = "model.onnx"
    LABEL_FILENAME = "selected_tags.csv"
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model = None
        self.labels = []

    def load(self):
        model_path = self.model_dir / self.MODEL_FILENAME
        label_path = self.model_dir / self.LABEL_FILENAME
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Labels not found: {label_path}")
        
        import onnxruntime as ort
        import torch
        
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.model = ort.InferenceSession(str(model_path), providers=providers)
        
        import pandas as pd
        self.labels = pd.read_csv(str(label_path))["name"].tolist()
        print(f"[WaifuDiffusionTagger] Loaded {len(self.labels)} labels from {model_path}")

    def unload(self):
        self.model = None
        self.labels = []

    def apply(self, image: Image.Image):
        """对图片运行模型，返回 [(label, confidence), ...]"""
        if not self.model:
            return dict()
        
        _, height, width, _ = self.model.get_inputs()[0].shape

        # resize image to model input size
        image = image.convert("RGB")
        image_np = np.array(image, dtype=np.float32)
        
        # resize while maintaining aspect ratio with padding
        h, w = image_np.shape[:2]
        scale = min(width / w, height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # pad to target size
        canvas = np.full((height, width, 3), 255, dtype=np.float32)
        y_off = (height - new_h) // 2
        x_off = (width - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        # PIL RGB to OpenCV BGR
        image_np = canvas[:, :, ::-1]
        image_np = np.expand_dims(image_np, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image_np})[0]
        labels: List[Tuple[str, float]] = list(zip(self.labels, probs[0].astype(float)))

        return labels