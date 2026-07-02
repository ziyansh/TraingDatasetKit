"""
WD14 Tagger 核心推理模块

适配说明：
- 移除 stable-diffusion-webui 的 modules.shared / modules.deepbooru 依赖
- 移除 huggingface_hub 自动下载（改为纯本地路径）
- 使用 torch 检测 CUDA 设备替代 webui 的设备管理
- 使用内置的 tag_escape_pattern 替代 modules.deepbooru.re_special

实例：
```python
from tagger.interrogator import WaifuDiffusionInterrogator

interrogator = WaifuDiffusionInterrogator(
    "wd14-vit-v2",
    model_path="E:/models/wd-tagger/wd-v1-4-vit-tagger-v2/model.onnx",
    tags_path="E:/models/wd-tagger/wd-v1-4-vit-tagger-v2/selected_tags.csv"
)
interrogator.load()
ratings, tags = interrogator.interrogate(image)
```
"""
import os
import gc
import pandas as pd
import numpy as np
import re

from typing import Tuple, List, Dict
from io import BytesIO
from PIL import Image
from pathlib import Path

from . import dbimutils

# 用 torch 检测 CUDA 设备（替代 webui 的 modules.shared.cmd_opts）
tag_escape_pattern = re.compile(r'([\\()])')

def get_tf_device_name() -> str:
    """获取 TensorFlow 设备名称"""
    import torch
    if torch.cuda.is_available():
        return '/gpu:0'
    return '/cpu:0'


class Interrogator:
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],

        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False
    ) -> Dict[str, float]:
        for t in additional_tags:
            tags[t] = 1.0

        tags = {
            t: c

            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            if (
                c >= threshold
                and t not in exclude_tags
            )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags

        return unloaded

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()


class WaifuDiffusionInterrogator(Interrogator):
    """
    WaifuDiffusion ONNX 推理器
    
    实例：
    ```python
    interrogator = WaifuDiffusionInterrogator(
        "wd14-vit-v2",
        model_path="E:/models/wd-tagger/model.onnx",
        tags_path="E:/models/wd-tagger/selected_tags.csv"
    )
    ```
    """
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
    ) -> None:
        super().__init__(name)
        self.model_path = Path(model_path) if model_path else Path('model.onnx')
        self.tags_path = Path(tags_path) if tags_path else Path('selected_tags.csv')

    def check_model_exist(self):
        return self.model_path.exists() and self.tags_path.exists()

    def load(self) -> None:
        if not self.check_model_exist():
            raise FileNotFoundError(
                f"Model or tags file not found.\n"
                f"  Model: {self.model_path}\n"
                f"  Tags: {self.tags_path}\n"
                "Please set the correct model path in configuration."
            )

        from onnxruntime import InferenceSession

        import torch
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.model = InferenceSession(str(self.model_path), providers=providers)

        print(f'Loaded {self.name} model from {self.model_path}')

        self.tags = pd.read_csv(str(self.tags_path))

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags