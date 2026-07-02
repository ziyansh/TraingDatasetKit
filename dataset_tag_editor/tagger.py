"""
Tagger 模块（dataset-editor 集成用）

适配说明：
- 移除 modules.devices / modules.shared / modules.deepbooru / modules.images 依赖
- 仅保留 WaifuDiffusion (ONNX)，移除 DeepDanbooru
- 使用纯本地模型路径

实例：
```python
from dataset_tag_editor.tagger import WaifuDiffusion
tagger = WaifuDiffusion("wd-v1-4-vit-tagger-v2", "E:/models/wd-tagger/wd-v1-4-vit-tagger-v2", 0.35)
```
"""
from PIL import Image
import re
from typing import Optional, Dict

from .interrogator import Interrogator
from .interrogators.waifu_diffusion_tagger import WaifuDiffusionTagger


# tag 转义模式（替代 webui 的 re_special）
tag_escape_pattern = re.compile(r'([\\()])')


def get_replaced_tag(tag: str, use_spaces=True, use_escape=False):
    """格式化 tag（替代 webui 的 deepbooru_use_spaces / deepbooru_escape 设置）"""
    if use_spaces:
        tag = tag.replace('_', ' ')
    if use_escape:
        tag = re.sub(tag_escape_pattern, r'\\\1', tag)
    return tag


class Tagger(Interrogator):
    def start(self):
        pass
    def stop(self):
        pass
    def predict(self, image: Image.Image, threshold: Optional[float]):
        raise NotImplementedError()
    def name(self):
        raise NotImplementedError()


class WaifuDiffusion(Tagger):
    """WaifuDiffusion 模型封装（for dataset-tag-editor）"""
    def __init__(self, name: str, model_dir: str, threshold: float):
        self.repo_name = name
        self.model_dir = model_dir
        self.tagger_inst = WaifuDiffusionTagger(model_dir)
        self.threshold = threshold

    def start(self):
        self.tagger_inst.load()
        return self

    def stop(self):
        self.tagger_inst.unload()

    def predict(self, image: Image.Image, threshold: Optional[float] = None):
        labels = self.tagger_inst.apply(image)
        
        if threshold is not None:
            if threshold < 0:
                threshold = self.threshold
            probability_dict = dict([(get_replaced_tag(x[0]), x[1]) for x in labels[4:] if x[1] > threshold])
        else:
            probability_dict = dict([(get_replaced_tag(x[0]), x[1]) for x in labels[4:]])

        return probability_dict

    def name(self):
        return self.repo_name