"""
Dataset Tag Editor 模块

适配说明：
- 移除所有 modules.shared / modules.devices / launch 等 webui 依赖
- 仅保留 WaifuDiffusion (ONNX) 模型，移除 DeepDanbooru / BLIP / GIT
- 使用 config.app_config 管理路径
"""
from . import dataset, filters, interrogators
from .dte_logic import DatasetTagEditor, interrogate_image
from .singleton import Singleton

# 可用的 interrogator 模型名称列表（用于 UI 下拉选择）
# 实例: ['wd14-convnextv2-v2', 'wd14-vit-v2', 'wd14-convnext-v2', 'wd14-swinv2-v2']
INTERROGATOR_NAMES = [
    'wd14-convnextv2-v2',
    'wd14-vit-v2',
    'wd14-convnext-v2',
    'wd14-swinv2-v2',
]

__all__ = [
    'dataset', 'filters', 'interrogators', 
    'DatasetTagEditor', 'Singleton', 
    'INTERROGATOR_NAMES', 'interrogate_image'
]