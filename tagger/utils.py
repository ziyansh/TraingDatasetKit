"""
Tagger 工具模块

适配说明：
- 完全重写：移除所有 webui modules.* 依赖
- 用 config.app_config 替换 modules.shared / modules.paths_internal
- 用本地的 WaifuDiffusionInterrogator 替换 webui 版本

实例：
```python
from tagger.utils import refresh_interrogators, interrogators
# 刷新可用的模型列表
models = refresh_interrogators()
# interrogators 是全局 dict，key 为模型名称
```
"""
import os

from typing import List, Dict
from pathlib import Path

from config import app_config
from tagger.interrogator import WaifuDiffusionInterrogator

# 全局 interrogator 实例字典
interrogators: Dict[str, WaifuDiffusionInterrogator] = {}


def refresh_interrogators() -> List[str]:
    """
    刷新 interrogator 列表，从配置的模型目录查找可用模型
    
    支持的模型名称：
    - wd14-convnextv2-v2
    - wd14-vit-v2
    - wd14-convnext-v2
    - wd14-swinv2-v2
    
    如果模型目录下存在对应子目录，则使用子目录中的模型文件。
    支持自定义模型名称，从 app_config.custom_model_names 读取。
    """
    global interrogators
    interrogators = {}
    
    model_dir = app_config.tagger_model_dir
    if not model_dir or not os.path.isdir(model_dir):
        print("[Tagger Utils] Model directory not configured or not found")
        return []
    
    custom_names = app_config.custom_model_names or {}
    
    # 扫描目录下所有包含 model.onnx 的子目录
    found_models = []
    
    for root, dirs, files in os.walk(model_dir):
        if 'model.onnx' in files:
            tags_path = Path(root, 'selected_tags.csv')
            if tags_path.exists():
                model_name = os.path.basename(root)
                
                # 获取自定义名称，如果没有则使用目录名 + "_custom"
                if root in custom_names and custom_names[root].strip():
                    display_name = custom_names[root].strip()
                else:
                    display_name = f"{model_name}_custom"
                
                # 避免名称冲突
                counter = 1
                final_name = display_name
                while final_name in interrogators:
                    final_name = f"{display_name}_{counter}"
                    counter += 1
                
                interrogators[final_name] = WaifuDiffusionInterrogator(
                    final_name,
                    model_path=str(Path(root, 'model.onnx')),
                    tags_path=str(tags_path)
                )
                found_models.append(final_name)
                print(f"[Tagger Utils] Found model: {final_name} at {root}")
    
    return sorted(interrogators.keys())


def split_str(s: str, separator=',') -> List[str]:
    """按分隔符拆分字符串，去除空白"""
    return [x.strip() for x in s.split(separator) if x]


def scan_models(directory: str) -> List[Dict[str, str]]:
    """
    扫描指定目录，查找所有包含 model.onnx 的子目录
    
    返回: [{name: str, path: str}, ...]
    """
    results = []
    
    if not directory or not os.path.isdir(directory):
        return results
    
    for root, dirs, files in os.walk(directory):
        if 'model.onnx' in files:
            tags_path = Path(root, 'selected_tags.csv')
            if tags_path.exists():
                model_name = os.path.basename(root)
                results.append({
                    'name': model_name,
                    'path': root
                })
    
    return sorted(results, key=lambda x: x['name'])