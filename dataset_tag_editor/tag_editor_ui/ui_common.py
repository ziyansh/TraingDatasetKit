"""
UI 公共导入模块

适配说明：
- 移除 scripts.dte_instance 依赖，使用本地的 dte_logic
"""
from ..dte_logic import DatasetTagEditor, dte_settings, interrogate_image
import dataset_tag_editor.dte_logic as dte_module

dte_instance = DatasetTagEditor.get_instance()