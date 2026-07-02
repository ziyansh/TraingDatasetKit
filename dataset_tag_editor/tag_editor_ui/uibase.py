"""
UI 基类（单例模式）

适配说明：
- 移除 scripts.singleton 依赖，使用 dataset_tag_editor.singleton
"""
from ..singleton import Singleton


class UIBase(Singleton):
    def create_ui(self, *args, **kwargs):
        raise NotImplementedError()
    
    def set_callbacks(self, *args, **kwargs):
        raise NotImplementedError()
    
    def func_to_set_value(self, name, type=None):
        def func(value):
            if type is not None:
                value = type(value)
            setattr(self, name, value)
            return value
        return func