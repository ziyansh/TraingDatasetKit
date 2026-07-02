"""
Captioning 模块（简化版）

适配说明：
- 移除 BLIP / GIT / DeepDanbooru 依赖
- 保留类层次结构以便 dte_logic 兼容引用
"""
from .interrogator import Interrogator


class Captioning(Interrogator):
    def start(self):
        pass
    def stop(self):
        pass
    def predict(self, image):
        raise NotImplementedError()
    def name(self):
        raise NotImplementedError()


# 注意：BLIP 和 GIT 在独立模式下不可用
# 因为需要 webui 的 shared.interrogator 或 transformers 的大模型
# 如需要可后续通过 transformers 库集成