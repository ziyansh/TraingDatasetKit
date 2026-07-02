"""
全局配置文件
管理 tagger 模型路径等全局配置项
"""
import json
import os
from pathlib import Path
from typing import Optional


class AppConfig:
    """
    应用全局配置
    
    实例：
    ```python
    config = AppConfig()
    config.tagger_model_dir = "E:/models/wd-tagger"
    config.tags_csv_path = "E:/models/wd-tagger/selected_tags.csv"
    ```
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # 默认配置文件路径
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'app_config.json'
        )
        
        # ---- Tagger 配置 ----
        # 模型目录路径，用户可指定
        self.tagger_model_dir: str = ""
        # 标签 CSV 文件路径（如果和模型在同一目录可留空）
        self.tags_csv_path: str = ""
        # 所使用的模型文件名
        self.model_filename: str = "model.onnx"
        # 标签文件名
        self.tags_filename: str = "selected_tags.csv"
        # 自定义模型名称映射 {model_path: custom_name}
        self.custom_model_names: dict = {}
        
        # ---- 通用配置 ----
        self.host: str = "127.0.0.1"
        self.port: int = 7000
        
        # 加载已有配置
        self.load()
    
    @property
    def resolved_model_path(self) -> str:
        """获取完整的 model.onnx 路径"""
        if self.tagger_model_dir:
            return os.path.join(self.tagger_model_dir, self.model_filename)
        return ""
    
    @property
    def resolved_tags_path(self) -> str:
        """获取完整的 selected_tags.csv 路径"""
        if self.tags_csv_path:
            return self.tags_csv_path
        if self.tagger_model_dir:
            return os.path.join(self.tagger_model_dir, self.tags_filename)
        return ""
    
    def save(self):
        """保存配置到 JSON 文件"""
        data = {
            "tagger_model_dir": self.tagger_model_dir,
            "tags_csv_path": self.tags_csv_path,
            "custom_model_names": self.custom_model_names,
            "host": self.host,
            "port": self.port,
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Config] Configuration saved to {self.config_path}")
    
    def load(self):
        """从 JSON 文件加载配置"""
        if os.path.isfile(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.tagger_model_dir = data.get("tagger_model_dir", "")
                self.tags_csv_path = data.get("tags_csv_path", "")
                self.custom_model_names = data.get("custom_model_names", {})
                self.host = data.get("host", "127.0.0.1")
                self.port = data.get("port", 7000)
                print(f"[Config] Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"[Config] Failed to load config: {e}")
    
    def validate(self) -> tuple[bool, str]:
        """
        验证配置是否有效
        返回: (is_valid, error_message)
        """
        if not self.tagger_model_dir:
            return False, "Tagger model directory is not configured"
        if not os.path.isdir(self.tagger_model_dir):
            return False, f"Model directory does not exist: {self.tagger_model_dir}"
        if not os.path.isfile(self.resolved_model_path):
            return False, f"Model file not found: {self.resolved_model_path}"
        if not os.path.isfile(self.resolved_tags_path):
            return False, f"Tags file not found: {self.resolved_tags_path}"
        return True, ""


# 全局单例配置实例
# 实例：在其他模块中通过 from config import app_config 引用
app_config = AppConfig()