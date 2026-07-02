"""
CLIP Tokenizer 封装

适配说明：
- 移除 webui 的 FrozenCLIPEmbedder / sd_model / extra_networks 依赖
- 使用 transformers 库的 CLIPTokenizer 作为独立的 token 计数工具
- 不支持 raw CLIP 模式（独立模式下无法访问 webui 的 sd_model）

实例：
```python
from dataset_tag_editor.tokenizer.clip_tokenizer import tokenize
tokens, count = tokenize("1girl, solo, looking at viewer")
print(f"Token count: {count}")
```
"""
from typing import Tuple, List, Optional
from transformers import CLIPTokenizer

# 使用标准 CLIP tokenizer（openai/clip-vit-large-patch14）
_tokenizer: Optional[CLIPTokenizer] = None


def _get_tokenizer() -> CLIPTokenizer:
    """懒加载 CLIP tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    return _tokenizer


def tokenize(text: str, use_raw_clip: bool = True) -> Tuple[List[int], int]:
    """
    对文本进行 tokenize 并返回 token 列表和数量
    
    参数:
        text: 输入文本
        use_raw_clip: 是否使用原始 CLIP tokenize（独立模式下始终为 True）
    
    返回:
        (tokens_list, token_count)
    """
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)
    return tokens, len(tokens)


def get_target_token_count(token_count: int) -> int:
    """获取目标 token 数量（简单版本，直接返回输入值）"""
    return token_count