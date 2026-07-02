# TraingDatasetKit

**版本: V1.0.1**

[English Version](readme.md)

> 一款用于 AI 训练数据集打标与标签编辑的全能本地工具。后端基于 [wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) 和 [dataset-tag-editor](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor) 两款 WebUI 扩展开发，完全独立运行，无 WebUI 依赖。

## 功能特性

### 🧷 Tagger（打标器）
- 支持单张图片和批量目录打标
- **自动反推**：上传图片后自动执行打标
- 支持 WaifuDiffusion ONNX 模型系列（`wd14-convnextv2-v2`, `wd14-vit-v2`, `wd14-convnext-v2`, `wd14-swinv2-v2`）
- 使用秋葉启动器可以在`stable-diffusion-webUI\.cache\sdwebuilauncher\hfmirror\refs\SmilingWolf\` 下找到webUI的模型并导入
- 自定义模型目录扫描，支持用户自定义模型名称
- 标签过滤：阈值控制、排除标签、附加标签、字母排序
- 标签格式化：替换下划线、转义括号、包含置信度权重
- 输出格式：纯文本、JSON
- Rating 以柱状图展示，显示置信度数值
- 默认启用运行后卸载模型

### ✏️ Dataset Tag Editor（数据集标签编辑器）
- 加载图片数据集，自动匹配对应的标签文件
- 画廊视图浏览图片
- 编辑选中图片的标签
- 批量编辑标签（添加、删除、替换、排序、去重、截断）
- 按选择或按标签筛选
- 移动或删除文件
- CLIP token 计数显示
- 完整的撤销/重做支持

### ⚙️ Settings（设置）
- 配置模型扫描目录
- 自动扫描并发现可用模型
- 自定义模型命名
- **Others（其他）**：启动时自动打开浏览器选项
- 持久化配置（保存至 `app_config.json`）

## 快速开始

### 前置条件
- Python 3.10+
- 手动下载 [WD14 ONNX 模型](https://huggingface.co/SmilingWolf) 并放置于本地目录
- 或者在中导入webUI中的模型`stable-diffusion-webUI\.cache\sdwebuilauncher\hfmirror\refs\SmilingWolf\`（秋葉启动器）

### 安装

```bash
# 克隆或下载项目后：
cd TraingDatasetKit-AIO

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行
python app.py

# 或者双击 start.bat（仅 Windows）
```

在浏览器中打开 http://127.0.0.1:7000

### 模型配置

1. 进入 **⚙️ Settings** 标签页
2. 输入模型目录路径（例如 `E:/models/wd14-tagger`）
3. 点击 **Scan** 扫描可用模型
4. 可选：在 **Custom Name** 列中为模型重命名
5. 点击 **Refresh Tagger Models** 同步模型列表
6. 切换到 **🧷 Tagger** 标签页选择模型

模型目录结构示例：

```
wd14-tagger/
├── wd14-convnextv2-v2/
│   ├── model.onnx
│   └── selected_tags.csv
├── wd14-vit-v2/
│   ├── model.onnx
│   └── selected_tags.csv
└── your-custom-model/
    ├── model.onnx
    └── selected_tags.csv
```

## 配置文件

配置存储在 `app_config.json` 中：

```json
{
    "tagger_model_dir": "E:/models/wd14-tagger",
    "custom_model_names": {
        "E:/models/wd14-tagger/my-model": "My Custom Model"
    },
    "host": "127.0.0.1",
    "port": 7000,
    "open_browser": true
}
```

## 依赖项

- gradio==3.41.2
- onnxruntime>=1.15.0
- Pillow, numpy, pandas, opencv-python
- transformers, torch
- huggingface-hub<0.20.0, jinja2==3.1.2

## 致谢

- [wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) - 原始 WD14 Tagger WebUI 扩展
- [dataset-tag-editor](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor) - 原始 Dataset Tag Editor WebUI 扩展
- [SmilingWolf](https://huggingface.co/SmilingWolf) - WD14 模型训练
- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Stable Diffusion WebUI


## 版本历史

- **V1.0.1** - 新增自动反推功能：上传图片后自动执行打标
- **V1.0.0** - 初始版本