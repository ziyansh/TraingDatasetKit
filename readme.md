# TraingDatasetKit

[中文版](readme_cn.md)

> An all-in-one local tool for AI training dataset tagging and editing. Backend based on [wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) and [dataset-tag-editor](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor) webui extensions, standalone with no webui dependency.

## Features

### 🧷 Tagger
- Single image and batch directory tagging with WD14 models
- Supports WaifuDiffusion ONNX models (`wd14-convnextv2-v2`, `wd14-vit-v2`, `wd14-convnext-v2`, `wd14-swinv2-v2`)
- DraWorld Launcher users can find the models in `.\stable-diffusion-webUI\.cache\sdwebuilauncher\hfmirror\refs\SmilingWolf\`
- Custom model directory scanning with user-defined model names
- Tag filtering: threshold, exclude, additional tags, alphabetical sort
- Tag formatting: replace underscores, escape brackets, include confidence weights
- Output formats: caption, JSON
- Ratings visualized as bar charts with confidence values
- Unload model after running (default on)

### ✏️ Dataset Tag Editor
- Load image datasets with caption file pairing
- Gallery view for browsing images
- Edit captions of selected images
- Batch edit captions (add, remove, replace, sort, deduplicate, truncate)
- Tag filtering by selection or by tags
- Move or delete files
- CLIP token count display
- Full UTKF (Undo/Redo) support

### ⚙️ Settings
- Configure model scan directory
- Scan and discover models automatically
- Rename models with custom names
- Persistent configuration (saved to `app_config.json`)

## Quick Start

### Prerequisites
- Python 3.10+
- [Download WD14 ONNX models](https://huggingface.co/SmilingWolf) manually and place them in a local directory

### Installation

```bash
# Clone or download the project, then:
cd TraingDatasetKit-AIO

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open http://127.0.0.1:7000 in your browser.

### Model Setup

1. Go to **⚙️ Settings** tab
2. Enter your model directory path (e.g., `E:/models/wd14-tagger`)
3. Click **Scan** to find available models
4. Optionally rename models in the **Custom Name** column
5. Click **Refresh Tagger Models** to update the model list
6. Switch to **🧷 Tagger** tab and select a model

Your model directory should be structured as:

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

## Configuration

Configuration is stored in `app_config.json`:

```json
{
    "tagger_model_dir": "E:/models/wd14-tagger",
    "custom_model_names": {
        "E:/models/wd14-tagger/my-model": "My Custom Model"
    },
    "host": "127.0.0.1",
    "port": 7000
}
```

## Dependencies

- gradio==3.41.2
- onnxruntime>=1.15.0
- Pillow, numpy, pandas, opencv-python
- transformers, torch
- huggingface-hub<0.20.0, jinja2==3.1.2

## Acknowledgments

- [wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) - Original WD14 Tagger webui extension
- [dataset-tag-editor](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor) - Original Dataset Tag Editor webui extension
- [SmilingWolf](https://huggingface.co/SmilingWolf) - WD14 model training
- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Stable Diffusion WebUI
