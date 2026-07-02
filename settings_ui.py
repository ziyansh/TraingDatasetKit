"""
Settings UI 模块

功能：
1. 配置模型扫描路径
2. 扫描并显示找到的模型，支持自定义命名
3. 保存配置

实例：
```python
from settings_ui import create_settings_ui
settings_ui = create_settings_ui(interrogator_dropdown)
```
"""
import gradio as gr
from pathlib import Path
from typing import List, Dict

from config import app_config
from tagger.utils import scan_models, refresh_interrogators


def create_settings_ui(interrogator_dropdown=None):
    """创建 Settings 页面
    
    Args:
        interrogator_dropdown: Tagger UI 中的 interrogator 下拉组件，用于同步更新
    """
    
    with gr.Blocks(analytics_enabled=False) as settings_interface:
        gr.Markdown("# Settings")
        gr.Markdown("Configure model scanning path and manage models.")
        
        with gr.Row():
            with gr.Column(scale=3):
                tb_model_dir = gr.Textbox(
                    label='Model Scan Directory',
                    placeholder='E:/StableDiffusion/models/wd14-tagger',
                    value=app_config.tagger_model_dir,
                    lines=1
                )
            with gr.Column(scale=1, min_width=100):
                btn_scan = gr.Button(value='Scan', variant='primary')
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Found Models")
                df_models = gr.DataFrame(
                    headers=["Model Name", "Path", "Custom Name"],
                    datatype=["str", "str", "str"],
                    interactive=True,
                    col_count=(3, "fixed")
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Others")
                cb_open_browser = gr.Checkbox(
                    label='Open browser automatically on startup',
                    value=app_config.open_browser
                )
        
        with gr.Row():
            btn_save_config = gr.Button(value='Save Config')
            btn_refresh_tagger = gr.Button(value='Refresh Tagger Models')
        
        txt_status = gr.Textbox(label='Status', interactive=False)
        
        # ================================================================
        # 回调函数
        # ================================================================
        
        def scan_models_fn(directory: str):
            """扫描模型目录"""
            if not directory:
                return [], 'Please enter a directory path.'
            
            if not Path(directory).is_dir():
                return [], f'Error: Directory does not exist - {directory}'
            
            models = scan_models(directory)
            custom_names = app_config.custom_model_names or {}
            
            if not models:
                return [], f'Scanned {directory} but no models found.'
            
            data = []
            for m in models:
                custom_name = custom_names.get(m['path'], '')
                data.append([m['name'], m['path'], custom_name])
            
            msg = f'Found {len(models)} model(s) in {directory}'
            return data, msg
        
        def save_config_fn(directory: str, models_data, open_browser: bool):
            """保存配置，包括自定义模型名称"""
            app_config.tagger_model_dir = directory
            app_config.open_browser = open_browser
            
            custom_names = {}
            if models_data:
                for row in models_data:
                    if len(row) >= 3 and row[1]:
                        path = row[1]
                        custom_name = row[2] if len(row) > 2 else ''
                        if custom_name and custom_name.strip():
                            custom_names[path] = custom_name.strip()
            
            app_config.custom_model_names = custom_names
            app_config.save()
            
            msg = f'Configuration saved. Model directory: {directory}'
            if custom_names:
                msg += f', Custom names: {len(custom_names)} models'
            return msg
        
        def refresh_tagger_fn(directory: str, models_data, open_browser: bool):
            """刷新 Tagger 模型列表，保存自定义名称"""
            app_config.tagger_model_dir = directory
            app_config.open_browser = open_browser
            
            custom_names = {}
            if models_data:
                for row in models_data:
                    if len(row) >= 3 and row[1]:
                        path = row[1]
                        custom_name = row[2] if len(row) > 2 else ''
                        if custom_name and custom_name.strip():
                            custom_names[path] = custom_name.strip()
            
            app_config.custom_model_names = custom_names
            app_config.save()
            refresh_interrogators()
            
            from tagger.utils import interrogators
            model_names = list(interrogators.keys())
            
            if not model_names:
                dropdown_update = gr.Dropdown.update(choices=[], value=None)
                return f'No models found in {directory}. Please check the path.', dropdown_update
            
            msg = f'Refreshed {len(model_names)} model(s): {", ".join(model_names)}'
            dropdown_update = gr.Dropdown.update(choices=model_names, value=model_names[-1])
            return msg, dropdown_update
        
        # ================================================================
        # 绑定回调
        # ================================================================
        
        btn_scan.click(
            fn=scan_models_fn,
            inputs=[tb_model_dir],
            outputs=[df_models, txt_status]
        )
        
        btn_save_config.click(
            fn=save_config_fn,
            inputs=[tb_model_dir, df_models, cb_open_browser],
            outputs=[txt_status]
        )
        
        outputs_list = [txt_status]
        if interrogator_dropdown is not None:
            outputs_list.append(interrogator_dropdown)
        
        btn_refresh_tagger.click(
            fn=refresh_tagger_fn,
            inputs=[tb_model_dir, df_models, cb_open_browser],
            outputs=outputs_list
        )
    
    return settings_interface