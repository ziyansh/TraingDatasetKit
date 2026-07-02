"""
本地 Tag 编辑器 - 主入口

功能：
1. WD14 Tagger 服务 - 单图/批量打标
2. 训练集 Tag 编辑 - 加载、编辑、过滤、保存数据集标签

适配说明：
- 基于 webui 扩展 wd14-tagger 和 dataset-tag-editor 后端开发
- 使用 Gradio 实现独立本地 UI
- 移除所有 webui 依赖

实例：
```python
# 启动服务
python app.py

# 或指定模型路径和端口
python app.py --model-dir "E:/models/wd-tagger" --port 7860
```
"""
import os
import sys
import argparse
import gradio as gr
from typing import List
from collections import namedtuple

from config import app_config
from dataset_tag_editor import dte_logic, filters
from dataset_tag_editor.dte_logic import DatasetTagEditor, dte_settings
from dataset_tag_editor.tag_editor_ui import ui_instance as ui
from dataset_tag_editor.tag_editor_ui.ui_common import dte_instance, dte_module
from tagger.ui import TaggerUI
from settings_ui import create_settings_ui

# ================================================================
# 配置定义（与原始扩展保持一致）
# ================================================================

SortBy = dte_instance.SortBy
SortOrder = dte_instance.SortOrder

GeneralConfig = namedtuple('GeneralConfig', [
    'backup', 'dataset_dir', 'caption_ext', 'load_recursive',
    'load_caption_from_filename', 'replace_new_line',
    'use_interrogator', 'use_interrogator_names',
    'use_custom_threshold_booru', 'custom_threshold_booru',
    'use_custom_threshold_waifu', 'custom_threshold_waifu',
    'save_kohya_metadata', 'meta_output_path', 'meta_input_path',
    'meta_overwrite', 'meta_save_as_caption', 'meta_use_full_path'
])
FilterConfig = namedtuple('FilterConfig', [
    'sw_prefix', 'sw_suffix', 'sw_regex', 'sort_by', 'sort_order', 'logic'
])
BatchEditConfig = namedtuple('BatchEditConfig', [
    'show_only_selected', 'prepend', 'use_regex', 'target',
    'sw_prefix', 'sw_suffix', 'sw_regex', 'sory_by', 'sort_order',
    'batch_sort_by', 'batch_sort_order', 'token_count'
])
EditSelectedConfig = namedtuple('EditSelectedConfig', [
    'auto_copy', 'sort_on_save', 'warn_change_not_saved',
    'use_interrogator_name', 'sort_by', 'sort_order'
])
MoveDeleteConfig = namedtuple('MoveDeleteConfig', [
    'range', 'target', 'caption_ext', 'destination'
])

# 默认配置
CFG_GENERAL_DEFAULT = GeneralConfig(
    True, '', '.txt', False, True, False,
    'No', [], False, 0.7, False, 0.35,
    False, '', '', True, False, False
)
CFG_FILTER_P_DEFAULT = FilterConfig(False, False, False, SortBy.ALPHA.value, SortOrder.ASC.value, 'AND')
CFG_FILTER_N_DEFAULT = FilterConfig(False, False, False, SortBy.ALPHA.value, SortOrder.ASC.value, 'OR')
CFG_BATCH_EDIT_DEFAULT = BatchEditConfig(
    True, False, False, 'Only Selected Tags',
    False, False, False, SortBy.ALPHA.value, SortOrder.ASC.value,
    SortBy.ALPHA.value, SortOrder.ASC.value, 75
)
CFG_EDIT_SELECTED_DEFAULT = EditSelectedConfig(False, False, False, '', SortBy.ALPHA.value, SortOrder.ASC.value)
CFG_MOVE_DELETE_DEFAULT = MoveDeleteConfig('Selected One', ['.txt'], '.txt', '')


# ================================================================
# UI 回调函数
# ================================================================

def get_filters():
    """获取当前所有过滤器"""
    filters_list = [
        ui.filter_by_tags.tag_filter_ui.get_filter(),
        ui.filter_by_tags.tag_filter_ui_neg.get_filter(),
        ui.filter_by_selection.path_filter
    ]
    return filters_list


def update_gallery():
    """更新画廊状态"""
    img_indices = dte_instance.get_filtered_imgindices(filters=get_filters())
    total_image_num = len(dte_instance.dataset)
    displayed_image_num = len(img_indices)
    ui.gallery_state.register_value('Displayed Images', f'{displayed_image_num} / {total_image_num} total')
    
    current_tag_filter = str(ui.filter_by_tags.tag_filter_ui.get_filter())
    neg_tag_filter = str(ui.filter_by_tags.tag_filter_ui_neg.get_filter())
    if current_tag_filter and neg_tag_filter:
        filter_text = f"{current_tag_filter} AND {neg_tag_filter}"
    else:
        filter_text = current_tag_filter or neg_tag_filter or ''
    ui.gallery_state.register_value('Current Tag Filter', filter_text)
    
    ui.gallery_state.register_value(
        'Current Selection Filter',
        f'{len(ui.filter_by_selection.path_filter.paths)} images'
    )
    
    return [
        [str(i) for i in img_indices],
        1,  # nb_hidden_dataset_filter_apply
        -1,  # nb_hidden_image_index
        -1,  # nb_hidden_image_index_prev
        -1,  # nb_hidden_image_index_save_or_not
        ui.gallery_state.get_current_gallery_txt()
    ]


def update_filter_and_gallery():
    """更新过滤器和画廊"""
    return \
        [ui.filter_by_tags.tag_filter_ui.cbg_tags_update(),
         ui.filter_by_tags.tag_filter_ui_neg.cbg_tags_update()] + \
        update_gallery() + \
        ui.batch_edit_captions.get_common_tags(get_filters, ui.filter_by_tags) + \
        [', '.join(ui.filter_by_tags.tag_filter_ui.filter.tags)] + \
        [ui.batch_edit_captions.tag_select_ui_remove.cbg_tags_update()] + \
        ['', '']  # tb_caption, tb_edit_caption


# ================================================================
# Dataset Tag Editor UI
# ================================================================

def create_dataset_tag_editor_ui():
    """创建 Dataset Tag Editor 主界面"""
    
    cfg_general = CFG_GENERAL_DEFAULT
    cfg_filter_p = CFG_FILTER_P_DEFAULT
    cfg_filter_n = CFG_FILTER_N_DEFAULT
    cfg_batch_edit = CFG_BATCH_EDIT_DEFAULT
    cfg_edit_selected = CFG_EDIT_SELECTED_DEFAULT
    cfg_file_move_delete = CFG_MOVE_DELETE_DEFAULT

    with gr.Blocks(analytics_enabled=False) as dte_interface:
        gr.HTML(value="""
        This extension works well with text captions in comma-separated style 
        (such as the tags generated by DeepBooru interrogator).
        """)

        ui.toprow.create_ui(cfg_general)

        with gr.Accordion(label='Reload/Save Settings (config.json)', open=False):
            with gr.Row():
                btn_reload_config_file = gr.Button(value='Reload settings')
                btn_save_setting_as_default = gr.Button(value='Save current settings')
                btn_restore_default = gr.Button(value='Restore settings to default')

        with gr.Row(equal_height=False):
            with gr.Column():
                ui.load_dataset.create_ui(cfg_general)
                ui.dataset_gallery.create_ui(6)  # 6 columns for image gallery
                ui.gallery_state.create_ui()

            with gr.Tab(label='Filter by Tags'):
                ui.filter_by_tags.create_ui(cfg_filter_p, cfg_filter_n, get_filters)

            with gr.Tab(label='Filter by Selection'):
                ui.filter_by_selection.create_ui(6)

            with gr.Tab(label='Batch Edit Captions'):
                ui.batch_edit_captions.create_ui(cfg_batch_edit, get_filters)

            with gr.Tab(label='Edit Caption of Selected Image'):
                ui.edit_caption_of_selected_image.create_ui(cfg_edit_selected)

            with gr.Tab(label='Move or Delete Files'):
                ui.move_or_delete_files.create_ui(cfg_file_move_delete)

        # ================================================================
        # UI 组件列表（用于配置保存/恢复）
        # ================================================================

        components_general = [
            ui.toprow.cb_backup, ui.load_dataset.tb_img_directory,
            ui.load_dataset.tb_caption_file_ext, ui.load_dataset.cb_load_recursive,
            ui.load_dataset.cb_load_caption_from_filename, ui.load_dataset.cb_replace_new_line_with_comma,
            ui.load_dataset.rb_use_interrogator, ui.load_dataset.dd_intterogator_names,
            ui.load_dataset.cb_use_custom_threshold_booru, ui.load_dataset.sl_custom_threshold_booru,
            ui.load_dataset.cb_use_custom_threshold_waifu, ui.load_dataset.sl_custom_threshold_waifu,
            ui.toprow.cb_save_kohya_metadata, ui.toprow.tb_metadata_output,
            ui.toprow.tb_metadata_input, ui.toprow.cb_metadata_overwrite,
            ui.toprow.cb_metadata_as_caption, ui.toprow.cb_metadata_use_fullpath
        ]
        components_filter = \
            [ui.filter_by_tags.tag_filter_ui.cb_prefix, ui.filter_by_tags.tag_filter_ui.cb_suffix,
             ui.filter_by_tags.tag_filter_ui.cb_regex, ui.filter_by_tags.tag_filter_ui.rb_sort_by,
             ui.filter_by_tags.tag_filter_ui.rb_sort_order, ui.filter_by_tags.tag_filter_ui.rb_logic] + \
            [ui.filter_by_tags.tag_filter_ui_neg.cb_prefix, ui.filter_by_tags.tag_filter_ui_neg.cb_suffix,
             ui.filter_by_tags.tag_filter_ui_neg.cb_regex, ui.filter_by_tags.tag_filter_ui_neg.rb_sort_by,
             ui.filter_by_tags.tag_filter_ui_neg.rb_sort_order, ui.filter_by_tags.tag_filter_ui_neg.rb_logic]
        components_batch_edit = [
            ui.batch_edit_captions.cb_show_only_tags_selected, ui.batch_edit_captions.cb_prepend_tags,
            ui.batch_edit_captions.cb_use_regex, ui.batch_edit_captions.rb_sr_replace_target,
            ui.batch_edit_captions.tag_select_ui_remove.cb_prefix,
            ui.batch_edit_captions.tag_select_ui_remove.cb_suffix,
            ui.batch_edit_captions.tag_select_ui_remove.cb_regex,
            ui.batch_edit_captions.tag_select_ui_remove.rb_sort_by,
            ui.batch_edit_captions.tag_select_ui_remove.rb_sort_order,
            ui.batch_edit_captions.rb_sort_by, ui.batch_edit_captions.rb_sort_order,
            ui.batch_edit_captions.nb_token_count
        ]
        components_edit_selected = [
            ui.edit_caption_of_selected_image.cb_copy_caption_automatically,
            ui.edit_caption_of_selected_image.cb_sort_caption_on_save,
            ui.edit_caption_of_selected_image.cb_ask_save_when_caption_changed,
            ui.edit_caption_of_selected_image.dd_intterogator_names_si,
            ui.edit_caption_of_selected_image.rb_sort_by,
            ui.edit_caption_of_selected_image.rb_sort_order
        ]
        components_move_delete = [
            ui.move_or_delete_files.rb_move_or_delete_target_data,
            ui.move_or_delete_files.cbg_move_or_delete_target_file,
            ui.move_or_delete_files.tb_move_or_delete_caption_ext,
            ui.move_or_delete_files.tb_move_or_delete_destination_dir
        ]

        configurable_components = components_general + components_filter + \
            components_batch_edit + components_edit_selected + components_move_delete

        # ================================================================
        # 设置回调
        # ================================================================

        o_update_gallery = [
            ui.dataset_gallery.cbg_hidden_dataset_filter,
            ui.dataset_gallery.nb_hidden_dataset_filter_apply,
            ui.dataset_gallery.nb_hidden_image_index,
            ui.dataset_gallery.nb_hidden_image_index_prev,
            ui.edit_caption_of_selected_image.nb_hidden_image_index_save_or_not,
            ui.gallery_state.txt_gallery
        ]

        o_update_filter_and_gallery = \
            [ui.filter_by_tags.tag_filter_ui.cbg_tags,
             ui.filter_by_tags.tag_filter_ui_neg.cbg_tags] + \
            o_update_gallery + \
            [ui.batch_edit_captions.tb_common_tags,
             ui.batch_edit_captions.tb_edit_tags] + \
            [ui.batch_edit_captions.tb_sr_selected_tags] + \
            [ui.batch_edit_captions.tag_select_ui_remove.cbg_tags] + \
            [ui.edit_caption_of_selected_image.tb_caption,
             ui.edit_caption_of_selected_image.tb_edit_caption]

        ui.toprow.set_callbacks(ui.load_dataset)
        ui.load_dataset.set_callbacks(
            o_update_filter_and_gallery, ui.toprow, ui.dataset_gallery,
            ui.filter_by_tags, ui.filter_by_selection, ui.batch_edit_captions,
            update_filter_and_gallery
        )
        ui.dataset_gallery.set_callbacks(ui.load_dataset, ui.gallery_state, get_filters)
        ui.gallery_state.set_callbacks(ui.dataset_gallery)
        ui.filter_by_tags.set_callbacks(
            o_update_gallery, o_update_filter_and_gallery,
            ui.batch_edit_captions, ui.move_or_delete_files,
            update_gallery, update_filter_and_gallery, get_filters
        )
        ui.filter_by_selection.set_callbacks(
            o_update_filter_and_gallery, ui.dataset_gallery, ui.filter_by_tags,
            get_filters, update_filter_and_gallery
        )
        ui.batch_edit_captions.set_callbacks(
            o_update_filter_and_gallery, ui.load_dataset, ui.filter_by_tags,
            get_filters, update_filter_and_gallery
        )
        ui.edit_caption_of_selected_image.set_callbacks(
            o_update_filter_and_gallery, ui.dataset_gallery, ui.load_dataset,
            get_filters, update_filter_and_gallery
        )
        ui.move_or_delete_files.set_callbacks(
            o_update_filter_and_gallery, ui.dataset_gallery, ui.filter_by_tags,
            ui.batch_edit_captions, ui.filter_by_selection,
            ui.edit_caption_of_selected_image, get_filters, update_filter_and_gallery
        )

    return dte_interface


# ================================================================
# 主入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local Tag Editor & Tagger')
    parser.add_argument('--model-dir', type=str, default='',
                        help='Path to the WD14 tagger model directory')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=7000,
                        help='Port to bind the server to')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.model_dir:
        app_config.tagger_model_dir = args.model_dir
    if args.host:
        app_config.host = args.host
    if args.port:
        app_config.port = args.port
    
    # 保存配置
    app_config.save()
    
    # 刷新 tagger 模型列表
    from tagger import utils
    utils.refresh_interrogators()
    
    print(f"Starting TraingDatasetKit...")
    print(f"  Host: {app_config.host}:{app_config.port}")
    print(f"  Model Dir: {app_config.tagger_model_dir}")
    
    # 创建 UI
    tagger_ui_instance = TaggerUI().create_ui()
    tagger_ui = tagger_ui_instance.interface
    
    dte_ui = create_dataset_tag_editor_ui()
    settings_ui = create_settings_ui(tagger_ui_instance.interrogator_dropdown)
    
    # 使用 TabbedInterface 组合
    app = gr.TabbedInterface(
        [tagger_ui, dte_ui, settings_ui],
        ["🧷 Tagger", "✏️ Dataset Tag Editor", "⚙️ Settings"],
        title="Local Tag Editor & Tagger"
    )
    
    app.queue()
    app.launch(
        server_name="127.0.0.1",
        server_port=app_config.port,
        show_error=True
    )