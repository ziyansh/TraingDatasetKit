"""
WD14 Tagger UI 模块

适配说明：
- 移除 modules.call_queue.wrap_gradio_gpu_call / modules.ui / modules.generation_parameters_copypaste 依赖
- 移除预设（preset）的 webui 样式按钮
- 使用 tagger.utils 替代 webui 版本
- 简化批处理功能

实例：
```python
from tagger.ui import create_tagger_ui
# 在 app.py 中调用 create_tagger_ui() 获取 Gradio Blocks
```
"""
import os
import json
import gradio as gr

from collections import OrderedDict
from pathlib import Path
from glob import glob
from PIL import Image, UnidentifiedImageError

from tagger import format, utils
from tagger.utils import split_str
from tagger.interrogator import Interrogator


class TaggerUI:
    """Tagger UI 类"""
    
    def __init__(self):
        self.interface = None
        self.interrogator_dropdown = None
    
    def create_ui(self):
        """创建 Tagger UI 的 Gradio Blocks"""
        
        with gr.Blocks(analytics_enabled=False) as tagger_interface:
            self.interface = tagger_interface
            
            with gr.Column(variant='panel'):

                with gr.Tabs():
                    with gr.TabItem(label='Single process'):
                        image = gr.Image(
                            label='Source',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )

                    with gr.TabItem(label='Batch from directory'):
                        batch_input_glob = gr.Textbox(
                            label='Input directory',
                            placeholder='/path/to/images or /path/to/images/**/*'
                        )
                        batch_input_recursive = gr.Checkbox(
                            label='Use recursive with glob pattern'
                        )

                        batch_output_dir = gr.Textbox(
                            label='Output directory',
                            placeholder='Leave blank to save images to the same path.'
                        )

                        batch_output_filename_format = gr.Textbox(
                            label='Output filename format',
                            placeholder='Leave blank to use same filename as original.',
                            value='[name].[output_extension]'
                        )

                        import hashlib
                        with gr.Accordion(
                            label='Output filename formats',
                            open=False
                        ):
                            gr.Markdown(
                                value=f'''
                                ### Related to original file
                                - `[name]`: Original filename without extension
                                - `[extension]`: Original extension
                                - `[hash:<algorithms>]`: Original extension
                                    Available algorithms: `{{', '.join(hashlib.algorithms_available)}}`

                                ### Related to output file
                                - `[output_extension]`: Output extension (has no dot)

                                ## Examples
                                ### Original filename without extension
                                `[name].[output_extension]`

                                ### Original file's hash (good for deleting duplication)
                                `[hash:sha1].[output_extension]`
                                '''
                            )

                        batch_output_action_on_conflict = gr.Dropdown(
                            label='Action on existing caption',
                            value='ignore',
                            choices=['ignore', 'copy', 'append', 'prepend']
                        )

                        batch_remove_duplicated_tag = gr.Checkbox(
                            label='Remove duplicated tag'
                        )

                        batch_output_save_json = gr.Checkbox(
                            label='Save with JSON'
                        )

                with gr.Tab(label='Ratings'):
                    ratings = gr.HTML(
                        label='Ratings'
                    )

                with gr.Tab(label='Output'):
                    output = gr.Textbox(
                        label='Caption',
                        interactive=False,
                        lines=10,
                        elem_id='tagger-output'
                    )

                submit = gr.Button(
                    value='Interrogate',
                    variant='primary'
                )

                info = gr.HTML()

                with gr.Column():
                    with gr.Row(variant='compact'):
                        interrogator_names = utils.refresh_interrogators()
                        self.interrogator_dropdown = gr.Dropdown(
                            label='Interrogator',
                            choices=interrogator_names,
                            value=(
                                None
                                if len(interrogator_names) < 1 else
                                interrogator_names[-1]
                            )
                        )
                        refresh_interrogator_btn = gr.Button(value='🔄 Refresh')

                    threshold = gr.Slider(
                        label='Threshold',
                        minimum=0,
                        maximum=1,
                        value=0.35
                    )

                    additional_tags = gr.Textbox(
                        label='Additional tags (split by comma)',
                        elem_id='additional-tags'
                    )

                    exclude_tags = gr.Textbox(
                        label='Exclude tags (split by comma)',
                        elem_id='exclude-tags'
                    )

                    sort_by_alphabetical_order = gr.Checkbox(
                        label='Sort by alphabetical order',
                    )
                    add_confident_as_weight = gr.Checkbox(
                        label='Include confident of tags matches in results'
                    )
                    replace_underscore = gr.Checkbox(
                        label='Use spaces instead of underscore',
                        value=True
                    )
                    replace_underscore_excludes = gr.Textbox(
                        label='Excludes (split by comma)',
                        value='0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'
                    )
                    escape_tag = gr.Checkbox(
                        label='Escape brackets',
                    )

                    unload_model_after_running = gr.Checkbox(
                        label='Unload model after running',
                        value=True
                    )

                def refresh_interrogators_fn():
                    names = utils.refresh_interrogators()
                    return gr.Dropdown.update(choices=names, value=names[-1] if names else None)

                refresh_interrogator_btn.click(
                    fn=refresh_interrogators_fn,
                    outputs=[self.interrogator_dropdown]
                )

            def unload_interrogators():
                unloaded_models = 0
                for i in utils.interrogators.values():
                    if i.unload():
                        unloaded_models = unloaded_models + 1
                return [f'Successfully unload {unloaded_models} model(s)']

            # 处理推理
            def on_interrogate(
                image_input,
                batch_input_glob_str,
                batch_input_recursive_bool,
                batch_output_dir_str,
                batch_output_filename_format_str,
                batch_output_action_on_conflict_str,
                batch_remove_duplicated_tag_bool,
                batch_output_save_json_bool,
                interrogator_name,
                threshold_val,
                additional_tags_str,
                exclude_tags_str,
                sort_by_alphabetical_order_bool,
                add_confident_as_weight_bool,
                replace_underscore_bool,
                replace_underscore_excludes_str,
                escape_tag_bool,
                unload_model_after_running_bool
            ):
                if interrogator_name not in utils.interrogators:
                    return ['', None, None, f"'{interrogator_name}' is not a valid interrogator"]

                interrogator_inst: Interrogator = utils.interrogators[interrogator_name]

                postprocess_opts = (
                    threshold_val,
                    split_str(additional_tags_str),
                    split_str(exclude_tags_str),
                    sort_by_alphabetical_order_bool,
                    add_confident_as_weight_bool,
                    replace_underscore_bool,
                    split_str(replace_underscore_excludes_str),
                    escape_tag_bool
                )

                # single process
                if image_input is not None:
                    # 如果是 numpy array，转为 PIL Image
                    if isinstance(image_input, dict) and 'image' in image_input:
                        img = image_input['image']
                    else:
                        img = image_input
                    
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img).convert('RGB')
                    
                    img = img.convert('RGB')
                    result_ratings, result_tags = interrogator_inst.interrogate(img)
                    processed_tags = Interrogator.postprocess_tags(
                        result_tags,
                        *postprocess_opts
                    )

                    if unload_model_after_running_bool:
                        interrogator_inst.unload()

                    # 格式化 ratings 为柱状图 HTML
                    ratings_html = '<div style="font-size:12px;">'
                    for k, v in sorted(result_ratings.items(), key=lambda x: -x[1]):
                        pct = v * 100
                        ratings_html += f'''
                            <div style="margin-bottom:4px;">
                                <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                                    <span style="font-weight:bold;">{k}</span>
                                    <span>{v:.4f}</span>
                                </div>
                                <div style="width:100%;height:12px;background:#333;border-radius:6px;overflow:hidden;">
                                    <div style="width:{pct}%;height:100%;background:#4CAF50;border-radius:6px;transition:width 0.3s ease;"></div>
                                </div>
                            </div>
                        '''
                    ratings_html += '</div>'

                    return [
                        ', '.join(processed_tags),
                        ratings_html,
                        json.dumps({k: float(v) for k, v in result_tags.items()}, indent=2),
                        ''
                    ]

                # batch process
                batch_input_glob_str = batch_input_glob_str.strip()
                batch_output_dir_str = batch_output_dir_str.strip()
                batch_output_filename_format_str = batch_output_filename_format_str.strip()

                if batch_input_glob_str != '':
                    # if there is no glob pattern, insert it automatically
                    if not batch_input_glob_str.endswith('*'):
                        if not batch_input_glob_str.endswith(os.sep):
                            batch_input_glob_str += os.sep
                        batch_input_glob_str += '*'

                    # get root directory of input glob pattern
                    base_dir = batch_input_glob_str.replace('?', '*')
                    base_dir = base_dir.split(os.sep + '*').pop(0)

                    # check the input directory path
                    if not os.path.isdir(base_dir):
                        return ['', None, None, 'input path is not a directory']

                    supported_extensions = [
                        e
                        for e, f in Image.registered_extensions().items()
                        if f in Image.OPEN
                    ]

                    paths = [
                        Path(p)
                        for p in glob(batch_input_glob_str, recursive=batch_input_recursive_bool)
                        if '.' + p.split('.').pop().lower() in supported_extensions
                    ]

                    print(f'found {len(paths)} image(s)')

                    for path in paths:
                        try:
                            img = Image.open(path)
                        except UnidentifiedImageError:
                            print(f'${path} is not supported image type')
                            continue

                        # guess the output path
                        base_dir_last = Path(base_dir).parts[-1]
                        base_dir_last_idx = path.parts.index(base_dir_last)
                        output_dir_obj = Path(batch_output_dir_str) if batch_output_dir_str else Path(base_dir)
                        output_dir_obj = output_dir_obj.joinpath(*path.parts[base_dir_last_idx + 1:]).parent

                        output_dir_obj.mkdir(0o777, True, True)

                        # format output filename
                        format_info = format.Info(path, 'txt')

                        try:
                            formatted_output_filename = format.pattern.sub(
                                lambda m: format.format(m, format_info),
                                batch_output_filename_format_str
                            )
                        except (TypeError, ValueError) as error:
                            return ['', None, None, str(error)]

                        output_path = output_dir_obj.joinpath(formatted_output_filename)

                        output_lines = []

                        if output_path.is_file():
                            output_lines.append(output_path.read_text(errors='ignore').strip())

                            if batch_output_action_on_conflict_str == 'ignore':
                                print(f'skipping {path}')
                                continue

                        result_ratings, result_tags = interrogator_inst.interrogate(img)
                        processed_tags = Interrogator.postprocess_tags(
                            result_tags,
                            *postprocess_opts
                        )

                        print(f'found {len(processed_tags)} tags out of {len(result_tags)} from {path}')

                        plain_tags = ', '.join(processed_tags)

                        if batch_output_action_on_conflict_str == 'copy':
                            output_lines = [plain_tags]
                        elif batch_output_action_on_conflict_str == 'prepend':
                            output_lines.insert(0, plain_tags)
                        else:
                            output_lines.append(plain_tags)

                        if batch_remove_duplicated_tag_bool:
                            output_path.write_text(
                                ', '.join(
                                    OrderedDict.fromkeys(
                                        map(str.strip, ','.join(output_lines).split(','))
                                    )
                                ),
                                encoding='utf-8'
                            )
                        else:
                            output_path.write_text(
                                ', '.join(output_lines),
                                encoding='utf-8'
                            )

                        if batch_output_save_json_bool:
                            output_path.with_suffix('.json').write_text(
                                json.dumps([result_ratings, result_tags])
                            )

                    print('all done :)')

                if unload_model_after_running_bool:
                    interrogator_inst.unload()

                return ['', None, None, '']

            submit.click(
            fn=on_interrogate,
            inputs=[
                image,
                batch_input_glob, batch_input_recursive,
                batch_output_dir, batch_output_filename_format,
                batch_output_action_on_conflict,
                batch_remove_duplicated_tag, batch_output_save_json,
                self.interrogator_dropdown, threshold,
                additional_tags, exclude_tags,
                sort_by_alphabetical_order, add_confident_as_weight,
                replace_underscore, replace_underscore_excludes, escape_tag,
                unload_model_after_running
            ],
            outputs=[output, ratings, info, info]
        )

        image.change(
            fn=on_interrogate,
            inputs=[
                image,
                batch_input_glob, batch_input_recursive,
                batch_output_dir, batch_output_filename_format,
                batch_output_action_on_conflict,
                batch_remove_duplicated_tag, batch_output_save_json,
                self.interrogator_dropdown, threshold,
                additional_tags, exclude_tags,
                sort_by_alphabetical_order, add_confident_as_weight,
                replace_underscore, replace_underscore_excludes, escape_tag,
                unload_model_after_running
            ],
            outputs=[output, ratings, info, info]
        )

        return self