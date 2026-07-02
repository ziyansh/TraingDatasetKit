"""
预设管理模块

适配说明：
- 移除 modules.images.sanitize_filename_part 依赖，使用简单的文件名清洗函数

实例：
```python
from tagger.preset import Preset
preset = Preset(Path("./presets"))
```
"""
import os
import json
import re

from typing import Tuple, List, Dict
from pathlib import Path

PresetDict = Dict[str, Dict[str, any]]


def sanitize_filename_part(filename: str) -> str:
    """简易文件名清洗（替代 webui 的 sanitize_filename_part）"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


class Preset:
    base_dir: Path
    default_filename: str
    default_values: PresetDict
    components: List[object]

    def __init__(
        self,
        base_dir: os.PathLike,
        default_filename='default.json'
    ) -> None:
        self.base_dir = Path(base_dir)
        self.default_filename = default_filename
        self.default_values = self.load(default_filename)[1]
        self.components = []

    def component(self, component_class: object, **kwargs) -> object:
        from gradio.context import Context
        parent = Context.block
        paths = [kwargs['label']]

        while parent is not None:
            if hasattr(parent, 'label'):
                paths.insert(0, parent.label)

            parent = parent.parent

        path = '/'.join(paths)

        component = component_class(**{
            **kwargs,
            **self.default_values.get(path, {})
        })

        setattr(component, 'path', path)

        self.components.append(component)
        return component

    def load(self, filename: str) -> Tuple[str, PresetDict]:
        if not filename.endswith('.json'):
            filename += '.json'

        path = self.base_dir.joinpath(sanitize_filename_part(filename))
        configs = {}

        if path.is_file():
            configs = json.loads(path.read_text())

        return path, configs

    def save(self, filename: str, *values) -> Tuple:
        path, configs = self.load(filename)

        for index, component in enumerate(self.components):
            config = configs.get(component.path, {})
            config['value'] = values[index]

            for attr in ['visible', 'min', 'max', 'step']:
                if hasattr(component, attr):
                    config[attr] = config.get(attr, getattr(component, attr))

            configs[component.path] = config

        self.base_dir.mkdir(0o777, True, True)
        path.write_text(
            json.dumps(configs, indent=4)
        )

        return 'successfully saved the preset'

    def apply(self, filename: str) -> Tuple:
        values = self.load(filename)[1]
        outputs = []

        for component in self.components:
            config = values.get(component.path, {})

            if 'value' in config and hasattr(component, 'choices'):
                if config['value'] not in component.choices:
                    config['value'] = None

            outputs.append(component.update(**config))

        return (*outputs, 'successfully loaded the preset')

    def list(self) -> List[str]:
        presets = [
            p.name
            for p in self.base_dir.glob('*.json')
            if p.is_file()
        ]

        if len(presets) < 1:
            presets.append(self.default_filename)

        return presets