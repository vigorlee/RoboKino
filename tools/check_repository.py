"""Validate the repository without importing or starting Isaac Sim.

Copyright (c) 2026 vigorlee. SPDX-License-Identifier: MIT
"""
import ast
import json
from pathlib import Path
import re
import struct
from urllib.parse import unquote

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    python_files = sorted(ROOT.rglob('*.py'))
    python_files = [path for path in python_files
                    if not any(part.startswith('.') or part == 'venv'
                               for part in path.relative_to(ROOT).parts)]
    for path in python_files:
        compile(path.read_bytes(), str(path), 'exec')

    configs = {}
    for directory in ('controllers', 'data_collect', 'envs'):
        for path in (ROOT / directory).rglob('*.yaml'):
            config = yaml.safe_load(path.read_text(encoding='utf-8'))
            assert isinstance(config, dict), f'Expected mapping: {path}'
            configs[path.relative_to(ROOT).as_posix()] = config

    tree = ast.parse((ROOT / 'evaluate.py').read_text(encoding='utf-8'))
    presets = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name)
                           and target.id == 'EVAL_TASK_PRESETS'
                           for target in node.targets))
    robot_configs = configs['envs/cfg/robots.yaml']
    object_configs = configs['envs/cfg/objects.yaml']
    task_tree = ast.parse((ROOT / 'tasks/__init__.py').read_text(encoding='utf-8'))
    task_names = {alias.name for node in task_tree.body
                  if isinstance(node, ast.ImportFrom) for alias in node.names}
    for task, preset in presets.items():
        config = configs[preset['config']]
        assert config['task'] in task_names, f'Unregistered task: {task}'
        assert f"{preset['env']}_table" in object_configs, f'Missing scene: {task}'
        assert f"base_{preset['env']}" in robot_configs, f'Missing base: {task}'
        assert preset['max_steps'] > 0, f'Invalid step budget: {task}'
        assert len(config['cameras']) == 3, f'Expected three cameras: {task}'
        for camera in config['cameras'].values():
            assert camera['width'] > 0 and camera['height'] > 0
            assert len(camera['intrinsics']) == 4
        for key, _, fallback in preset['env_vars']:
            assert config.get(key, fallback) in object_configs, f'Missing object: {task}/{key}'
        assert task in (ROOT / 'robokino_benchmark.sh').read_text(encoding='utf-8')

    links_checked = 0
    for path in ROOT.rglob('*.md'):
        if any(part.startswith('.') for part in path.relative_to(ROOT).parts):
            continue
        for link in re.findall(r'\]\(([^\s)]+)\)', path.read_text(encoding='utf-8')):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', link) or link.startswith('#'):
                continue
            local = unquote(link.split('#', 1)[0])
            assert (path.parent / local).exists(), f'Broken link in {path.name}: {link}'
            links_checked += 1

    image = ROOT / 'docs/images/robokino-framework.png'
    header = image.read_bytes()[:24]
    assert header[:8] == b'\x89PNG\r\n\x1a\n', 'Invalid framework PNG'
    width, height = struct.unpack('>II', header[16:24])
    assert width >= 2000 and height >= 700, 'Framework figure resolution is too small'
    assert (ROOT / 'docs/images/robokino-framework.pdf').read_bytes().startswith(b'%PDF-')
    assert not (ROOT / 'docs/framework.jpg').exists(), 'Unexpected upstream figure'
    assert (ROOT / '.project-root').exists(), 'Missing project root marker'
    manifest = json.loads((ROOT / 'docs/upstream-import.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        assert (ROOT / entry['path']).is_file(), f"Missing imported file: {entry['path']}"

    print(f'Validated {len(python_files)} Python files, {len(configs)} YAML files, '
          f'{len(presets)} evaluation presets, {links_checked} documentation links, '
          f'and the {width}x{height} RoboKino figure.')
    print('Isaac Sim, GPU physics, assets, and ROS runtime are not exercised by these checks.')


if __name__ == '__main__':
    main()
