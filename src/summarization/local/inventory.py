"""Inventory source files and flag cleanup candidates without deleting anything."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .settings import PROJECT_ROOT
SKIP = {'.git', '.local_tools', '.project_backups', '__pycache__', '.venv', 'venv',
        'node_modules', 'artifacts', 'data', '.pytest_cache', '.cache'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    files, groups = [], {}
    paths = []
    for directory, children, names in os.walk(root):
        children[:] = sorted(name for name in children if name not in SKIP)
        paths.extend(Path(directory) / name for name in names)
    for path in sorted(paths):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() in ('.zip', '.gguf', '.safetensors', '.pt', '.bin'):
            continue
        size = path.stat().st_size
        item = {'path': str(relative), 'bytes': size}
        if not size:
            item['note'] = 'Normal package marker' if path.name == '__init__.py' else 'Empty file; review before removing'
        if size <= 2_000_000 and path.suffix in ('.py', '.json', '.yaml', '.yml', '.md', '.txt', '.toml'):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(str(relative))
        files.append(item)
    report = {'files': files, 'exact_duplicate_groups': [v for v in groups.values() if len(v) > 1],
              'note': 'No files were moved or deleted. Empty __init__.py files are normal package markers.'}
    output = root / 'reports/project_inventory.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', output)
    print('Files inventoried:', len(files), '| No deletion performed.')


if __name__ == '__main__':
    main()
