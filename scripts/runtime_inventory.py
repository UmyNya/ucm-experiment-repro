#!/usr/bin/env python3
"""Record import resolution and installed UCM hashes without importing torch/UCM."""
import importlib.metadata
import importlib.util
from pathlib import Path
import platform
import sys

from common import save_json, sha256


def inventory():
    spec = importlib.util.find_spec('ucm')
    if spec is None or not spec.origin:
        raise RuntimeError('The selected service Python cannot find UCM')
    root = Path(spec.origin).resolve().parent
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
             if p.is_file() and (p.suffix == '.py' or '.so' in p.name)}
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name', '')
        if name:
            packages[name] = dist.version
    return {'python_executable': sys.executable, 'python_prefix': sys.prefix,
            'python_version': platform.python_version(), 'machine': platform.machine(),
            'ucm_root': str(root), 'ucm_files': files, 'packages': packages}


if __name__ == '__main__':
    save_json(Path(sys.argv[1]), inventory())
