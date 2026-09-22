"""按逐记录来源清单核验发布文件。"""
import hashlib
import json
from pathlib import Path


def record_matches_manifest(record, path, root):
    root = Path(root)
    manifest = root/'results/provenance.json'
    if not manifest.exists():
        return False
    evidence = json.loads(manifest.read_text(encoding='utf-8'))
    relative = Path(path).relative_to(root).as_posix()
    row = evidence['records'].get(relative)
    return bool(row and row['method'] == record['method']
                and row['origin'] in {'input_equivalent_reuse', 'direct_run'}
                and row['input_equivalent']
                and row['sha256'] == hashlib.sha256(Path(path).read_bytes()).hexdigest())
