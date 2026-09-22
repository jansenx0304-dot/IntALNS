"""Verify original, unmodified record metadata when equivalent inputs are reused."""
import hashlib,json
from pathlib import Path

def information_matches(record,path,protocol,root):
    if record['method']!='agent':return True
    if record['config']['information_revision']==protocol['information_revision']:return True
    manifest=Path(root)/'results/provenance.json'
    if not manifest.exists():return False
    evidence=json.loads(manifest.read_text(encoding='utf-8'))
    relative=Path(path).relative_to(Path(root)).as_posix()
    row=evidence['records'].get(relative)
    return bool(row and row['input_equivalent'] and row['method']=='agent'
        and row['original_information_revision']==record['config']['information_revision']
        and row['target_information_revision']==protocol['information_revision']
        and row['sha256']==hashlib.sha256(Path(path).read_bytes()).hexdigest()
        and row['reuse_basis']=='V15 retained at T50/T100 and T300 M90; prompt dependency hashes and conditional retrieval checked'
        and (record['n_tasks'] in [50,100] or record['n_tasks']==300 and record['maturity']=='M90'))
