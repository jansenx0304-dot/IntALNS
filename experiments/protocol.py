"""正式实例与输出目录。"""
import json
from sar_alloc.paths import PROJECT_ROOT

SPLITS = ('test',)

def load_protocol():
    return json.loads((PROJECT_ROOT/'configs/protocol.json').read_text(encoding='utf-8'))

def data_path(split, protocol=None):
    return PROJECT_ROOT/(protocol or load_protocol())['dataset_roots'][split]

def result_path(split):
    return PROJECT_ROOT/'outputs'/split/'seed_0'
