"""渲染正式检索示例。"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping







def render_decision_examples(examples: list[Dict[str, Any]]) -> str:
    return json.dumps(examples, ensure_ascii=False, indent=2)




