"""Ask the Agent to repair invalid output; never substitute an executable policy."""
from __future__ import annotations
import json
from typing import Any
from .decision_rationale import valid_rationale_refs
from .agent_io import _compiled_schema_validator

def repair_rationale_envelope(raw: str) -> tuple[str, dict | None]:
    """Move a uniquely misplaced rationale; preserve every decision value.

    Only this unambiguous envelope defect is repaired locally. The result must
    still pass the complete live schema and semantic validator before execution.
    """
    try:
        payload=json.loads(raw)
    except (ValueError,TypeError):
        return raw,None
    if not isinstance(payload,dict) or set(payload)!={'step_decision','rationale'}:
        return raw,None
    step=payload['step_decision']
    if not isinstance(step,dict) or set(step)!={'action','control'} or not isinstance(payload['rationale'],dict):
        return raw,None
    fixed={'step_decision':{**step,'rationale':payload['rationale']}}
    return json.dumps(fixed,ensure_ascii=False),{'kind':'move_root_rationale_into_step_decision','original_raw':raw,'controls_unchanged':True}

def repair_messages(*, raw: str, error: str, schema: dict, observation: dict,
                    attempt: int, previous_errors: list[str], protected_decision: dict | None = None) -> list[dict[str,str]]:
    diagnostics: dict[str,Any]={'reported_error':error,'repair_number':attempt}
    # Diagnostic extraction only. This candidate is never compiled or executed.
    # The model must return its own full corrected response, checked by the normal validator.
    try:
        candidate,end=json.JSONDecoder().raw_decode(raw.lstrip())
        tail=raw.lstrip()[end:]
        diagnostics['decoded_candidate_for_inspection_only']=candidate
        if tail.strip():
            diagnostics['invalid_trailing_text']=tail
            diagnostics['syntax_hint']='One complete object already ends before this trailing text. Re-emit a single balanced JSON object.'
        validator=_compiled_schema_validator(json.dumps(schema,sort_keys=True,separators=(',',':')))
        diagnostics['additional_schema_errors']=[e.message for e in validator.iter_errors(candidate)][:8]
    except (ValueError,TypeError):
        diagnostics['syntax_hint']='The response could not be decoded. Reconstruct one complete JSON object from the live schema.'
    diagnostics['previous_errors']=previous_errors[-3:]
    if protected_decision is not None:
        diagnostics['original_legal_decision_must_be_preserved']=protected_decision
    diagnostics['valid_rationale_refs']=valid_rationale_refs(observation)
    system=(
        'You are the same search-control Agent performing output self-repair. Return exactly one valid JSON object. '
        'Repair formatting, schema violations and unsupported references. Preserve every already legal control choice; '
        'this is not a new search decision. If a control value itself is illegal, choose its valid replacement using '
        'the live schema and observation. No solver action will run until the entire response passes validation. '
        'The root contains only step_decision; action, control and rationale are inside that object. '
        'Use 1-2 concise sentences for reason. Copy refs verbatim from valid_rationale_refs; nested feedback fields '
        'such as removal_size_coverage belong to STEP03_feedback.search_evidence. Do not invent reference names. '
        'Fix ALL listed syntax and schema problems together. Never copy invalid trailing brackets. '
        'Do not output Markdown, comments, an explanation outside the JSON, or multiple objects.'
    )
    payload={'repair_diagnostics':diagnostics,'invalid_response':raw,'live_observation':observation,'output_json_schema':schema}
    return [{'role':'system','content':system},
            {'role':'user','content':'Repair this response and return JSON only:\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]


def legal_decision_for_repair(*, raw: str, schema: dict, observation: dict, contract):
    """Validate an extracted control for protection only; never return executable control.

    A diagnostic copy receives a neutral rationale so formatting/rationale defects
    do not erase the Agent's already legal search choice. This copy is never executed;
    only a complete fresh Agent response may pass the normal execution path.
    """
    from copy import deepcopy
    from .agent_io import AgentIOError, parse_validate_compile_step
    try:
        payload,_=json.JSONDecoder().raw_decode(raw.lstrip())
        step=payload['step_decision']
        protected={'action':step['action'],'control':deepcopy(step['control'])}
        diagnostic={'step_decision':{**protected,'rationale':{'reason':'Diagnostic validation only; never executed.','refs':[]}}}
        parse_validate_compile_step(raw_text=json.dumps(diagnostic),schema=schema,observation=observation,contract=contract)
        return protected
    except (ValueError,TypeError,KeyError,AgentIOError):
        return None
