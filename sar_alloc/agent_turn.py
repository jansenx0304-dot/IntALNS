'Prompt rendering for the fixed single-agent search controller.'

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from .decision_examples import render_decision_examples
from .focus_feedback import FOCUS_FEEDBACK_REVISION, REVIEW_INSTRUCTION
from .decision_rationale import valid_rationale_refs
from .example_bank import large_examples, compact_initial_example
from . import energy_guidance, decision_consistency, large_guidance


@dataclass(frozen=True, slots=True)
class PromptBundle:
    role: str
    system_message: str
    instruction: str
    decision_examples: list[Dict[str, Any]]
    context: Dict[str, Any]
    output_json_schema: Dict[str, Any]
    user_message: str
    messages: list[dict[str, str]]


def system_prompt(*, examples_enabled: bool = True) -> str:
    return (
        "You control a task-assignment optimizer. Return exactly one raw JSON object "
        "matching the live schema. The first non-whitespace character must be { and the last must be }. "
        "Never use Markdown code fences such as ```json or add surrounding prose. The root must "
        "contain only the role envelope required by the schema. Treat the current observation "
        "as run-time facts, the decision catalog as static program semantics, and any retrieved "
        "examples that are present as strategy lessons rather than facts or value templates."
    )


def _common_instruction(
    *, examples_enabled: bool = True
) -> str:
    del examples_enabled
    lines = [
        "Protocol:",
        "- JSON structure: the ROOT has exactly one key, step_decision. INSIDE step_decision there are exactly action, control and rationale. rationale is a sibling of control INSIDE step_decision, never at the root. Check the closing braces before returning.",
        "- Energy in this arm is navigation rate times closed route distance, including departure and return to depot. Waiting and service duration affect time only; required skills affect eligibility only.",
        "- Read metric_order from the current state observation. It is fixed to missed_priority, then unassigned_count, then energy_total. Lower energy matters ONLY at equal service. Zero unassigned tasks starts energy refinement, not proof of global optimality. All acceptance modes protect the two service metrics; bounded exploration may temporarily increase energy but the output always retains the best full vector. Never invent, reorder, reweight, or scalarize these terms.",
        "- Use only controls exposed by the live schema. Observation values are run-time facts; catalog descriptions define program semantics; retrieved cases, when present, are reasoning demonstrations rather than current-run facts.",
        "- For a retrieved case, compare its state and judgment basis with the current observation before using it. Transfer the causal test, not its wording or exact control values; use the alternative when the stated failure condition matches better.",
        "- The rationale must describe the exact controls selected in the JSON. Never claim that a value or strategy changed unless the selected JSON actually differs from the cited recent strategy.",
    ]
    lines.append(
        "- Provide one concise action-level rationale={reason,refs}, with 1-2 sentences and no more than four refs copied exactly from the live legal_rationale_refs list. Nested search history fields all use STEP03_feedback.search_evidence; do not invent refs such as STEP03_feedback.removal_size_coverage. Rationale explains the decision and is not a control."
    )
    lines.append(
        "- Respect the remaining budget and return only the JSON object required by the schema."
    )
    return "\n".join(lines)




def step_instruction(
    *, examples_enabled: bool = True, focus_control: bool = True,
    focus_feedback_revision: str | None = None,
) -> str:
    lines = [
        "Step role:",
        "- There is no Supervisor. Both semantic feature vectors are disabled and the fixed action/run budgets cannot be changed.",
        "- Choose only Destroy/Repair priorities, remove ratio, destroy/task selectors and acceptance for the next fixed 100-trial action.",
        "- Keep operator priorities concise: prefer no more than three Destroy operators and no more than three Repair operators, selected only from the live legal choices.",
        "- Random TASK selection is separate from insertion POSITION choice. repair_priority=[best_insertion] is fully compatible with both random selectors: it inserts the randomly drawn task at the cheapest feasible position. Do not automatically add random_insertion to satisfy random selection. Adding it samples arbitrary feasible positions and changes the strategy. Compare its actual service/energy outcomes before retaining it.",
        "- Greedy protects all three objectives. Tolerant and simulated_annealing can cross an energy barrier at unchanged service, never worsen the service prefix. Use a small nonzero tolerance only when observed stagnation and rejected structural changes justify exploration; best output remains protected. Acceptance alone cannot cross a service barrier.",
        "- Search intensity is state dependent. Use the current residual service gap and the outcomes of actual removal sizes, not the initial maturity label. Both larger destruction releasing coupled blockers and smaller destruction preserving useful structure are hypotheses to check. Read the legal remove_ratio_options; do not anchor every action to a value in an example.",
    ]
    if focus_control:
        lines[1] = "- There is no Supervisor. Both semantic feature vectors are disabled and the fixed action/run budgets cannot be changed."
        lines[2] = "- Choose one live focus_id plus Destroy/Repair priorities, remove ratio, the exposed random selectors, and acceptance for the next fixed 100-trial action. Global is uniform random; service_bottleneck applies the existing target sampling bias described in the catalog. It is not a separate candidate pool or a feature score."
    lines.append("- Use current-run feedback when available and adapt one interpretable low-level hypothesis at a time when possible. If the exact strategy's marginal credit is declining or absent, either change one selected control or explicitly justify why repeating it remains the better test.")
    lines.append("- During service_recovery, use service_improved and current_service_gap_action_indices to diagnose residual stagnation even when energy decreases. Energy-only improvement is useful but does not justify indefinitely keeping a control that repeatedly leaves the same tasks unassigned. During energy_refinement, use energy_improved_at_equal_service and objective stagnation instead. Look at working-versus-best energy when evaluating tolerant exploration; final output retains the best vector.")
    lines.append("- acceptance_evidence.worsening_by_metric distinguishes rejected service loss from energy barriers. When service stalls and feasible candidates are rejected on ENERGY, tolerant acceptance at a tested small tolerance is a meaningful one-window probe while retaining the same promising neighborhood. At equal service, permitting some temporary extra travel can enable later task recovery; global best is never lost. A low structural passage rate alone does not establish the reason: consult the metric counts. If exploration fails, keep the negative evidence and compare another control; do not repeatedly restart an unproductive greedy example.")
    lines.append("- Read control_history as the authoritative sequence of complete controls and observed before/after objectives. current_gap_action_indices contains ONLY windows that started at the current best vector. Do not claim that all previously used sizes were tested at this gap. An improvement resets current-gap counts but is not a reason to abandon the productive control or mechanically cycle through untried sizes. Prefer an interpretable continuation while progress remains credible; on repeated stalls compare changing the leading destroy family, intensity, repair mix or focus using their actual outcomes.")
    lines.append("- Inspect actual_destroy_counts_at_current_state: the unchanged solver caps the ratio base at 100 assigned tasks, so .30 at T300 removes about 30 tasks, not 90. Zero insertability describes the current route before destruction, not every future partial route. High no-op with best-only repair may indicate reconstruction of the same neighborhood; a different leading spatial/time/random family can be more informative than repeatedly changing size in the same family. These are hypotheses, not a required policy. Preserve negative evidence and never attribute cumulative early wins to the current residual state.")
    lines.append("- On large instances, distinguish reaching residual tasks from releasing their assigned blockers. If two or more windows at the same service gap fail, consult service_gap_neighborhood_trials and service_gap_remove_ratio_trials before repeating an earlier control. Switching global/service_bottleneck back and forth with the same destruction and repair does not test a new removal neighborhood. If only .20/.30 were tried, a .40 or .50 neighborhood is a concrete alternative for releasing coupled blockers, subject to live feasibility and earlier outcomes. Prefer changing intensity while retaining a promising family before simultaneously replacing everything. A larger size is a hypothesis, not a guaranteed improvement or a fixed schedule.")
    lines.append("- Acceptance counts are measured at candidate level. A rejected L1/L2 loss cannot be repaired by raising energy tolerance. When service rejection dominates despite energy exploration, test a different removal/repair neighborhood instead of increasing tolerance. If energy-only progress persists during service recovery, use the unchanged service-gap history, not reset exact-vector history. Continue an improving service control; after unsuccessful probes retain the failures and avoid calling them untried.")
    instruction = _common_instruction(examples_enabled=examples_enabled) + "\n\n" + "\n".join(lines)
    if focus_feedback_revision == FOCUS_FEEDBACK_REVISION:
        instruction += "\n" + REVIEW_INSTRUCTION
    return instruction




def build_step_prompt_bundle(
    *,
    user_goal: str,
    decision_catalog: Dict[str, Any],
    observation: Dict[str, Any],
    schema: Dict[str, Any],
    max_examples: int = 1,
    examples_enabled: bool = True,
) -> PromptBundle:
    revision = decision_catalog.get("focus_feedback_revision")
    example_count = max(0, min(2, int(max_examples))) if examples_enabled else 0
    large_case = large_guidance.example(observation)
    examples = ([large_case] if large_case is not None else large_examples(observation))[:example_count]
    compact_case = compact_initial_example(observation)
    if compact_case is not None:
        examples = [compact_case][:example_count]
    instruction = step_instruction(examples_enabled=examples_enabled, focus_control=bool(decision_catalog.get("step_focus_control")), focus_feedback_revision=revision)
    if energy_guidance.applies(observation):
        examples = [energy_guidance.example(observation)][:example_count]
        instruction += "\n" + energy_guidance.INSTRUCTION + "\n" + decision_consistency.INSTRUCTION
        observation, schema, examples = decision_consistency.prepare(observation, schema, examples)
    return _bundle(
        role="STEP",
        instruction=instruction,
        decision_examples=examples,
        context={"user_goal": user_goal, "decision_catalog": decision_catalog, "observation": observation,
                 "legal_rationale_refs":valid_rationale_refs(observation)},
        schema=schema,
    )






def _bundle(
    *,
    role: str,
    instruction: str,
    decision_examples: list[Dict[str, Any]],
    context: Dict[str, Any],
    schema: Dict[str, Any],
) -> PromptBundle:
    system_message = system_prompt(examples_enabled=bool(decision_examples))
    example_section = (
        f"RETRIEVED STRATEGY CASES (adapt, do not copy):\n{render_decision_examples(decision_examples)}\n\n"
        if decision_examples
        else ""
    )
    user_message = (
        f"ROLE: {role}\n\n"
        f"INSTRUCTION:\n{instruction}\n\n"
        f"{example_section}"
        f"CONTEXT:\n{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"OUTPUT JSON SCHEMA:\n{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
    )
    return PromptBundle(
        role=role,
        system_message=system_message,
        instruction=instruction,
        decision_examples=decision_examples,
        context=context,
        output_json_schema=schema,
        user_message=user_message,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
