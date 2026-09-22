"""One unified, reference-feasible conference scenario; one draw per seed.

Adapted directly from the parent route-based generator. No rejection by solver
performance, no instance-type branches, and no maturity-specific budgets.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any
from sar_alloc.config import Config
from sar_alloc.evaluator import evaluate
from sar_alloc.models import Agent, Depot, Instance, Task
from sar_alloc.operators.route_metrics import simulate_route
from sar_alloc.paths import PROJECT_ROOT
from sar_alloc.solution import AssignmentSolution

def _agent_templates() -> list[dict[str, Any]]:
    skills = [
        {"s0", "s1", "s2"},
        {"s0", "s2", "s3"},
        {"s1", "s2", "s3"},
        {"s0", "s1"},
        {"s2", "s3"},
        {"s0", "s1", "s2", "s3"},
    ]
    speeds = [1.16, 1.03, 1.24, 0.96, 1.08, 1.12]
    travel = [0.96, 1.05, 1.18, 0.90, 1.12, 1.02]
    return [
        {
            "id": i,
            "skills": skills[i],
            "speed": speeds[i],
            "travel_energy_rate": travel[i],
        }
        for i in range(6)
    ]


def _coverage_count(requirement: set[str], templates: list[dict[str, Any]]) -> int:
    return sum(1 for row in templates if requirement.issubset(row["skills"]))


def _requirement_candidates(
    ref: dict[str, Any], templates: list[dict[str, Any]]
) -> list[tuple[set[str], int]]:
    skills = sorted(ref["skills"])
    candidates: list[tuple[set[str], int]] = [(set(), len(templates))]
    for size in (1, 2):
        for combo in itertools.combinations(skills, size):
            req = set(combo)
            candidates.append((req, _coverage_count(req, templates)))
    return candidates


def _choose_requirement(
    rng: random.Random,
    ref: dict[str, Any],
    templates: list[dict[str, Any]],
    rare_fraction: float,
) -> tuple[set[str], int, bool]:
    candidates = _requirement_candidates(ref, templates)
    rare = [(req, count) for req, count in candidates if req and count <= 2]
    common = [(req, count) for req, count in candidates if count >= 3]
    choose_rare = bool(rare and rng.random() < rare_fraction)
    pool = rare if choose_rare else (common or candidates)
    req, count = rng.choice(pool)
    return set(req), int(count), choose_rare


def _route_lengths(n_tasks: int, n_agents: int) -> list[int]:
    base, rem = divmod(n_tasks, n_agents)
    return [base + (1 if i < rem else 0) for i in range(n_agents)]


def _task_position(
    *,
    rng: random.Random,
    aid: int,
    local_index: int,
    route_len: int,
    cross_route: bool,
    jitter: float,
) -> tuple[float, float]:
    target_aid = (aid + rng.choice([-1, 1])) % 6 if cross_route else aid
    base_angle = 2.0 * math.pi * target_aid / 6.0
    frac = (local_index + 1) / (route_len + 1)
    radius = 280.0 + 980.0 * frac + rng.uniform(-90.0, 90.0)
    angle = base_angle + rng.gauss(0.0, jitter)
    if local_index % 7 == 0:
        radius *= rng.uniform(0.50, 0.75)
    return radius * math.cos(angle), radius * math.sin(angle)


def _priority(rng: random.Random, rare: bool, boost: float) -> float:
    base = rng.choices([0.0, 1.0, 2.0, 3.0], weights=[5, 24, 38, 33], k=1)[0]
    return 3.0 if rare and rng.random() < boost else float(base)


def _build_instance_once(
    n_tasks: int, seed: int, spec: dict[str, Any]
) -> tuple[Instance, dict[int, list[int]], dict[str, Any]]:
    draw_seed = (
        9_973 * (seed + 1)
        + 101 * n_tasks
        + 271_828
    )
    rng = random.Random(draw_seed)
    templates = _agent_templates()
    depot = Depot(id=0, loc=(0.0, 0.0))
    route_lengths = _route_lengths(n_tasks, len(templates))
    routes: dict[int, list[int]] = {i: [] for i in range(len(templates))}
    raw_tasks: dict[int, dict[str, Any]] = {}
    capability_counts: list[int] = []
    rare_flags: list[bool] = []
    task_id = 0

    for aid, route_len in enumerate(route_lengths):
        ref = templates[aid]
        for local_index in range(route_len):
            cross = rng.random() < float(spec["cross_route_fraction"])
            x, y = _task_position(
                rng=rng,
                aid=aid,
                local_index=local_index,
                route_len=route_len,
                cross_route=cross,
                jitter=float(spec["spatial_jitter"]),
            )
            req, count, rare = _choose_requirement(
                rng, ref, templates, float(spec["rare_fraction"])
            )
            raw_tasks[task_id] = {
                "id": task_id,
                "x": float(x),
                "y": float(y),
                "service_time": float(rng.uniform(35.0, 145.0)),
                "skill_req": sorted(req),
                "priority": _priority(
                    rng, rare, float(spec["priority_rare_boost"])
                ),
                "cross_route": bool(cross),
            }
            routes[aid].append(task_id)
            capability_counts.append(count)
            rare_flags.append(rare)
            task_id += 1

    provisional_agents = tuple(
        Agent(
            id=row["id"],
            init_energy=1e12,
            skills=set(row["skills"]),
            speed=row["speed"],
            travel_energy_rate=row["travel_energy_rate"],
        )
        for row in templates
    )

    for aid, tids in routes.items():
        agent = provisional_agents[aid]
        cur_time = 0.0
        cur_loc = depot.loc
        for tid in tids:
            row = raw_tasks[tid]
            loc = (row["x"], row["y"])
            arrival = cur_time + math.hypot(
                loc[0] - cur_loc[0], loc[1] - cur_loc[1]
            ) / agent.speed
            wait_lo, wait_hi = spec["planned_wait"]
            planned_start = arrival + rng.uniform(float(wait_lo), float(wait_hi))
            half = float(spec["window_half"])
            early = half
            late = half
            row["tw_start"] = max(0.0, planned_start - early)
            row["tw_end"] = planned_start + late
            cur_time = planned_start + row["service_time"]
            cur_loc = loc

    tasks = tuple(
        Task(
            id=tid,
            loc=(row["x"], row["y"]),
            tw_start=row["tw_start"],
            tw_end=row["tw_end"],
            service_time=row["service_time"],
            skill_req=set(row["skill_req"]),
            priority=row["priority"],
        )
        for tid, row in sorted(raw_tasks.items())
    )
    provisional = Instance(
        tasks=tasks,
        agents=provisional_agents,
        depot=depot,
        default_speed=1.0,
    )
    cfg = Config(rng_seed=0)
    reference_energy: dict[int, float] = {}
    final_agents: list[Agent] = []
    for row in templates:
        aid = int(row["id"])
        energy = float(simulate_route(provisional, cfg, aid, routes[aid]).energy)
        reference_energy[aid] = energy
        margin = float(spec["energy_margin"]) * rng.uniform(0.985, 1.025)
        final_agents.append(
            Agent(
                id=aid,
                init_energy=max(1.0, energy * margin),
                skills=set(row["skills"]),
                speed=row["speed"],
                travel_energy_rate=row["travel_energy_rate"],
            )
        )
    instance = Instance(
        tasks=tasks,
        agents=tuple(final_agents),
        depot=depot,
        default_speed=1.0,
    )
    reference = AssignmentSolution(
        routes={aid: list(tids) for aid, tids in routes.items()}, unassigned=set()
    )
    ref_eval = evaluate(reference, instance, cfg, update_solution_schedule=True)
    if not ref_eval.is_feasible or ref_eval.quality_metrics["unassigned_count"] != 0:
        raise RuntimeError(
            f"reference route is not feasible: {n_tasks=} {seed=}"
        )

    metadata = {
        "draw_seed": int(draw_seed),
        "reference_quality": {
            k: float(v) for k, v in ref_eval.quality_metrics.items()
        },
        "reference_energy_by_agent": {
            str(k): float(v) for k, v in reference_energy.items()
        },
        "capability_count_min": int(min(capability_counts)),
        "capability_count_median": float(
            sorted(capability_counts)[len(capability_counts) // 2]
        ),
        "rare_task_fraction_realized": float(sum(rare_flags) / len(rare_flags)),
        "cross_route_fraction_realized": float(
            sum(1 for r in raw_tasks.values() if r["cross_route"]) / n_tasks
        ),
        "time_window_width_mean": float(
            sum(t.tw_end - t.tw_start for t in tasks) / len(tasks)
        ),
        "energy_margin_nominal": float(spec["energy_margin"]),
    }
    return instance, routes, metadata


def _instance_payload(instance: Instance) -> dict[str, Any]:
    return {
        "depot": {
            "id": instance.depot.id,
            "x": instance.depot.loc[0],
            "y": instance.depot.loc[1],
        },
        "agents": [
            {
                "id": a.id,
                "init_energy": a.init_energy,
                "skills": sorted(a.skills),
                "speed": a.speed,
                "travel_energy_rate": a.travel_energy_rate,
            }
            for a in instance.agents
        ],
        "tasks": [
            {
                "id": t.id,
                "x": t.loc[0],
                "y": t.loc[1],
                "tw_start": t.tw_start,
                "tw_end": t.tw_end,
                "service_time": t.service_time,
                "skill_req": sorted(t.skill_req),
                "priority": t.priority,
            }
            for t in instance.tasks
        ],
        "default_speed": instance.default_speed,
    }



def generate(split: str, config: Path) -> list[dict[str, Any]]:
    protocol = json.loads(config.read_text(encoding="utf-8"))
    from experiments.protocol import data_path
    if split == 'test' and protocol['status'] != 'frozen':
        raise RuntimeError('Freeze both methods before generating the formal test data')
    root = data_path(split, protocol)
    if root.exists():
        raise FileExistsError(f"Refuse to overwrite existing dataset: {root}")
    root.mkdir(parents=True)
    manifest, starts = [], []
    for size in protocol.get('split_sizes', {}).get(split, protocol["sizes"]):
        for seed in protocol["instance_seeds"][split]:
            instance, routes, metadata = _build_instance_once(size, seed, protocol["generation"])
            case_id = f"{split}_T{size}_s{seed}"
            path = root / f"{case_id}.json"
            path.write_text(json.dumps(_instance_payload(instance), indent=2), encoding="utf-8", newline="\n")
            manifest.append({"case_id":case_id, "split":split, "n_tasks":size,
                "instance_seed":seed, "instance_path":path.relative_to(PROJECT_ROOT).as_posix(),
                "reference_routes":routes, "generation":metadata})
            order = list(instance.all_task_ids())
            # One permutation for all nested starts, independent of algorithm RNG.
            start_seed = 31337 * (seed + 1) + 1009 * size
            random.Random(start_seed).shuffle(order)
            for fraction in protocol["initial_fractions"]:
                assigned = set(order[:round(size * fraction)])
                solution = AssignmentSolution(
                    routes={aid:[tid for tid in route if tid in assigned] for aid,route in routes.items()},
                    unassigned=set(order)-assigned)
                evaluation = evaluate(solution, instance, Config(), update_solution_schedule=True)
                if not evaluation.is_feasible:
                    raise RuntimeError(f"Infeasible nested start: {case_id}, {fraction}")
                starts.append({"case_id":case_id, "fraction":fraction,
                    "maturity":f"M{round(100*fraction)}", "start_seed":start_seed,
                    "solution":{"routes":solution.routes,"unassigned":sorted(solution.unassigned)},
                    "quality":{k:evaluation.quality_metrics[k] for k in ("missed_priority","unassigned_count","energy_total")}})
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8",newline="\n")
    (root/"starts.json").write_text(json.dumps(starts,indent=2),encoding="utf-8",newline="\n")
    (root/"generation_config.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8",newline="\n")
    print(f"Generated {len(manifest)} instances and {len(starts)} nested starts in {root}")
    return manifest

def main():
    parser=argparse.ArgumentParser()
    from experiments.protocol import SPLITS
    parser.add_argument("--split",choices=SPLITS,required=True)
    parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/protocol.json")
    args=parser.parse_args()
    generate(args.split,args.config)

if __name__ == "__main__": main()
