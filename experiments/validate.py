"""Independent route-level validation; does not call the search evaluator."""
import math


def recheck(instance, solution):
    tasks = {task.id: task for task in instance.tasks}
    assigned = []
    routes = solution['routes']
    expected_agents = {str(agent.id) for agent in instance.agents}
    if not {str(key) for key in routes}.issubset(expected_agents):
        raise ValueError('Unknown platform in routes')
    details = []
    for agent in instance.agents:
        route = routes.get(str(agent.id), routes.get(agent.id, []))
        loc = instance.depot.loc
        time = 0.0
        distance = 0.0
        waiting = 0.0
        min_slack = None
        for tid in route:
            task = tasks[tid]
            assigned.append(tid)
            if not task.skill_req.issubset(agent.skills):
                raise ValueError(f'Capability violation: platform {agent.id}, task {tid}')
            leg = math.dist(loc, task.loc)
            distance += leg
            arrival = time + leg / agent.speed
            start = max(arrival, task.tw_start)
            waiting += start-arrival
            if start > task.tw_end + 1e-7:
                raise ValueError(f'Time-window violation: platform {agent.id}, task {tid}')
            slack = task.tw_end-start
            min_slack = slack if min_slack is None else min(min_slack,slack)
            time = start + task.service_time
            loc = task.loc
        if route:
            distance += math.dist(loc, instance.depot.loc)
        energy = agent.travel_energy_rate*distance
        if energy > agent.init_energy+1e-7:
            raise ValueError(f'Closed-route energy violation: platform {agent.id}')
        details.append({'agent_id':agent.id,'energy':energy,'energy_budget':agent.init_energy,
            'utilization':energy/agent.init_energy,'waiting_time':waiting,'minimum_window_slack':min_slack})
    if len(assigned) != len(set(assigned)):
        raise ValueError('Duplicate assigned task')
    unassigned = set(tasks)-set(assigned)
    declared = solution['unassigned']
    if set(declared) != unassigned or len(declared) != len(unassigned):
        raise ValueError('Unassigned task set does not complement the routes')
    return {'missed_priority':sum(tasks[i].priority for i in sorted(unassigned)),
            'unassigned_count':len(unassigned), 'energy_total':math.fsum(row['energy'] for row in details), 'routes':details}
