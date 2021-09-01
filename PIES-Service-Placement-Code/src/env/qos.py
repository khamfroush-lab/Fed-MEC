from typing import Any, Dict

from .environment import Environment

def delay_comp(user: int, service: int, model: int, env: Environment) -> float:
    if env.services[service][model].get("comp_delay", None) is not None:
        return env.services[service][model]["comp_delay"]

    edge = env.covering_edge(user)
    num = env.computation_cost(service, model) * len(env.covered_requests(edge))
    den = env.computation_capacity(edge)
    return num/den

def delay_tran(user: int, service: int, model: int, env: Environment) -> float:
    edge = env.covering_edge(user)
    num = env.communication_cost(service, model) * len(env.covered_requests(edge))
    den = env.communication_capacity(edge)
    return num/den

def delay_fn(user: int, service: int, model: int, env: Environment) -> float:
    return delay_tran(user, service, model, env) + delay_comp(user, service, model, env)

def quality_of_accuracy(u: int, s: int, m: int, env: Environment) -> float:
    accuracy = env.accuracy(s, m)
    req_accuracy = env.req_accuracy(u)
    if accuracy >= req_accuracy:
        qoa = 1.0
    else:
        qoa = max(0, 1 - (req_accuracy-accuracy))
    assert 0.0 <= qoa <= 1.0
    return qoa

def quality_of_delay(u: int, s: int, m: int, env: Environment) -> float:
    delay = delay_fn(u, s, m, env)
    req_delay = env.req_delay(u)
    if delay <= req_delay:
        qod = 1.0
    else:
        qod = max(0, 1 - min(delay-req_delay, env.max_delay)/env.max_delay)
    assert 0.0 <= qod <= 1.0
    return qod

def QoS_coeff(user: int, service: int, model: int, env: Environment) -> float:
    if service == env.req_service(user):
        qoa = quality_of_accuracy(user, service, model, env)
        qod = quality_of_delay(user, service, model, env)
        qos = 0.5 * (qoa + qod)
        assert 0.0 <= qos <= 1.0
        return qos
    else:
        return 0.0

def QoS_objective(x: Dict[Any, int], y: Dict[Any, int], env: Environment) -> float:
    return sum(
        y.get((u,m),0) * QoS_coeff(u,env.req_service(u),m,env)
        for e in env.edges
        for u in env.covered_requests(e)
        for m in env.models_for_request(u)
        if x.get((e,env.req_service(u),m), 0) == 1
    )

def num_req_satisfied(env: Environment, y: Dict[Any, int]) -> int:
        number = 0
        for (u, m) in y:
            s = env.req_service(u)
            if y[u,m] != 1:
                continue
            elif env.accuracy(s, m) >= env.req_accuracy(u) and \
                 delay_fn(u, s, m, env) <= env.req_delay(u):
                 number += 1
        return number

def num_partially_req_satisfied(env: Environment, y: Dict[Any, int]) -> int:
        number = 0
        for (u, m) in y:
            s = env.req_service(u)
            if y[u,m] != 1:
                continue
            elif env.accuracy(s, m) >= env.req_accuracy(u) or \
                 delay_fn(u, s, m, env) <= env.req_delay(u):
                 number += 1
        return number