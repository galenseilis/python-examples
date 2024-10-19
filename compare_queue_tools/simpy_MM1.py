import simpy
import random

arrival_rate = 10.0
number_of_servers = 3
service_rate = 4.0
max_simulation_time = 800
warmup = 100
num_trials = 20


def source(env, arrival_rate, service_rate, server, records):
    """Source generates customers randomly"""
    while True:
        c = customer(env, server, service_rate, records)
        env.process(c)
        t = random.expovariate(arrival_rate)
        yield env.timeout(t)


def customer(env, server, service_rate, records):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    with server.request() as req:
        results = yield req
        wait = env.now - arrive
        records.append((env.now, wait))
        tib = random.expovariate(service_rate)
        yield env.timeout(tib)


def run_trial(
    seed, arrival_rate, service_rate, number_of_servers, max_simulation_time, warmup
):
    """Run one trial of the simulation, returning the average waiting time"""
    random.seed(seed)
    records = []
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=number_of_servers)
    env.process(source(env, arrival_rate, service_rate, server, records))
    env.run(until=max_simulation_time)
    waiting_times = [r[1] for r in records if r[0] > warmup]
    return sum(waiting_times) / len(waiting_times)


mean_waits = [
    run_trial(
        s, arrival_rate, service_rate, number_of_servers, max_simulation_time, warmup
    )
    for s in range(num_trials)
]
average_waits = sum(mean_waits) / len(mean_waits)
