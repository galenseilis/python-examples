import ciw

max_simulation_time = 800
warmup = 100
num_trials = 20

N = ciw.create_network(
    arrival_distributions=[ciw.dists.Exponential(10.0)],
    service_distributions=[ciw.dists.Exponential(4.0)],
    number_of_servers=[3],
)


def run_trial(s, max_simulation_time, warmup):
    """Run one trial of the simulation, returning the average waiting time"""
    ciw.seed(s)
    Q = ciw.Simulation(N)
    Q.simulate_until_max_time(max_simulation_time)
    recs = Q.get_all_records()
    waits = [r.waiting_time for r in recs if r.arrival_date > warmup]
    return sum(waits) / len(waits)


mean_waits = [run_trial(s, max_simulation_time, warmup) for s in range(num_trials)]
average_waits = sum(mean_waits) / len(mean_waits)
