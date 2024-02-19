import ciw

N = ciw.create_network(
    arrival_distributions=[
        ciw.dists.Exponential(rate=0.3),
        ciw.dists.Exponential(rate=0.2),
        None,
    ],
    service_distributions=[
        ciw.dists.Exponential(rate=1.0),
        ciw.dists.Exponential(rate=0.4),
        ciw.dists.Exponential(rate=0.5),
    ],
    routing=[[0.0, 0.3, 0.7], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
    number_of_servers=[1, 2, 2],
)

Q = ciw.Simulation(N)
Q.simulate_until_max_time(10)
recs = Q.get_all_records()
