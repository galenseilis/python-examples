import ciw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('Solarize_Light2')
ciw.seed(2018)

rate  = 10
ARRIVAL_TIME = rate
SERVICE_TIME = rate / 2
HORIZON = 1000

network = ciw.create_network(
    arrival_distributions = [ciw.dists.Exponential(ARRIVAL_TIME)],
    service_distributions = [ciw.dists.Exponential(SERVICE_TIME)],
    number_of_servers = [3]
    )

simulation = ciw.Simulation(network)
simulation.simulate_until_max_time(HORIZON)
records = pd.DataFrame(simulation.get_all_records())

records.plot.scatter(x='arrival_date', y='waiting_time')
plt.title('Avg. Demand < Avg. Capacity', c='gray')
plt.tight_layout()
plt.savefig('stable_rates_wait_time.png', dpi=300, transparent=True)
plt.close()

