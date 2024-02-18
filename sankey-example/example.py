import ciw
import pandas as pd
import plotly.graph_objects as go

ciw.seed(2018)

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

Q.simulate_until_max_time(200)

recs = pd.DataFrame(Q.get_all_records())


first_nodes = (
    recs.sort_values(by="arrival_date")
    .groupby("id_number")["node"]
    .apply(lambda x: x.iloc[0])
    .value_counts()
    .reset_index(name="flow")
    .rename(columns={"index": "destination"})
    .assign(node=0)
)

recs = (
        recs
        .groupby(by=["node", "destination"])
        .size()
        .reset_index(name="flow")
        .replace({-1:4})
        )


recs = pd.concat((first_nodes, recs))


recs = recs.sort_values(by=['destination', 'node', 'flow'])

fig = go.Figure(
            go.Sankey(
                    arrangement='snap',
                    node=dict(
                        label=['ArrivalNode', 'ColdFood', 'HotFood', 'Till', 'Exit'],
                        pad=10
                        ),
                    link=dict(
                        arrowlen=15,
                        source=recs.node,
                        target=recs.destination,
                        value=recs.flow
                        )
                )
        )

fig.layout.paper_bgcolor = 'rgba(0.5,0.5,0.5,0.5)'
fig.layout.plot_bgcolor = 'rgba(0.5,0.5,0.5,0.5)'

print(fig.to_html(full_html=False, include_plotlyjs='cdn'))

