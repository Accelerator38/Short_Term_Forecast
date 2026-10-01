import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import pairwise_distances

from ..preprocessing.connections_processing import WellsConnectionsGraph


def wells_connections_graph(wcg: WellsConnectionsGraph):
    layout = form_layout(title='Wells Connections Graph')
    verts = [go.Scatter(
        x=wcg.verts.X,
        y=wcg.verts.Y,
        text=wcg.verts.index,
        hoverinfo='text',
        mode='markers',
        marker=go.Marker(
            size=15,
            color='rgb(77, 26, 0)'
        )
    )]

    edges = []
    for n1, n2 in wcg.edges:
        edge = go.Scatter(
            x=[wcg.verts.iloc[n1].X, wcg.verts.iloc[n2].X],
            y=[wcg.verts.iloc[n1].Y, wcg.verts.iloc[n2].Y],
            mode='lines',
            line=dict(
                color=f'rgb(255, 102, 26)',
                width=2.5
            )
        )
        edges.append(edge)
    return go.Figure(data=edges + verts, layout=layout)


def wells_clustered_positions(coordinates, production, injection, clusters):
    layout = form_layout()

    prod_nodes = []
    inj_nodes = []

    # cmap = []
    for cl in np.unique(clusters['cluster']):
        wells = clusters.loc[clusters['cluster'] == cl].index
        clust_color = 'rgb(%s)' % ', '.join((255 * np.random.rand(3)).astype(str))
        prod_nodes_cl = form_scatter(coordinates.loc[wells], production[[w for w in wells if w in production.columns]],
                                       'PROD', clust_color, 'rgb(143, 9, 9)', coef=.5, bw=4)
        prod_nodes.extend(prod_nodes_cl)
        inj_nodes_cl = form_scatter(coordinates.loc[wells], injection[[w for w in wells if w in injection.columns]],
                                      'INJ', clust_color, 'rgb(9, 9, 143)', coef=.08, bw=4)
        inj_nodes.extend(inj_nodes_cl)
        for node in [*prod_nodes_cl, *inj_nodes_cl]:
            node.text += f'<br>CLUSTER #{cl}'

    data = prod_nodes + inj_nodes
    return go.Figure(data=data, layout=layout)


def wells_positions(coordinates, production, injection):
    layout = form_layout()

    prod_nodes = form_scatter(coordinates, production, 'PROD', 'rgba(240, 20, 20, 1)', 'rgb(143, 9, 9)', coef=.5)
    inj_nodes = form_scatter(coordinates, injection, 'INJ', 'rgba(15, 127, 239, 1)', 'rgb(9, 9, 143)', coef=.08)
    data = prod_nodes + inj_nodes

    return go.Figure(data=data, layout=layout)


def wells_connectivities(coordinates, f, production, injection, coef=1.):
    layout = form_layout()

    max_prod, max_inj = np.max(production), np.max(injection)
    prw_dist = pairwise_distances(coordinates, coordinates)
    min_dist = np.min(prw_dist[np.nonzero(prw_dist)]) / 4

    prod_nodes = form_scatter(coordinates, production, 'PROD', 'rgba(240, 20, 20, 1)', 'rgb(143, 9, 9)', coef=coef*min_dist/max_prod)
    inj_nodes = form_scatter(coordinates, injection, 'INJ', 'rgba(15, 127, 239, 1)', 'rgb(9, 9, 143)', coef=coef*min_dist/max_inj)

    edges = []

    for i in enumerate(injection.columns):
        for p in enumerate(production.columns):
            if f[i[0], p[0]] != 0:
                edge = go.Scatter(
                    x=[coordinates.loc[p[1], 'X'], coordinates.loc[i[1], 'X']],
                    y=[coordinates.loc[p[1], 'Y'], coordinates.loc[i[1], 'Y']],
                    text=f'{f[i[0], p[0]]}',
                    hoverinfo='text',
                    mode='lines',
                    line=dict(
                        color=f'rgba(0, 0, 0, 1',  # {-(f[i[0], d[0]])**2 + 1}',
                        width=int(10 * f[i[0], p[0]]),
                    )
                )
                edges.append(edge)

    data = prod_nodes + inj_nodes + edges
    return go.Figure(data=data, layout=layout)


def form_scatter(coordinates, debit, type, point_color, border_color, coef=.5, bw=2):
    nodes = []
    for w in list(enumerate(debit.columns)):
        avr_deb = np.average(debit.loc[:, w[1]])
        node = go.Scatter(
            x=[coordinates.loc[w[1]].X],
            y=[coordinates.loc[w[1]].Y],
            text=f'{type} WELL #{w[1]}<br>{type} = {avr_deb:.2f}',
            hoverinfo='text',
            mode='markers',
            marker=go.Marker(
                size=coef * avr_deb,
                color=point_color,
                line=dict(
                    width=bw,
                    color=border_color
                )
            )
        )
        nodes.append(node)
    return nodes


def form_layout(title='Wells map'):
    return go.Layout(
        font=go.Font(
            family='Times New Roman',
            size=17
        ),
        title=f'Карта влияний скважин',
        titlefont=dict(size=20),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=go.XAxis(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=go.YAxis(showgrid=True, zeroline=False, showticklabels=False))