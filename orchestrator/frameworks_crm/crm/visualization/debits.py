from collections import defaultdict
import pandas as pd
import plotly.graph_objs as go
from functools import partial

def comparive_report(production, bhp, metrics, predictions, max_pres=300):
    figs = []
    metric_results = defaultdict(partial(defaultdict, dict))
    for i, p in enumerate(production.columns):
        layout_not_def = True

        trace_true = go.Scatter(
            x=production.index,
            y=production.iloc[:, i],
            name='Истинное значние',
            mode='lines',
            line=dict(color='gray', width=3)
        )

        trace_bhp = go.Scatter(
            x=bhp.index,
            y=bhp.iloc[:, i],
            name='Забойное давление',
            line=dict(
                dash='dot',
                color='black',
            ),
            yaxis='y2'
        )

        well_traces = [trace_true, trace_bhp]
        for model in predictions:
            pred_train, pred_test = predictions[model][:2]

            if layout_not_def:
                layout = get_layout(pred_train.index, pred_test.index, max_pres)
                layout['title'] = f'Скважина {p}.'
                layout_not_def = False

            metric_records = []
            for name, func in metrics.items():
                val = func(production.iloc[:, i].loc[pred_test.index], pred_test.iloc[:, i])
                metric_results[model][name][p] = val
                metric_records.append(f'{name}={val:.2f}')

            prediction = pd.concat([pred_train, pred_test], axis=0)
            prediction = prediction[~prediction.index.duplicated(keep='first')]
            trace_model = go.Scatter(
                x=prediction.index,
                y=prediction.iloc[:, i],
                name=f'{model}, ' + ', '.join(metric_records) + '.',
                mode='lines'
            )
            well_traces.append(trace_model)

        figs.append(go.Figure(data=well_traces, layout=layout))
        metric_results_df = pd.concat([pd.DataFrame(m) for m in metric_results.values()], keys=metric_results.keys())
    return figs, metric_results_df


def production_summary_omega(production, pred_train, pred_test, bhp, cp, metrics, model_name='', max_pres=300):
    figs, metric_results = production_summary(production, pred_train, pred_test, bhp, metrics, model_name, max_pres)

    for well_n in range(production.shape[1]):
        trace_cont_pres = go.Scatter(
            x=cp.index,
            y=cp.iloc[:, well_n],
            name='Контурное давление',
            line=dict(
                width=3,
                color='rgb(179, 0, 0)',
            ),
            yaxis='y2')

        figs[well_n].data.append(trace_cont_pres)
    return figs, metric_results


def production_summary(production, pred_train, pred_test, pressure, metrics, model_name='', max_pres=300):
    figs = []
    metric_results = defaultdict(list)
    for i, p in enumerate(production.columns):
        title = model_name + f'. Скважина {p}. '
        metric_records = []
        for name, func in metrics.items():
            val = func(production.iloc[:, i].loc[pred_test.index], pred_test.iloc[:, i])
            metric_results[name].append(val)
            metric_records.append(f'{name}={val:.2f}')
        title += ', '.join(metric_records) + '.'

        layout = get_layout(pred_train.index, pred_test.index, max_pres)
        layout['title'] = title

        trace_train = go.Scatter(
            x=pred_train.index,
            y=pred_train.iloc[:, i],
            name='Предсказание на обучении',
            line=dict(
                    color='rgb(255, 128, 0)',
                    width=2.7,
                ),
            mode='lines'
        )

        trace_test = go.Scatter(
            x=pred_test.index,
            y=pred_test.iloc[:, i],
            mode='lines',
            line=dict(
                    color='rgb(41, 163, 41)',
                    width=2.7,
                ),
            name='Предсказание на тесте'
        )

        trace_true = go.Scatter(
            x=production.index,
            y=production.iloc[:, i],
            mode='lines',
            line=dict(
                    color='rgb(140, 140, 140)',
                ),
            name='Истиное значение'
        )

        trace_pres = go.Scatter(
            x=pressure.index,
            y=pressure.iloc[:, i],
            name='Забойное давление',
            line=dict(
                    dash='dot',
                    color='rgb(179, 0, 0)',
                ),
            yaxis='y2'
        )
        figs.append(go.Figure(data=[trace_train, trace_test, trace_true, trace_pres], layout=layout))
    metric_results = pd.DataFrame.from_dict(metric_results, orient='columns')
    metric_results.index = production.columns
    return figs, metric_results


def get_layout(train_index, test_index, max_pres):
    return go.Layout(
        font=go.Font(
            family='Times New Roman',
            size=17
        ),
        titlefont=dict(
            size=20),
        legend=dict(
            orientation='h',
            font={'size': 14}),
        xaxis=dict(
            showline=True,
            ticks='outside'
        ),
        yaxis=dict(
            title='Дебит',
            showline=True,
            ticks='outside',
            rangemode='nonnegative',
        ),
        yaxis2=dict(
            title='Давление',
            showline=True,
            showgrid=False,
            overlaying='y',
            side='right',
            ticks='outside',
            range=[0, max_pres],
        ),
        shapes=[
            dict(
                type='line',
                yref='paper',
                x0=test_index[0],
                y0=0,
                x1=test_index[0],
                y1=1,
                line=dict(
                    color='rgb(0, 0, 0)',
                    width=2,
                ),
            )
        ],
        annotations=[
            get_annotation('Обучение', 'rgb(255, 128, 0)',
                           train_index[len(train_index) // 2], 1),
            get_annotation('Тест', 'rgb(41, 163, 41)',
                           test_index[len(test_index) // 2], 1)
        ]
    )


def get_annotation(text, bd_color, x_indx, y_rel):
    return dict(
        xref='x',
        x=x_indx,
        yref='paper',
        y=y_rel,
        text=text,
        bgcolor=bd_color,
        opacity=.8,
        font=dict(
            color='white'
        ),
        width=80,
        height=20,
        align='center',
        valign='center',
        showarrow=False,
    )