from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODEL_PLOT_COLORS = {
    "ftor": "#2E91E5",
    "wolfram": "#E15F99",
    "CRM": "#1CA71C",
    "fedot": "#FB0D0D",
    "shelf": "#DA16FF",
    "ensemble": "#FECB52",
    "CRMIP": "#00CC96",
    "Hybrid": "#AB63FA",
    "Syntetic": "#FFA15A",
}

MODEL_PLOT_FALLBACK_COLORS = [
    "#2E91E5",
    "#E15F99",
    "#1CA71C",
    "#FB0D0D",
    "#DA16FF",
    "#FECB52",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]

FACT_PLOT_COLORS = [
    "#FFFFFF",
    "#C7D2FE",
    "#BAE6FD",
    "#D9F99D",
    "#FED7AA",
]


def model_plot_color(model: str, index: int = 0) -> str:
    return MODEL_PLOT_COLORS.get(
        model, MODEL_PLOT_FALLBACK_COLORS[index % len(MODEL_PLOT_FALLBACK_COLORS)]
    )


def fact_plot_color(index: int = 0) -> str:
    return FACT_PLOT_COLORS[index % len(FACT_PLOT_COLORS)]


def calc_relative_error(
        y_true: pd.Series, y_pred: pd.Series, use_abs: bool = True
) -> pd.Series:
    if use_abs:
        # TODO: разобраться с расчётом ошибки
        err = np.abs(y_pred - y_true) / y_true  # np.maximum(y_pred, y_true)
    else:
        err = (y_pred - y_true) / y_true  # np.maximum(y_pred, y_true)
    # Ошибка может быть больше 100%, если одно из значений отрицательное. Исключаем такие случаи.
    err[err > 1] = 1
    err[err < -1] = -1
    return err * 100


def create_well_plot(name: str, dfs: dict, oilfield: str, mode: str = "oil"):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        column_widths=[0.65, 0.35],
        subplot_titles=[
            f"Дебит: {mode}, м3",
            "Относительная ошибка, %",
        ],
    )
    fig.layout.template = "seaborn"
    fig.update_layout(
        font=dict(size=15),
        title_text=f'Скважина "{name}"; {oilfield};',
        legend=dict(
            # orientation="h",
            font=dict(size=15)
        ),
    )

    mark = dict(size=4)
    m = "markers"
    ml = "markers+lines"

    # сейчас факт строится по всем моделям
    for ind, (model, df) in enumerate(dfs.items()):
        if f"{name}_{mode}_pred" in df.columns:
            clr = model_plot_color(model, ind)

            trace = go.Scatter(
                name=f"факт_{model}",
                x=df.index,
                y=df[f"{name}_{mode}_true"],
                mode=m,
                marker=mark,
                marker_color=clr,
                legendgroup=f'group_{model}'
            )
            fig.add_trace(trace, row=1, col=1)

            trace = go.Scatter(
                name=model,
                x=df.index,
                y=df[f"{name}_{mode}_pred"],
                mode=ml,
                marker={**mark, "color": clr},
                line=dict(width=1, color=clr),
                legendgroup=f'group_{model}'
            )
            fig.add_trace(trace, row=1, col=1)

            relative_error = calc_relative_error(
                df[f"{name}_{mode}_true"], df[f"{name}_{mode}_pred"], use_abs=True
            )
            trace = go.Scatter(
                name=f"re_{model}",
                x=df.index,
                y=relative_error,
                mode=ml,
                marker={**mark, "color": clr},
                line=dict(width=1, color=clr),
                showlegend=False,
            )
            fig.add_trace(trace, row=2, col=1)
    return fig


def draw_histogram_model(df_err: pd.DataFrame, bin_size: int, oilfield: str, mode: str):
    fig = make_subplots(rows=1, cols=1)
    fig.layout.template = "seaborn"
    fig.update_layout(
        title_text=f"{mode}. Распределение средней ошибки за весь период прогноза",
        bargap=0.005,
        font=dict(size=15),
        showlegend=False,
        height=500,
    )

    x = df_err.mean()
    fig.add_trace(
        go.Histogram(
            x=x,
            opacity=0.9,
            # histnorm='percent',
            xbins=dict(size=bin_size),
            # nbinsx=8,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(dtick=bin_size, row=1, col=1)
    fig.update_yaxes(title_text="Скважин", title_font_size=15, row=1, col=1)
    fig.update_xaxes(
        title_text=f"Усредненная относительная ошибка, %<br><br>"
                   f"<i>Среднее значениe: <em>{x.mean():.2f}</i></em><br>"
                   f"<i>Стандартное отклонениe: <em>{x.std():.2f}</i></em><br>"
                   f"Месторождение: <em>{oilfield}</em>. Количество скважин: <em>{df_err.shape[1]}</em>",
        title_font_size=16,
        row=1,
        col=1,
    )
    return fig


def draw_wells_model(df_err_model: pd.DataFrame, mode: str):
    fig = make_subplots(
        rows=1,
        cols=1,
    )
    fig.layout.template = "seaborn"
    fig.update_layout(
        title_text=f"{mode}. Средняя относит. ошибка на периоде прогноза, %",
        # bargap=0.005,
        font=dict(size=15),
    )

    mean_err = df_err_model.mean(axis=0)
    mean_err = mean_err.sort_values()
    trace = go.Bar(x=mean_err.index, y=mean_err)
    fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(
        title_text=f"Номер скважины<br><br>"
                   f"<i>Среднее значение ошибки: <em>{mean_err.mean():.2f}</em></i>",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Относит. ошибка, %", row=1, col=1)
    return fig


def draw_performance(
        dfs: dict, df_perf: dict, df_err: dict, oilfield: str, MODEL_NAMES: dict, mode="oil"
):
    modes_decode = {
        "oil": "нефти",
        "liq": "жидкости",
        "gas": "газа",
    }
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f"Суммарная суточная добыча {modes_decode[mode]}, м3/сут",
            "Относительное отклонение от факта, %",
        ],
    )
    fig.layout.template = "seaborn"
    fig.update_layout(
        title=dict(text=f"Месторождение {oilfield}", x=0.05, xanchor="left"),
        font=dict(size=10),
        legend=dict(font=dict(size=15)),
        height=630,
    )
    mark = dict(size=6)

    models = [model for model in dfs.keys() if not (df_perf[model]["факт"] == 0).all()]
    # сортировка моделей, у которых факт совпадает
    models_count = defaultdict(int)
    for combination in list(combinations(models, 2)):
        if df_perf[combination[0]]["факт"].equals(df_perf[combination[1]]["факт"]):
            models_count[combination[0]] += 1
            models_count[combination[1]] += 1
    # словарь с одинаковыми фактами
    models_same = defaultdict(list)
    for model, values in models_count.items():
        models_same[values].append(model)
    # построение факта
    for ind, model in models_same.items():
        clr = fact_plot_color(ind)
        x = df_perf[model[0]].index
        if len(models_same) == 1:
            name_legend = f"Факт"
        else:
            name_legend = f"Факт {[MODEL_NAMES[m] for m in model]}"
        trace = go.Scatter(
            name=name_legend,
            x=x,
            y=df_perf[model[0]]["факт"],
            mode="markers",
            marker=mark,
            marker_color=clr,
            showlegend=True,
        )
        fig.add_trace(trace, row=1, col=1)
    annotation_text = ""
    # Model errors
    for ind, model in enumerate(models):
        clr = model_plot_color(model, ind)
        x = df_perf[model].index
        trace1 = go.Scatter(
            name=f"{MODEL_NAMES[model]}",
            x=x,
            y=df_perf[model]["модель"],
            mode="markers+lines",
            marker={**mark, "color": clr},
            line=dict(width=2, color=clr),
            legendgroup=f'group_{model}'
        )
        trace2 = go.Scatter(
            x=x,
            y=df_err[model]["модель"],
            mode="markers+lines",
            marker={**mark, "color": clr},
            line=dict(width=2, color=clr),
            showlegend=False,
            legendgroup=f'group_{model}'
        )
        annotation_text += (
            f"<i>Среднее значение ошибки <em>{MODEL_NAMES[model]}</em>: "
            f'{df_err[model]["модель"].mean():.2f}</i><br>'
        )
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)
        fig.update_layout(
            legend=dict(
                # orientation="h",
                font=dict(size=10))
        )
    # fig.update_xaxes(title_text=annotation_text, title_font_size=16, row=2, col=1)
    return fig


def draw_statistics(
        models: list,
        model_mean: dict,
        model_std: dict,
        model_mean_daily: dict,
        model_std_daily: dict,
        oilfield: str,
        dates: pd.date_range,
        MODEL_NAMES: dict,
        mode: str,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Средняя относит. ошибка по накопленной добыче, %",
            # "Стандартное отклонение по накопленной добыче, %",
            "Средняя относит. ошибка суточной добычи, %",
            # "Стандартное отклонение по суточной добыче, %",
        ],
    )
    fig.layout.template = "seaborn"
    fig.update_layout(
        title=dict(
            text=f"Месторождение <em>{oilfield}</em>. {mode}", x=0.05, xanchor="left"
        ),
        font=dict(size=10),
        height=630,
    )
    mark = dict(size=6)
    ml = "markers+lines"
    # Model errors
    for ind, model in enumerate(models):
        if MODEL_NAMES[model] == "CRM" and mode == "Дебит нефти":
            continue
        clr = model_plot_color(model, ind)
        trace1 = go.Scatter(
            name=f"{MODEL_NAMES[model]}",
            x=dates,
            y=model_mean[model],
            mode=ml,
            marker={**mark, "color": clr},
            line=dict(width=1, color=clr),
            legendgroup=f'group_{model}'
        )
        # trace2 = go.Scatter(
        #     x=dates,
        #     y=model_std[model],
        #     mode=ml,
        #     marker=mark,
        #     line=dict(width=1, color=clr),
        #     showlegend=False,
        # )
        trace3 = go.Scatter(
            x=dates,
            y=model_mean_daily[model],
            mode=ml,
            marker={**mark, "color": clr},
            line=dict(width=1, color=clr),
            showlegend=False,
            legendgroup=f'group_{model}'
        )
        # trace4 = go.Scatter(
        #     x=dates,
        #     y=model_std_daily[model],
        #     mode=ml,
        #     marker=mark,
        #     line=dict(width=1, color=clr),
        #     showlegend=False,
        # )
        fig.add_trace(trace1, row=1, col=1)
        # fig.add_trace(trace2, row=2, col=1)
        fig.add_trace(trace3, row=2, col=1)
        # fig.add_trace(trace4, row=4, col=1)
    return fig


# бары с ошибками по скважинам
def draw_wells_model_multi(
        df_err: pd.DataFrame, models: list, MODEL_NAMES: dict, mode: str
):
    fig = make_subplots(
        rows=1,
        cols=1,
    )
    fig.layout.template = "seaborn"
    fig.update_layout(
        title_text=f"{mode}. Средняя относит. ошибка на периоде прогноза, %",
        # bargap=0.005,
        font=dict(size=15),
    )
    title_text = f"Номер скважины<br><br>"
    # Сортировка начиная со скважин с большей ошибкой. Для лучшей визуализации
    models_dict = {}
    for model in models:
        if mode == "Дебит нефти" and MODEL_NAMES[model] == "CRM":
            continue
        models_dict[model] = df_err[model].mean(axis=0).mean()
    models_dict = dict(sorted(models_dict.items(), key=lambda x: x[1], reverse=True))
    models_name = list(models_dict.keys())
    for i in range(len(models_name)):
        mean_err = df_err[models_name[i]].mean(axis=0)
        mean_err = mean_err.sort_values()
        trace = go.Bar(
            x=mean_err.index,
            y=mean_err,
            opacity=1 - i / 10,
            name=MODEL_NAMES[models_name[i]],
        )
        fig.add_trace(trace, row=1, col=1)
        title_text += f"<i>Среднее значениe {MODEL_NAMES[models_name[i]]}: <em>{mean_err.mean():.2f}</i></em><br>"

    fig.update_layout(barmode="overlay")
    # fig.update_xaxes(title_text=title_text, title_font_size=16, row=1, col=1)
    fig.update_yaxes(title_text="Относит. ошибка, %", row=1, col=1)
    return fig


# гистограмма распределения ошибки
def draw_histogram_model_multi(
        df_err: pd.DataFrame,
        bin_size: int,
        oilfield: str,
        models: list,
        MODEL_NAMES: dict,
        mode: str,
):
    fig = make_subplots(rows=1, cols=1)
    fig.layout.template = "seaborn"
    fig.update_layout(
        title_text=f"{mode}. Распределение средней ошибки за весь период прогноза",
        bargap=0.005,
        font=dict(size=15),
        showlegend=True,
        height=500,
    )
    title_text = f"Усредненная относительная ошибка, %<br><br>"
    for i in range(len(models)):
        if mode == "Дебит нефти" and MODEL_NAMES[models[i]] == "CRM":
            continue
        x = df_err[models[i]].mean()
        fig.add_trace(
            go.Histogram(
                x=x,
                opacity=1 - i / 10,
                # histnorm='percent',
                xbins=dict(size=bin_size),
                # nbinsx=8,
                name=MODEL_NAMES[models[i]],
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        title_text += (
                f"<i>Среднее значениe {MODEL_NAMES[models[i]]}: <em>{x.mean():.2f}</i></em><br>"
                + f"<i>Стандартное отклонениe {MODEL_NAMES[models[i]]}: <em>{x.std():.2f}</i></em><br>"
        )

    fig.update_layout(barmode="overlay")
    fig.update_xaxes(dtick=bin_size, row=1, col=1)
    fig.update_yaxes(title_text="Скважин", title_font_size=15, row=1, col=1)
    # fig.update_xaxes(
    #     title_text=title_text + "\n"
    #                             f"Месторождение: <em>{oilfield}</em>. Количество скважин: <em>{df_err[models[i]].shape[1]}</em>",
    #     title_font_size=16,
    #     row=1,
    #     col=1,
    # )
    return fig


# таблица со статистическими результатами всех моделей
def draw_table_statistics(
        models: list,
        df_err: dict,
        df_err_liq: dict,
        df_err_gas: dict,
        model_mean: dict,
        model_mean_liq: dict,
        model_mean_gas: dict,
        MODEL_NAMES: dict,
):
    models_names = []  # названия скважин
    mean_error_liq_wells = []  # модуль средняя относительная ошибка жидкости по скважинам
    rmse_liq = []  # СКО жидкости
    cumsum_liq_mean_error = []  # ошибка по накопленной добыче жидкости
    mean_error_wells = []  # модуль средняя относительная ошибка нефти по скважинам
    rmse_oil = []  # СКО нефти
    cumsum_mean_error = []  # ошибка по накопленной добыче нефти
    mean_error_gas_wells = []  # модуль средняя относительная ошибка газа по скважинам
    rmse_gas = []  # СКО газа
    cumsum_gas_mean_error = []  # ошибка по накопленной добыче газа
    for model in models:
        models_names.append(f"<b>{MODEL_NAMES[model]}<b>")
        mean_error_liq_wells.append(round(df_err_liq[model].mean().mean(), 2))
        rmse_liq.append(round(df_err_liq[model].mean().std(), 2))
        cumsum_liq_mean_error.append(round(model_mean_liq[model].mean(), 2))
        if MODEL_NAMES[model] == "CRM":
            mean_error_wells.append("")
            rmse_oil.append("")
            cumsum_mean_error.append("")
        else:
            mean_error_wells.append(round(df_err[model].mean().mean(), 2))
            rmse_oil.append(round(df_err[model].mean().std(), 2))
            cumsum_mean_error.append(round(model_mean[model].mean(), 2))
            mean_error_gas_wells.append(round(df_err_gas[model].mean().mean(), 2))
            rmse_gas.append(round(df_err_gas[model].mean().std(), 2))
            cumsum_gas_mean_error.append(round(model_mean_gas[model].mean(), 2))
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[140, 250, 200, 250, 250, 200, 250, 250, 200, 250],
                header=dict(
                    values=[
                        "",
                        "<b>Модуль относительной<br>средней ошибки<br>по жидкости, %<b>",
                        "<b>СКО ошибки<br>по жидкости<b>",
                        "<b>Ошибка<br>по накопленной<br>добыче жидкости, %<b>",
                        "<b>Модуль относительной<br>средней ошибки<br>по нефти, %<b>",
                        "<b>СКО ошибки<br>по нефти<b>",
                        "<b>Ошибка<br>по накопленной<br>добыче нефти, %<b>",
                        "<b>Модуль относительной<br>средней ошибки<br>по газу, %<b>",
                        "<b>СКО ошибки<br>по газу<b>",
                        "<b>Ошибка<br>по накопленной<br>добыче газа, %<b>"
                    ],
                    align="center",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=[
                        models_names,
                        mean_error_liq_wells,
                        rmse_liq,
                        cumsum_liq_mean_error,
                        mean_error_wells,
                        rmse_oil,
                        cumsum_mean_error,
                        mean_error_gas_wells,
                        rmse_gas,
                        cumsum_gas_mean_error,
                    ],
                    align="center",
                    font=dict(color="black", size=14),
                ),
            )
        ]
    )

    fig.update_layout(height=630, width=1500)
    return fig
