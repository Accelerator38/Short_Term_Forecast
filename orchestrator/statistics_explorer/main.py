import datetime
from pathlib import Path

import pandas as pd
import plotly as pl

from statistics_explorer.config import ConfigStatistics
from statistics_explorer.plots import (
    calc_relative_error,
    create_well_plot,
    draw_histogram_model,
    draw_histogram_model_multi,
    draw_performance,
    draw_statistics,
    draw_table_statistics,
    draw_wells_model,
    draw_wells_model_multi,
)


_REQUIRED_WELL_SUFFIXES = ("oil_true", "oil_pred", "liq_true", "liq_pred")


def _has_well_columns(df: pd.DataFrame, well_name: str, suffixes: tuple[str, ...]) -> bool:
    return all(f"{well_name}_{suffix}" in df.columns for suffix in suffixes)


def _zero_series(index) -> pd.Series:
    return pd.Series(0.0, index=index)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _apply_fact_weight(error: pd.Series, fact: pd.Series, max_fact: float) -> pd.Series:
    return error.fillna(0) * _safe_ratio(fact.sum(), max_fact)


def _weighted_relative_error(
        y_true: pd.Series,
        y_pred: pd.Series,
        max_fact: float,
        use_abs: bool,
) -> pd.Series:
    return _apply_fact_weight(calc_relative_error(y_true, y_pred, use_abs=use_abs), y_true, max_fact)


def calculate_statistics(dfs: dict, config: ConfigStatistics):
    # Initialize data
    analytics_plots = {}
    models = list(dfs.keys())
    df_perf = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["факт", "модель"])
        for key in models
    }
    df_perf_liq = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["факт", "модель"])
        for key in models
    }
    df_perf_gas = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["факт", "модель"])
        for key in models
    }
    df_err = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["модель"])
        for key in models
    }
    df_err_liq = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["модель"])
        for key in models
    }
    df_err_gas = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["модель"])
        for key in models
    }
    # Daily model error
    df_err_model = {key: pd.DataFrame(index=config.dates) for key in models}
    df_err_model_liq = {key: pd.DataFrame(index=config.dates) for key in models}
    df_err_model_gas = {key: pd.DataFrame(index=config.dates) for key in models}
    # для распределения ошибки
    df_err_model_distribution = {
        key: pd.DataFrame(index=config.dates) for key in models
    }
    df_err_model_liq_distribution = {
        key: pd.DataFrame(index=config.dates) for key in models
    }
    df_err_model_gas_distribution = {
        key: pd.DataFrame(index=config.dates) for key in models
    }
    # Cumulative model error
    df_cumerr_model = {key: pd.DataFrame(index=config.dates) for key in models}
    df_cumerr_model_liq = {key: pd.DataFrame(index=config.dates) for key in models}
    df_cumerr_model_gas = {key: pd.DataFrame(index=config.dates) for key in models}

    model_mean = dict.fromkeys(models)
    model_std = dict.fromkeys(models)
    model_mean_liq = dict.fromkeys(models)
    model_std_liq = dict.fromkeys(models)
    model_mean_gas = dict.fromkeys(models)
    model_std_gas = dict.fromkeys(models)
    model_mean_daily = dict.fromkeys(models)
    model_std_daily = dict.fromkeys(models)
    model_mean_daily_liq = dict.fromkeys(models)
    model_std_daily_liq = dict.fromkeys(models)
    model_mean_daily_gas = dict.fromkeys(models)
    model_std_daily_gas = dict.fromkeys(models)
    # find the most well
    mvp_wells = {key: {} for key in ["liq", "oil", "gas"]}
    for model in models:
        model_well_sum_liq_fact = 0
        model_well_sum_oil_fact = 0
        model_well_sum_gas_fact = 0
        df_model = dfs[model]
        for _well_name in config.well_names:
            # Check if current model has this well
            if not _has_well_columns(df_model, _well_name, _REQUIRED_WELL_SUFFIXES):
                continue
            if df_model[f"{_well_name}_liq_true"].sum() >= model_well_sum_liq_fact:
                model_well_sum_liq_fact = df_model[f"{_well_name}_liq_true"].sum()
            if df_model[f"{_well_name}_oil_true"].sum() >= model_well_sum_oil_fact:
                model_well_sum_oil_fact = df_model[f"{_well_name}_oil_true"].sum()
            if _has_well_columns(df_model, _well_name, ("gas_true", "gas_pred")):
                if df_model[f"{_well_name}_gas_true"].sum() >= model_well_sum_gas_fact:
                    model_well_sum_gas_fact = df_model[f"{_well_name}_gas_true"].sum()
        mvp_wells["liq"][model] = model_well_sum_liq_fact
        mvp_wells["oil"][model] = model_well_sum_oil_fact
        mvp_wells["gas"][model] = model_well_sum_gas_fact

    # Calculations
    for model in models:
        df_model = dfs[model]
        for _well_name in config.well_names:
            # Check if current model has this well
            if not _has_well_columns(df_model, _well_name, _REQUIRED_WELL_SUFFIXES):
                continue
            q_fact = df_model[f"{_well_name}_oil_true"]
            q_model = df_model[f"{_well_name}_oil_pred"]
            q_fact_liq = df_model[f"{_well_name}_liq_true"]
            q_model_liq = df_model[f"{_well_name}_liq_pred"]
            has_gas = _has_well_columns(df_model, _well_name, ("gas_true", "gas_pred"))
            q_fact_gas = df_model[f"{_well_name}_gas_true"] if has_gas else _zero_series(q_fact.index)
            q_model_gas = df_model[f"{_well_name}_gas_pred"] if has_gas else _zero_series(q_fact.index)
            # Ошибка по суточной добыче
            df_err_model[model][f"{_well_name}"] = _weighted_relative_error(
                q_fact, q_model, mvp_wells["oil"][model], use_abs=config.use_abs
            )
            df_err_model_liq[model][f"{_well_name}"] = _weighted_relative_error(
                q_fact_liq, q_model_liq, mvp_wells["liq"][model], use_abs=config.use_abs
            )
            if has_gas:
                df_err_model_gas[model][f"{_well_name}"] = _weighted_relative_error(
                    q_fact_gas, q_model_gas, mvp_wells["gas"][model], use_abs=config.use_abs
                )
            # ошибка с учётом дебита
            # ошибка для распределения
            df_err_model_distribution[model][f"{_well_name}"] = _weighted_relative_error(
                q_fact, q_model, mvp_wells["oil"][model], use_abs=False
            )
            # ошибка для распределения
            df_err_model_liq_distribution[model][f"{_well_name}"] = _weighted_relative_error(
                q_fact_liq, q_model_liq, mvp_wells["liq"][model], use_abs=False
            )
            # ошибка для распределения
            if has_gas:
                df_err_model_gas_distribution[model][f"{_well_name}"] = _weighted_relative_error(
                    q_fact_gas, q_model_gas, mvp_wells["gas"][model], use_abs=False
                )
            # ошибка с учётом дебита
            # Ошибка по накопленной добыче
            Q_model = q_model.cumsum()
            Q_fact = q_fact.cumsum()
            df_cumerr_model[model][f"{_well_name}"] = _apply_fact_weight(
                calc_relative_error(Q_fact, Q_model, use_abs=config.use_abs),
                q_fact,
                mvp_wells["oil"][model],
            )
            Q_model_liq = q_model_liq.cumsum()
            Q_fact_liq = q_fact_liq.cumsum()
            df_cumerr_model_liq[model][f"{_well_name}"] = _apply_fact_weight(
                calc_relative_error(Q_fact_liq, Q_model_liq, use_abs=config.use_abs),
                q_fact_liq,
                mvp_wells["liq"][model],
            )
            if has_gas:
                Q_model_gas = q_model_gas.cumsum()
                Q_fact_gas = q_fact_gas.cumsum()
                df_cumerr_model_gas[model][f"{_well_name}"] = _apply_fact_weight(
                    calc_relative_error(Q_fact_gas, Q_model_gas, use_abs=config.use_abs),
                    q_fact_gas,
                    mvp_wells["gas"][model],
                )
            # ошибка с учётом дебита
            df_perf[model]["факт"] += q_fact.fillna(0)
            df_perf[model]["модель"] += q_model.fillna(0)
            df_perf_liq[model]["факт"] += q_fact_liq.fillna(0)
            df_perf_liq[model]["модель"] += q_model_liq.fillna(0)
            df_perf_gas[model]["факт"] += q_fact_gas.fillna(0)
            df_perf_gas[model]["модель"] += q_model_gas.fillna(0)

        # for model in models:
        # Ошибка по суммарной добыче на каждые сутки
        df_err[model]["модель"] = calc_relative_error(
            df_perf[model]["факт"], df_perf[model]["модель"], use_abs=config.use_abs
        )
        df_err_liq[model]["модель"] = calc_relative_error(
            df_perf_liq[model]["факт"],
            df_perf_liq[model]["модель"],
            use_abs=config.use_abs,
        )
        df_err_gas[model]["модель"] = calc_relative_error(
            df_perf_gas[model]["факт"],
            df_perf_gas[model]["модель"],
            use_abs=config.use_abs,
        )

        df_err[model].fillna(0, inplace=True)
        df_err_liq[model].fillna(0, inplace=True)
        df_err_gas[model].fillna(0, inplace=True)

        model_mean[model] = df_cumerr_model[model].mean(axis=1)
        model_std[model] = df_cumerr_model[model].std(axis=1)

        model_mean_liq[model] = df_cumerr_model_liq[model].mean(axis=1)
        model_std_liq[model] = df_cumerr_model_liq[model].std(axis=1)

        model_mean_gas[model] = df_cumerr_model_gas[model].mean(axis=1)
        model_std_gas[model] = df_cumerr_model_gas[model].std(axis=1)

        model_mean_daily[model] = df_err_model[model].mean(axis=1)
        model_std_daily[model] = df_err_model[model].std(axis=1)

        model_mean_daily_liq[model] = df_err_model_liq[model].mean(axis=1)
        model_std_daily_liq[model] = df_err_model_liq[model].std(axis=1)

        model_mean_daily_gas[model] = df_err_model_gas[model].mean(axis=1)
        model_std_daily_gas[model] = df_err_model_gas[model].std(axis=1)

        # temp_name = f'Распределение ошибки (нефть) "{config.MODEL_NAMES[model]}"'
        # analytics_plots[temp_name] = draw_histogram_model(
        #     df_err_model[model], config.bin_size, config.oilfield, "Дебит нефти"
        # )
        # temp_name = f'Ошибка прогноза (нефть) "{config.MODEL_NAMES[model]}"'
        # analytics_plots[temp_name] = draw_wells_model(
        #     df_err_model[model], "Дебит нефти"
        # )
        #
        # temp_name = f'Распределение ошибки (жидкость) "{config.MODEL_NAMES[model]}"'
        # analytics_plots[temp_name] = draw_histogram_model(
        #     df_err_model_liq[model], config.bin_size, config.oilfield, "Дебит жидкости"
        # )
        # temp_name = f'Ошибка прогноза (жидкость) "{config.MODEL_NAMES[model]}"'
        # analytics_plots[temp_name] = draw_wells_model(
        #     df_err_model_liq[model], "Дебит жидкости"
        # )

    temp_name = f"Ошибка прогноза (нефть)"
    analytics_plots[temp_name] = draw_wells_model_multi(
        df_err_model, models, config.MODEL_NAMES, "Дебит нефти"
    )
    temp_name_multi_oil = f"Распределение ошибки (нефть)"
    analytics_plots[temp_name_multi_oil] = draw_histogram_model_multi(
        df_err_model_distribution,
        config.bin_size,
        config.oilfield,
        models,
        config.MODEL_NAMES,
        "Дебит нефти",
    )
    temp_name = f"Ошибка прогноза (жидкость)"
    analytics_plots[temp_name] = draw_wells_model_multi(
        df_err_model_liq, models, config.MODEL_NAMES, "Дебит жидкости"
    )
    temp_name_multi_liq = f"Распределение ошибки (жидкость)"
    analytics_plots[temp_name_multi_liq] = draw_histogram_model_multi(
        df_err_model_liq_distribution,
        config.bin_size,
        config.oilfield,
        models,
        config.MODEL_NAMES,
        "Дебит жидкости",
    )
    temp_name = f"Ошибка прогноза (газ)"
    analytics_plots[temp_name] = draw_wells_model_multi(
        df_err_model_gas, models, config.MODEL_NAMES, "Дебит газа"
    )
    temp_name_multi_gas = f"Распределение ошибки (газ)"
    analytics_plots[temp_name_multi_gas] = draw_histogram_model_multi(
        df_err_model_gas_distribution,
        config.bin_size,
        config.oilfield,
        models,
        config.MODEL_NAMES,
        "Дебит газа",
    )

    # Draw common statistics
    analytics_plots["Суммарная добыча нефти"] = draw_performance(
        dfs, df_perf, df_err, config.oilfield, config.MODEL_NAMES, mode="oil"
    )
    analytics_plots["Суммарная добыча жидкости"] = draw_performance(
        dfs, df_perf_liq, df_err_liq, config.oilfield, config.MODEL_NAMES, mode="liq"
    )
    analytics_plots["Суммарная добыча газа"] = draw_performance(
        dfs, df_perf_gas, df_err_gas, config.oilfield, config.MODEL_NAMES, mode="gas"
    )

    analytics_plots["Статистика по нефти"] = draw_statistics(
        models,
        model_mean,
        model_std,
        model_mean_daily,
        model_std_daily,
        config.oilfield,
        config.dates,
        config.MODEL_NAMES,
        "Дебит нефти",
    )
    analytics_plots["Статистика по жидкости"] = draw_statistics(
        models,
        model_mean_liq,
        model_std_liq,
        model_mean_daily_liq,
        model_std_daily_liq,
        config.oilfield,
        config.dates,
        config.MODEL_NAMES,
        "Дебит жидкости",
    )
    analytics_plots["Статистика по газу"] = draw_statistics(
        models,
        model_mean_gas,
        model_std_gas,
        model_mean_daily_gas,
        model_std_daily_gas,
        config.oilfield,
        config.dates,
        config.MODEL_NAMES,
        "Дебит газа",
    )
    analytics_plots["Статистика по моделям"] = draw_table_statistics(
        models,
        df_err_model,
        df_err_model_liq,
        df_err_model_gas,
        model_mean,
        model_mean_liq,
        model_mean_gas,
        config.MODEL_NAMES,
    )
    return analytics_plots


if __name__ == "__main__":
    # Конфиг
    config_stats = ConfigStatistics(
        oilfield="Крайнее",
        dates=pd.date_range(
            datetime.date(2022, 2, 1), datetime.date(2022, 4, 30), freq="D"
        ).date,
        use_abs=True,
        bin_size=10,
    )
    # Задание имен моделей на графиках. Ключ - название .xlsx файла, значение - название на графике.
    config_stats.MODEL_NAMES = {
        "ftor": "Пьезо",
        "wolfram": "ML",
        "ensemble": "Ансамбль",
        "CRM": "CRM",
        "fedot": "CRM+ML",
        "shelf": "ППТП",
    }
    path_read = Path.cwd() / config_stats.oilfield / "input_data"
    path_save = Path.cwd() / config_stats.oilfield / "output_data"

    # %% Read data
    # Store dataframes of each model
    xlsx_files = path_read.glob("*.xlsx")
    dfs = dict()
    for filepath in xlsx_files:
        df = pd.read_excel(filepath, engine="openpyxl")
        df.name = filepath.stem
        dfs[filepath.stem] = df

    # Obtain all well names
    wells_in_model = []
    for df in dfs.values():
        df.rename(columns={f"{df.columns[0]}": "date"}, inplace=True)
        df.date = df.date.apply(lambda x: x.date())
        df.set_index("date", inplace=True)
        df[df < 0] = 0
        df = df.reindex(config_stats.dates)
        wells_in_model.append(set([col.split("_")[0] for col in df.columns]))
    well_names_common = list(set.intersection(*wells_in_model))
    well_names_all = list(set.union(*wells_in_model))
    # Можно строить статистику только для общего набора скважин (скважина рассчитана всеми моделями),
    # либо для всех скважин (скважина рассчитана хотя бы одной моделью).
    # Выберите, что подать в конфиг ниже: well_names_common или well_names_all.
    config_stats.well_names = well_names_common
    # config_stats.exclude_wells(['2860242300'])  # Список названий скважин, которые нужно исключить из статистики

    analytics_plots = calculate_statistics(dfs, config_stats)
    available_plots = [*analytics_plots]
    plots_to_save = [
        plot_name
        for plot_name in available_plots
        if plot_name not in config_stats.ignore_plots
    ]

    # Сохранение поскважинных графиков
    # for mode in ["liq", "oil"]:
    #     if not Path(f"{path_save}/well plots/{mode}").exists():
    #         Path(f"{path_save}/well plots/{mode}").mkdir(parents=True, exist_ok=True)
    #     for name in config_stats.well_names:
    #         fig = create_well_plot(name, dfs, oilfield=config_stats.oilfield, mode=mode)
    #         pl.io.write_image(
    #             fig,
    #             file=f"{path_save}/well plots/{mode}/{name}.png",
    #             width=1450,
    #             height=700,
    #             scale=2,
    #             engine="kaleido",
    #         )
    # fig.write_html(f"{path_save}/well plots/{mode}/{name}.html")

    # Сохранение графиков статистики
    for plot_name in plots_to_save:
        savename = plot_name.replace('"', "")
        if not Path(f"{path_save}").exists():
            Path(f"{path_save}").mkdir(parents=True, exist_ok=True)
        pl.io.write_image(
            analytics_plots[plot_name],
            file=f"{path_save}/{savename}.png",
            width=1450,
            height=700,
            scale=2,
            engine="kaleido",
        )
        # analytics_plots[plot_name].write_html(f'{path_save}/{savename}.html')
