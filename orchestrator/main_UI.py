import datetime as dt
import sys
from importlib import import_module, util
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from UI.app_state import AppState
from UI.config import DEFAULT_FTOR_BOUNDS


APP_TITLE = "КСП: ансамблевый краткосрочный прогноз"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "tools_preprocessor" / "data"


def state_defaults() -> dict:
    return {
        "statistics": {},
        "statistics_test_only": {},
        "statistics_another_models": None,
        "statistics_test_index": pd.DatetimeIndex([]),
        "ensemble_interval": pd.DataFrame(),
        "models_weights": {},
        "adapt_params": {},
        "exclude_wells": [],
        "buffer": None,
        "constraints": {},
        "selected_wells_norm": [],
        "wellnames_key_normal": {},
        "wellnames_key_ois": {},
        "wells_coords_CRM": pd.DataFrame(),
        "field_name": "",
        "shops": [],
        "date_start": dt.date(2019, 1, 1),
        "date_test": dt.date(2019, 2, 1),
        "date_end": dt.date(2019, 4, 30),
        "CRM_influence_R": 1300,
        "CRM_maxiter": 100,
        "CRM_p_res": 220,
        "n_days_past": 14,
        "n_days_calc_avg": 7,
        "estimator_name_group": "xgb",
        "estimator_name_well": "svr",
        "is_deep_grid_search": False,
        "window_sizes": [3, 5, 7, 15, 30],
        "quantiles": [0.1, 0.3],
        "ensemble_adapt_period": 30,
        "interval_probability": 0.9,
        "draws": 300,
        "tune": 200,
        "chains": 1,
        "target_accept": 0.95,
        "calc_ftor": True,
        "calc_wolfram": True,
        "calc_CRM": False,
        "calc_CRMIP": False,
        "calc_shelf": False,
        "calc_ensemble": True,
        "was_calc_ensemble": False,
        "change_gtm": 0,
        "change_gtm_info": 0,
        "change_tp": 0,
        "shelf_json": {},
    }


def apply_state_defaults(state: AppState) -> None:
    for key, value in state_defaults().items():
        state.setdefault(key, value)

    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        state.setdefault(f"{param_name}_is_adapt", True)
        state.setdefault(f"{param_name}_lower", param_dict["lower_val"])
        state.setdefault(f"{param_name}_default", param_dict["default_val"])
        state.setdefault(f"{param_name}_upper", param_dict["upper_val"])


LEGACY_SESSION_KEYS = tuple(state_defaults().keys()) + tuple(
    key
    for param_name in DEFAULT_FTOR_BOUNDS
    for key in (
        f"{param_name}_is_adapt",
        f"{param_name}_lower",
        f"{param_name}_default",
        f"{param_name}_upper",
    )
)


def sync_state_from_legacy_session(session, state: AppState) -> None:
    for key in LEGACY_SESSION_KEYS:
        if key in session:
            state[key] = session[key]


def sync_legacy_session_state(session, state: AppState) -> None:
    for key in LEGACY_SESSION_KEYS:
        if key in state:
            session[key] = state[key]


def discover_fields() -> dict[str, list[str]]:
    """Read available fields and shops from tools_preprocessor/data."""
    fields: dict[str, list[str]] = {}
    if not DATA_DIR.exists():
        return fields
    for field_dir in sorted(DATA_DIR.iterdir()):
        welllist = field_dir / "welllist.feather"
        if not welllist.exists():
            continue
        try:
            df = pd.read_feather(welllist)
            shops = sorted(str(shop) for shop in df["ceh"].dropna().unique())
        except Exception:
            shops = []
        fields[field_dir.name] = shops
    return fields


def load_well_mapping(field_name: str, shops: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Return normal-name -> OIS and OIS -> normal-name mappings."""
    welllist = DATA_DIR / field_name / "welllist.feather"
    if not welllist.exists():
        return {}, {}

    try:
        df = pd.read_feather(welllist)
    except Exception as exc:
        st.sidebar.warning(f"Не удалось прочитать {welllist.name}: {exc}")
        return {}, {}
    if shops and "ceh" in df.columns:
        df = df[df["ceh"].isin(shops)]
    if "npath" in df.columns:
        df = df[df["npath"] == 0]

    normal_to_ois: dict[str, int] = {}
    ois_to_normal: dict[int, str] = {}
    for _, row in df.dropna(subset=["num", "ois"]).iterrows():
        normal_name = str(row["num"])
        ois_name = int(row["ois"])
        normal_to_ois[normal_name] = ois_name
        ois_to_normal[ois_name] = normal_name
    return dict(sorted(normal_to_ois.items())), ois_to_normal


def init_state() -> AppState:
    if "state" not in st.session_state:
        st.session_state.state = AppState()
    state: AppState = st.session_state.state

    apply_state_defaults(state)
    return state


def lazy_page(module_name: str, func_name: str = "show") -> Callable:
    module = import_module(module_name)
    return getattr(module, func_name)


def selected_wells_ois(state: AppState) -> list[int]:
    if not state.selected_wells_norm:
        return []
    if "Все скважины" in state.selected_wells_norm:
        return sorted(state.wellnames_key_ois)
    return [state.wellnames_key_normal[name] for name in state.selected_wells_norm]


def selected_wells_normal(state: AppState) -> list[str]:
    if not state.selected_wells_norm:
        return []
    if "Все скважины" in state.selected_wells_norm:
        return sorted(state.wellnames_key_normal)
    return [name for name in state.selected_wells_norm if name in state.wellnames_key_normal]


def selected_dependency_errors(state: AppState) -> list[str]:
    requirements: list[tuple[str, str, str]] = []
    if state.calc_ensemble:
        requirements.extend(
            [
                ("pymc3", "pymc3", "Ансамбль"),
                ("theano", "theano", "Ансамбль"),
            ]
        )

    errors = []
    for module_name, package_name, model_name in requirements:
        if util.find_spec(module_name) is None:
            errors.append(f"{model_name}: нет пакета `{package_name}`")
    return errors


def draw_sidebar(state: AppState) -> None:
    st.sidebar.header("Параметры расчета")
    fields = discover_fields()
    if not fields:
        st.sidebar.error("Не найдены данные в tools_preprocessor/data.")
        return

    field_names = sorted(fields)
    default_field_index = field_names.index(state.field_name) if state.field_name in field_names else 0
    state.field_name = st.sidebar.selectbox("Месторождение", field_names, index=default_field_index)

    shops = fields[state.field_name]
    default_shops = [shop for shop in state.shops if shop in shops] or shops
    state.shops = st.sidebar.multiselect("Цеха", shops, default=default_shops)

    state.date_start = st.sidebar.date_input("Начало адаптации", value=state.date_start)
    state.date_test = st.sidebar.date_input("Начало прогноза", value=state.date_test)
    state.date_end = st.sidebar.date_input("Конец прогноза", value=state.date_end)

    normal_to_ois, ois_to_normal = load_well_mapping(state.field_name, state.shops)
    state.wellnames_key_normal = normal_to_ois
    state.wellnames_key_ois = ois_to_normal

    well_options = ["Все скважины", *normal_to_ois.keys()]
    default_wells = [well for well in state.selected_wells_norm if well in well_options]
    if not default_wells and well_options:
        default_wells = well_options[:1]
    state.selected_wells_norm = st.sidebar.multiselect("Скважины", well_options, default=default_wells)

    st.sidebar.subheader("Модели")
    state.calc_ftor = st.sidebar.checkbox("Пьезопроводность / FTOR", value=state.calc_ftor)
    state.calc_wolfram = st.sidebar.checkbox("ML / Wolfram", value=state.calc_wolfram)
    state.calc_CRM = st.sidebar.checkbox("CRM", value=state.calc_CRM)
    state.calc_CRMIP = st.sidebar.checkbox("CRMIP", value=state.calc_CRMIP)
    state.calc_shelf = st.sidebar.checkbox("Shelf / ППТП", value=state.calc_shelf)
    state.calc_ensemble = st.sidebar.checkbox("Ансамбль", value=state.calc_ensemble)

    if st.sidebar.button("Запустить расчеты"):
        run_calculations(state)


def run_calculations(state: AppState) -> None:
    if state.date_start >= state.date_test or state.date_test > state.date_end:
        st.error("Проверьте даты: начало адаптации < начало прогноза <= конец прогноза.")
        return

    well_names = selected_wells_ois(state)
    if not well_names:
        st.error("Выберите хотя бы одну скважину.")
        return

    dependency_errors = selected_dependency_errors(state)
    if dependency_errors:
        st.error("Не хватает библиотек для выбранных моделей: " + "; ".join(dependency_errors))
        if sys.version_info >= (3, 12):
            st.info(
                "Текущий запуск использует Python "
                f"{sys.version_info.major}.{sys.version_info.minor}. "
                "PyMC3/Theano из исходного `environment.yml` рассчитаны на старый стек Python 3.9."
            )
        return

    from tools_preprocessor.config import Config as ConfigPreprocessor
    from UI.cached_funcs import (
        calculate_CRM,
        calculate_ensemble,
        calculate_ftor,
        calculate_shelf,
        calculate_wolfram,
        run_preprocessor,
    )
    from UI.data_processor import (
        convert_tones_to_m3_for_wolfram,
        cut_statistics_test_only,
        extract_data_CRM,
        extract_data_ensemble,
        extract_data_ftor,
        extract_data_shelf,
        extract_data_wolfram,
        extract_influence_coeff_CRM,
        make_models_stop_well,
        prepare_data_for_ensemble,
    )

    state.statistics = {}
    state.statistics_test_only = {}
    state.statistics_another_models = None
    state.statistics_test_index = pd.DatetimeIndex([])
    state.ensemble_interval = pd.DataFrame(
        index=pd.date_range(state.date_start, state.date_end, freq="D").date
    )
    state.models_weights = {}
    state.adapt_params = {}
    state.buffer = None

    config = ConfigPreprocessor(
        state.field_name,
        state.shops,
        state.date_start,
        state.date_test,
        state.date_end,
    )
    state.was_config = config
    state.was_date_start = state.date_start
    state.was_date_test = state.date_test
    state.was_date_end = state.date_end
    state.was_calc_ensemble = state.calc_ensemble
    state.was_date_test_if_ensemble = state.date_test + dt.timedelta(days=state.ensemble_adapt_period)

    with st.spinner("Подготовка входных данных..."):
        preprocessor = run_preprocessor(config)

    wells_ftor = []
    if state.calc_ftor:
        with st.spinner("Расчет модели пьезопроводности..."):
            ftor = calculate_ftor(preprocessor, well_names, state.constraints)
            wells_ftor = ftor.wells
            extract_data_ftor(ftor, state)

    if state.calc_wolfram:
        forecast_days_number = (state.date_end - state.date_test).days + 1
        with st.spinner("Расчет ML-модели..."):
            wolfram = calculate_wolfram(
                preprocessor,
                well_names,
                forecast_days_number,
                state.estimator_name_group,
                state.estimator_name_well,
                state.is_deep_grid_search,
                state.window_sizes,
                state.quantiles,
            )
            extract_data_wolfram(wolfram, state)
            if not wells_ftor:
                wells_ftor = preprocessor.create_wells_ftor(well_names)
            convert_tones_to_m3_for_wolfram(state, wells_ftor)

    if state.calc_CRM or state.calc_CRMIP:
        if not wells_ftor:
            wells_ftor = preprocessor.create_wells_ftor(well_names)
        with st.spinner("Расчет CRM/CRMIP..."):
            crm = calculate_CRM(
                state.date_start,
                state.date_test,
                state.date_end,
                state.field_name,
                calc_CRM=state.calc_CRM,
                calc_CRMIP=state.calc_CRMIP,
                influence_R=state.CRM_influence_R,
                maxiter=state.CRM_maxiter,
                p_res=state.CRM_p_res,
            )
            if crm is not None:
                if state.calc_CRM:
                    extract_data_CRM(crm.pred_CRM, state, wells_ftor, mode="CRM")
                if state.calc_CRMIP:
                    extract_data_CRM(crm.pred_CRMIP, state, wells_ftor, mode="CRMIP")
                if getattr(crm, "f", None) is not None:
                    extract_influence_coeff_CRM(crm.f, state)
                state.wells_coords_CRM = getattr(crm, "_coordinates", pd.DataFrame())

    if state.calc_shelf:
        with st.spinner("Расчет Shelf / ППТП..."):
            shelf = calculate_shelf(
                state.field_name,
                state.shops,
                well_names,
                state.date_start,
                state.date_test,
                state.date_test,
                state.date_end,
                state.n_days_past,
                state.n_days_calc_avg,
                state.get("change_gtm_info", 0),
            )
            extract_data_shelf(shelf, state)

    if state.calc_ensemble and state.statistics:
        with st.spinner("Расчет байесовского ансамбля..."):
            wells_for_ensemble = selected_wells_normal(state)
            for mode in ("liq", "oil", "gas"):
                input_data = prepare_data_for_ensemble(state, wells_for_ensemble, "true", mode=mode)
                input_data = [item for item in input_data if not item["df"].empty and len(item["df"].columns) > 1]
                if not input_data:
                    continue
                result_test, weights = calculate_ensemble(
                    input_data,
                    state.ensemble_adapt_period,
                    state.interval_probability,
                    state.draws,
                    state.tune,
                    state.chains,
                    state.target_accept,
                    "true",
                )
                state.models_weights[mode] = weights
                for well_name_normal, ensemble_df in result_test.items():
                    extract_data_ensemble(ensemble_df, state, well_name_normal, mode=mode)

    make_models_stop_well(state.statistics, list(state.wellnames_key_normal))
    state.statistics_test_only, state.statistics_test_index = cut_statistics_test_only(state)
    st.success("Расчет завершен.")


def draw_start_page(state: AppState) -> None:
    st.title(APP_TITLE)
    st.write(
        "Выберите месторождение, период, скважины и модели в боковой панели. "
        "После запуска результаты появятся на вкладках карты, аналитики и отдельной скважины."
    )
    if state.statistics:
        st.success(f"Доступны результаты моделей: {', '.join(sorted(state.statistics))}.")
    else:
        st.info("Расчеты в текущей сессии еще не выполнялись.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    state = init_state()
    sync_state_from_legacy_session(st.session_state, state)
    draw_sidebar(state)
    sync_legacy_session_state(st.session_state, state)

    pages = {
        "Старт": lambda _session: draw_start_page(_session.state),
        "Настройки моделей": lambda _session: lazy_page("UI.pages.models_settings")(_session),
        "Карта скважин": lambda _session: lazy_page("UI.pages.wells_map")(_session),
        "Скважина": lambda _session: lazy_page("UI.pages.specific_well")(_session),
        "Аналитика": lambda _session: lazy_page("UI.pages.analytics")(_session),
        "ГТМ": lambda _session: lazy_page("UI.pages.gtm_settings")(_session),
        "Импорт/экспорт": lambda _session: lazy_page("UI.pages.resume_app")(_session),
    }
    page_name = st.sidebar.radio("Раздел", list(pages))
    pages[page_name](st.session_state)


if __name__ == "__main__":
    main()
