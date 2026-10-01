import datetime as dt
import importlib
import inspect
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ORCHESTRATOR_DIR = Path(__file__).resolve().parents[1]
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))


class StartupRegressionTests(unittest.TestCase):
    def test_gtm_and_model_settings_pages_import(self):
        importlib.import_module("UI.pages.gtm_settings")
        importlib.import_module("UI.pages.models_settings")

    def test_gas_decline_rates_are_available(self):
        from frameworks_shelf_algo.class_Shelf.constants import DEC_RATES_GAS
        from frameworks_shelf_algo.class_Shelf.support_functions import get_s_decline_rates_gas

        data = {
            109: {
                DEC_RATES_GAS: {
                    dt.date(2020, 1, 1): 1.0,
                    dt.date(2020, 1, 3): 2.5,
                }
            }
        }

        result = get_s_decline_rates_gas(data, 109, dt.date(2020, 1, 1), dt.date(2020, 1, 4))

        self.assertEqual([1.0, 1.0, 2.5, 2.5], result.tolist())

    def test_run_preprocessor_cache_key_includes_config(self):
        from UI.cached_funcs import run_preprocessor

        wrapped = getattr(run_preprocessor, "__wrapped__", run_preprocessor)
        first_parameter = next(iter(inspect.signature(wrapped).parameters))

        self.assertFalse(first_parameter.startswith("_"))

    def test_preprocessor_data_path_is_module_relative(self):
        import tools_preprocessor.preprocessor as preprocessor_module

        try:
            fake_cwd = ORCHESTRATOR_DIR / ".not_project_cwd"
            with patch("pathlib.Path.cwd", return_value=fake_cwd):
                reloaded = importlib.reload(preprocessor_module)
                expected = Path(reloaded.__file__).resolve().parent / "data"

                self.assertEqual(expected, reloaded.Preprocessor._path_general)
        finally:
            importlib.reload(preprocessor_module)

    def test_specific_well_detects_missing_preprocessor_data(self):
        from UI.pages.specific_well import has_preprocessor_data

        self.assertFalse(has_preprocessor_data(SimpleNamespace(field_name="missing-field")))

    def test_specific_well_expands_all_wells_selection(self):
        from UI.app_state import AppState
        from UI.pages.specific_well import available_specific_wells

        state = AppState(
            selected_wells_norm=["Все скважины"],
            wellnames_key_normal={"109": 109, "101": 101},
        )

        self.assertEqual(["101", "109"], available_specific_wells(state))

    def test_specific_well_filters_unavailable_preprocessor_wells(self):
        from UI.app_state import AppState
        from UI.pages.specific_well import available_specific_wells

        state = AppState(
            selected_wells_norm=["Все скважины"],
            wellnames_key_normal={"100": 100, "109": 109},
        )

        self.assertEqual(["109"], available_specific_wells(state, available_ois_wells={109}))

    def test_specific_well_prefers_imported_statistics_wells(self):
        from UI.app_state import AppState
        from UI.pages.specific_well import available_specific_wells

        state = AppState(
            selected_wells_norm=["Все скважины"],
            wellnames_key_normal={"100": 100},
            statistics={
                "ftor": pd.DataFrame(
                    {
                        "100_liq_true": [10.0],
                        "100_liq_pred": [11.0],
                        "100_oil_true": [4.0],
                        "100_oil_pred": [4.5],
                    }
                )
            },
        )

        self.assertEqual(["100"], available_specific_wells(state, available_ois_wells={109}))

    def test_specific_well_builds_fact_frame_from_imported_statistics(self):
        from UI.pages.specific_well import build_df_chess_from_statistics

        dates = pd.date_range(dt.date(2020, 1, 1), periods=2, freq="D")
        statistics = {
            "ftor": pd.DataFrame(
                {
                    "100_liq_true": [10.0, 12.0],
                    "100_liq_pred": [11.0, 13.0],
                    "100_oil_true": [4.0, 5.0],
                    "100_oil_pred": [4.5, 5.5],
                },
                index=dates,
            )
        }

        df_chess = build_df_chess_from_statistics(statistics, "100")

        self.assertEqual([10.0, 12.0], df_chess["Дебит жидкости"].tolist())
        self.assertEqual([4.0, 5.0], df_chess["Дебит нефти"].tolist())
        self.assertIn("Давление забойное", df_chess.columns)
        self.assertIn("Мероприятие", df_chess.columns)

    def test_legacy_pages_receive_top_level_session_defaults(self):
        from UI.app_state import AppState
        from main_UI import apply_state_defaults, sync_legacy_session_state, sync_state_from_legacy_session

        session = {}
        state = AppState()
        apply_state_defaults(state)
        state.field_name = "Карамовское"
        state.selected_wells_norm = ["Все скважины"]

        sync_legacy_session_state(session, state)

        self.assertEqual(0, session["change_gtm"])
        self.assertEqual(0, session["change_gtm_info"])
        self.assertEqual("Карамовское", session["field_name"])
        self.assertEqual(["Все скважины"], session["selected_wells_norm"])

        session["n_days_past"] = 21
        sync_state_from_legacy_session(session, state)

        self.assertEqual(21, state.n_days_past)

    def test_gtm_page_initializes_missing_session_keys(self):
        from UI.pages.gtm_settings import ensure_gtm_session_defaults

        session = {}

        ensure_gtm_session_defaults(session)

        self.assertEqual(0, session["change_gtm"])
        self.assertEqual(0, session["change_gtm_info"])
        self.assertEqual({}, session["shelf_json"])

    def test_imported_state_is_repaired_and_synced_for_tabs(self):
        from UI.app_state import AppState
        from UI.pages.resume_app import restore_saved_state
        from tools_preprocessor.config import Config

        date_start = dt.date(2020, 1, 1)
        date_test = dt.date(2020, 1, 3)
        date_end = dt.date(2020, 1, 5)
        dates = pd.date_range(date_start, date_end, freq="D")
        statistics = {
            "ftor": pd.DataFrame(
                {
                    "109_liq_true": [10, 11, 12, 13, 14],
                    "109_liq_pred": [10, 11, 12, 12, 13],
                    "109_oil_true": [5, 5, 6, 6, 7],
                    "109_oil_pred": [5, 5, 5, 6, 6],
                    "109_gas_true": [1, 1, 2, 2, 3],
                    "109_gas_pred": [1, 1, 1, 2, 2],
                },
                index=dates,
            )
        }
        saved_state = {
            "statistics": statistics,
            "was_config": Config("missing-field", [], date_start, date_test, date_end),
            "was_date_start": date_start,
            "was_date_test": date_test,
            "was_date_end": date_end,
            "was_calc_ensemble": False,
        }
        state = AppState()
        session = {}

        restore_saved_state(saved_state, state, session)

        self.assertIn("ftor", state.statistics_test_only)
        self.assertEqual(3, len(state.statistics_test_index))
        self.assertEqual(["109"], state.selected_wells_norm)
        self.assertEqual({"109": "109"}, state.wellnames_key_normal)
        self.assertIs(session["statistics"], state.statistics)
        self.assertEqual(["109"], session["selected_wells_norm"])

    def test_wells_map_uses_treemap_when_crm_data_is_missing(self):
        from UI.app_state import AppState
        from UI.pages.wells_map import available_plot_options

        state = AppState()

        self.assertEqual(["TreeMap"], available_plot_options(state))

    def test_treemap_skips_wells_without_selected_mode_columns(self):
        from UI.app_state import AppState
        from UI.pages.wells_map import prepare_data_for_treemap

        dates = pd.date_range(dt.date(2020, 1, 1), periods=2, freq="D")
        state = AppState(
            was_date_test=dt.date(2020, 1, 1),
            exclude_wells=[],
            statistics={
                "ensemble": pd.DataFrame(
                    {
                        "12_liq_true": [10.0, 11.0],
                        "12_liq_pred": [9.0, 10.0],
                        "12_oil_true": [4.0, 5.0],
                        "12_oil_pred": [3.5, 4.5],
                        "13_gas_true": [2.0, 4.0],
                        "13_gas_pred": [1.5, 3.5],
                    },
                    index=dates,
                )
            },
        )

        df = prepare_data_for_treemap(state, "ensemble", ("12", "13"), mode="gas")

        self.assertEqual(["13"], df["wellname"].tolist())
        self.assertEqual([6.0], df["cum_q_gas"].tolist())
        self.assertFalse(df["err_gas"].isna().any())

    def test_treemap_model_options_require_selected_mode_data(self):
        from UI.app_state import AppState
        from UI.pages.wells_map import available_treemap_models

        dates = pd.date_range(dt.date(2020, 1, 1), periods=2, freq="D")
        state = AppState(
            was_date_test=dt.date(2020, 1, 1),
            exclude_wells=[],
            statistics={
                "ensemble": pd.DataFrame(
                    {
                        "12_liq_true": [10.0, 11.0],
                        "12_liq_pred": [9.0, 10.0],
                    },
                    index=dates,
                ),
                "wolfram": pd.DataFrame(
                    {
                        "12_gas_true": [2.0, 4.0],
                        "12_gas_pred": [1.5, 3.5],
                    },
                    index=dates,
                ),
            },
        )

        self.assertEqual(["wolfram"], available_treemap_models(state, "gas", ("12",)))

    def test_treemap_mode_options_require_data(self):
        from UI.app_state import AppState
        from UI.pages.wells_map import available_treemap_mode_labels

        dates = pd.date_range(dt.date(2020, 1, 1), periods=2, freq="D")
        state = AppState(
            was_date_test=dt.date(2020, 1, 1),
            exclude_wells=[],
            statistics={
                "ensemble": pd.DataFrame(
                    {
                        "12_liq_true": [10.0, 11.0],
                        "12_liq_pred": [9.0, 10.0],
                    },
                    index=dates,
                )
            },
        )

        self.assertEqual(["Жидкость"], available_treemap_mode_labels(state, ("12",)))

    def test_treemap_uses_full_frame_when_imported_date_slice_is_empty(self):
        from UI.app_state import AppState
        from UI.pages.wells_map import prepare_data_for_treemap

        dates = pd.date_range(dt.date(2020, 1, 1), periods=2, freq="D")
        state = AppState(
            was_date_test=dt.date(2030, 1, 1),
            exclude_wells=[],
            statistics={
                "ensemble": pd.DataFrame(
                    {
                        "12_liq_true": [10.0, 11.0],
                        "12_liq_pred": [9.0, 10.0],
                    },
                    index=dates,
                )
            },
        )

        df = prepare_data_for_treemap(state, "ensemble", ("12",), mode="liq")

        self.assertEqual(["12"], df["wellname"].tolist())
        self.assertEqual([21.0], df["cum_q_liq"].tolist())

    def test_analytics_ensemble_color_is_visible_on_dark_theme(self):
        from statistics_explorer.plots import draw_statistics, model_plot_color

        color = model_plot_color("ensemble", 5)

        self.assertRegex(color, r"^#[0-9A-Fa-f]{6}$")
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue

        self.assertGreater(luminance, 120)

        models = ["ftor", "wolfram", "CRM", "fedot", "shelf", "ensemble"]
        model_names = {
            "ftor": "Piezo",
            "wolfram": "ML",
            "CRM": "CRM",
            "fedot": "CRM+ML",
            "shelf": "PPTP",
            "ensemble": "Ensemble",
        }
        dates = pd.date_range(dt.date(2020, 1, 1), periods=2, freq="D")
        values = {model: [1.0, 2.0] for model in models}

        fig = draw_statistics(
            models,
            values,
            values,
            values,
            values,
            "field",
            dates,
            model_names,
            "Debit liquid",
        )
        ensemble_trace = next(trace for trace in fig.data if trace.name == "Ensemble")

        self.assertEqual(color, ensemble_trace.line.color)
        self.assertEqual(color, ensemble_trace.marker.color)


if __name__ == "__main__":
    unittest.main()
