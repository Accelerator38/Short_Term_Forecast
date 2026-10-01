from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_ensemble.calculator import Calculator
from models_ensemble.config import Config


def build_input() -> list[dict]:
    dates = pd.date_range("2022-01-01", periods=18, freq="D")
    base = np.linspace(100.0, 116.0, len(dates))
    df = pd.DataFrame(
        {
            "true": base,
            "ftor": base * 0.97 + 2.0,
            "wolfram": base * 1.02 - 1.5,
        },
        index=dates,
    )
    return [{"wellname": "SMOKE_WELL", "df": df}]


def main() -> None:
    config = Config(
        adaptation_days_number=7,
        interval_probability=0.8,
        draws=20,
        tune=10,
        chains=1,
        cores=1,
        target_accept=0.9,
        name_of_y_true="true",
    )
    calculator = Calculator(config, build_input(), n_workers=1, logging=False)
    result = calculator.result_test["SMOKE_WELL"]
    weights = calculator.weights["SMOKE_WELL"]
    print("ENSEMBLE_SMOKE_OK")
    print(f"rows={len(result)}")
    print(f"columns={','.join(result.columns)}")
    print(f"weights={weights}")


if __name__ == "__main__":
    main()
