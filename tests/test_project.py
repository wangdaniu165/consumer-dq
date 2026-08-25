import pandas as pd
import pytest

from src import project


def _simple_model():
    class M:
        params = {"const": 1.0, "dq_lag1": 0.5, "u_lag0": 0.1}

    return M()


def _frame():
    idx = pd.period_range("2020Q1", periods=24, freq="Q")
    return pd.DataFrame(
        {"UNRATE": [0.0] * 24, "DRALACBS": [2.0] * 24, "DRCCLACBS": [2.0] * 24,
         "NYFED_OTHER_90DPD": [2.0] * 24},
        index=idx,
    )


def test_project_recurses_ar_term():
    m = _simple_model()
    path = pd.Series([5.0, 5.0])
    out = project.project(m, _frame(), path, horizon=2)
    # step 0: 1 + 0.5*last_dq(2.0) + 0.1*5 = 1 + 1.0 + 0.5 = 2.5
    assert out.iloc[0] == pytest.approx(1.0 + 0.5 * 2.0 + 0.1 * 5.0)
    # step 1: reuses prior DQ through the AR term
    assert out.iloc[1] == pytest.approx(1.0 + 0.5 * out.iloc[0] + 0.1 * 5.0)


def test_project_rejects_short_path():
    m = _simple_model()
    with pytest.raises(ValueError):
        project.project(m, _frame(), pd.Series([5.0]), horizon=3)


def test_build_scenario_path_step():
    path = project.build_scenario_path(4.0, {"kind": "step", "delta": 5.0}, horizon=3)
    assert list(path) == [9.0, 9.0, 9.0]
