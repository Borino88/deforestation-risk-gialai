import numpy as np
import pytest
from src.pipeline import _best_f1_threshold, _capture_rate, _rename_ci
import pandas as pd

def test_rename_ci():
    df = pd.DataFrame({'COL1': [1, 2], 'col2': [3, 4]})
    renamed = _rename_ci(df, ['col1', 'COL2'])
    assert 'col1' in renamed.columns
    assert 'COL2' in renamed.columns

def test_capture_rate():
    y_true = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.85, 0.15, 0.05, 0.5])
    # Top 30% = top 3 items (indices 0, 6, 2). All 3 are actual positives (sum=3).
    # Total actual positives = 3. Capture rate should be 1.0 (100%).
    rate = _capture_rate(y_true, scores, 0.3)
    assert np.isclose(rate, 1.0)

def test_best_f1_threshold():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.4])
    thr = _best_f1_threshold(y_true, scores)
    assert 0.0 <= thr <= 1.0
