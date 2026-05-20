"""Tests for data loading and preprocessing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
from data.preprocessing import DynamicLabelEncoder


class TestDynamicLabelEncoder:
    def test_fit_and_transform(self):
        enc = DynamicLabelEncoder()
        enc.fit(["alice", "bob", "carol"])
        result = enc.transform(["carol", "alice"]).tolist()
        assert result == [2, 0]

    def test_is_known(self):
        enc = DynamicLabelEncoder()
        enc.fit(["alice", "bob"])
        assert enc.is_known("alice")
        assert not enc.is_known("dave")

    def test_add_new_preserves_existing(self):
        enc = DynamicLabelEncoder()
        enc.fit(["alice", "bob", "carol"])
        n = enc.add_new(["dave", "alice"])   # alice already known
        assert n == 1
        assert enc.transform(["alice"]).tolist() == [0]   # unchanged
        assert enc.transform(["dave"]).tolist() == [3]    # appended

    def test_pickle_round_trip(self):
        import pickle
        enc = DynamicLabelEncoder()
        enc.fit(["x", "y", "z"])
        enc.add_new(["w"])
        restored = pickle.loads(pickle.dumps(enc))
        assert list(restored.classes_) == list(enc.classes_)
        assert restored.transform(["w"]).tolist() == [3]
