"""Fast unit tests for VecBookIndex target aggregation modes (barycenter vs max).

The embedding model is replaced with a fake so these run without loading
sentence-transformers; only the aggregation math is exercised.
"""

from pathlib import Path

import numpy as np
import pytest

from evolvattention.vecx.vecbook_index import VecBookIndex


def _index_with_fake_model(vectors):
    """A VecBookIndex whose model.encode returns preset vectors keyed by text."""

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([vectors[t] for t in texts], dtype=np.float32)

    index = VecBookIndex(Path("test_data"))
    index.model = FakeModel()  # _initialize_model is a no-op once model is set
    return index


def test_max_mode_scores_nearest_target_above_barycenter():
    # Orthogonal targets; X is identical to target A.
    vectors = {
        "A": [1.0, 0.0],
        "B": [0.0, 1.0],
        "X": [1.0, 0.0],
    }
    index = _index_with_fake_model(vectors)
    assert index.set_target_strings(["A", "B"])["status"] == "success"

    index.set_target_mode("barycenter")
    bary = float(index.compare_against_barycenter(["X"])[0]["cosine_similarity"])

    index.set_target_mode("max")
    nearest = float(index.compare_against_barycenter(["X"])[0]["cosine_similarity"])

    # X sits on A, so max == 1.0, while the blended midpoint is ~0.707.
    assert nearest == pytest.approx(1.0, abs=1e-5)
    assert bary == pytest.approx(0.70710, abs=1e-4)
    assert nearest > bary


def test_default_target_mode_is_barycenter():
    assert _index_with_fake_model({}).target_mode == "barycenter"


def test_set_target_mode_rejects_unknown():
    with pytest.raises(ValueError):
        _index_with_fake_model({}).set_target_mode("median")
