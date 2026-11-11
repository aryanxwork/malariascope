import numpy as np
import pytest

from malariascope.data import Sample, patient_id_from_filename, preprocess_bytes, split_dataset


@pytest.mark.parametrize(
    "name, expected",
    [
        ("C33P1thinF_IMG_20150619_114756a_cell_179.png", "P1"),
        ("C100P61ThinF_IMG_20150918_144104_cell_162.png", "P61"),
        ("C47P8thinOriginal_IMG_20150621_112116_cell_204.png", "P8"),
        ("C1_thinF_IMG_20150604_104722_cell_9.png", "H-C1"),
        ("random_name.png", "U-random_name"),
    ],
)
def test_patient_id_parsing(name, expected):
    assert patient_id_from_filename(name) == expected


def _fake_samples(n_patients=40, per=20):
    out = []
    for p in range(n_patients):
        for i in range(per):
            out.append(Sample(f"P{p}_{i}.png", int(i % 2 == 0 or p % 5 == 0), f"P{p}"))
    return out


def test_patient_split_has_no_leakage_and_covers_everything():
    samples = _fake_samples()
    splits = split_dataset(samples, "patient", n_folds=7, seed=1)
    pats = {k: {s.patient for s in v} for k, v in splits.items()}
    assert not (pats["train"] & pats["val"])
    assert not (pats["train"] & pats["test"])
    assert not (pats["val"] & pats["test"])
    assert sum(len(v) for v in splits.values()) == len(samples)
    assert all(len(v) > 0 for v in splits.values())


def test_cell_split_leaks_patients():
    """The ablation split *does* mix patients — that's the point of comparing it."""
    splits = split_dataset(_fake_samples(), "cell", n_folds=7, seed=1)
    pats = {k: {s.patient for s in v} for k, v in splits.items()}
    assert pats["train"] & pats["test"]


def test_split_is_deterministic():
    a = split_dataset(_fake_samples(), "patient", seed=3)
    b = split_dataset(_fake_samples(), "patient", seed=3)
    assert [s.path for s in a["test"]] == [s.path for s in b["test"]]


def test_preprocess_bytes_shape_and_range():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((90, 140, 3), 200, np.uint8)).save(buf, format="PNG")
    x = preprocess_bytes(buf.getvalue(), 64)
    assert x.shape == (64, 64, 3) and x.dtype == np.float32
    assert 0 <= x.min() and x.max() <= 255
    assert x[0, 32].sum() == 0  # letterbox padding is black, aspect ratio preserved
