import numpy as np

from fmi_schema import MatchPackage, PitchDims


def test_pitch_keypoints_shape_and_symmetry():
    k = PitchDims().keypoints()
    assert k.shape == (32, 2)
    assert np.isclose(k[:, 0].min(), -52.5) and np.isclose(k[:, 0].max(), 52.5)
    assert np.isclose(k[:, 1].min(), -34) and np.isclose(k[:, 1].max(), 34)
    assert np.allclose(np.sort(k[:, 0]), np.sort(-k[:, 0]))  # mirror symmetric
    assert len(PitchDims().line_samples()) > 300


def test_package_roundtrip(tmp_path, passing_package):
    pkg = passing_package
    pkg.derived["summary"] = {"x": 1.5, "nan": float("nan")}
    assert pkg.validate() == []
    pkg.save(tmp_path / "p")
    back = MatchPackage.load(tmp_path / "p")
    assert back.meta.match_id == pkg.meta.match_id
    assert len(back.tracking) == len(pkg.tracking)
    assert back.derived["summary"] == {"x": 1.5, "nan": None}
    assert back.player_name("a") == "Ann (#7)"
    assert back.frame_at(1, 2.0) == 20
