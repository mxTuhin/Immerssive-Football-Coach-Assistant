"""Adapters on the bundled open data (skipped when the data folder is absent)."""

import pytest

from conftest import DATA

pytestmark = pytest.mark.data


@pytest.mark.skipif(not (DATA / "skillcorner").exists(), reason="no SkillCorner data")
def test_skillcorner_metadata_and_events():
    from fmi_analytics.common import attack_sign
    from fmi_schema.adapters import skillcorner

    pkg = skillcorner.load_match(DATA / "skillcorner", 1886347, with_tracking=False)
    assert pkg.meta.home.name == "Auckland FC" and pkg.meta.pitch_length == 104
    assert [p.home_attack_sign for p in pkg.meta.periods] == [-1, 1]
    shots = pkg.events[pkg.events["type"] == "shot"]
    assert len(shots) > 10
    # event coordinates were flipped into the tracking frame: shots happen near the attacked goal
    nx = [r.start_x * attack_sign(pkg.meta, int(r.period), r.team) for r in shots.itertuples()]
    assert sum(x > 20 for x in nx) / len(nx) > 0.8


@pytest.mark.skipif(not (DATA / "metrica").exists(), reason="no Metrica data")
def test_metrica_events():
    from fmi_schema.adapters import metrica

    ev = metrica.read_events_csv(DATA / "metrica/Sample_Game_1/Sample_Game_1_RawEventsData.csv", {1: 0.04, 2: 0.04})
    passes = ev[ev["type"].isin(["pass", "cross"])]
    rate = (passes["outcome"] == "success").mean()
    assert 0.7 < rate < 0.95
    assert ev["start_x"].abs().max() <= 60
