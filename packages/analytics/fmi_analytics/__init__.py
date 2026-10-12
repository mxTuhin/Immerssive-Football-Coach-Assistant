"""Tracking-level algorithms and match analytics.

Everything here works on a ``MatchPackage`` (or its tracking DataFrame) so it runs identically on our own
video pipeline output and on open tracking data (SkillCorner, Metrica).
"""

from fmi_analytics.run import run_all

__all__ = ["run_all"]
