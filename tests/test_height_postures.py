"""Recorded-posture corrections, grounding and independent export."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# These command-line utilities import sibling scripts when run directly.
sys.path.insert(0, str(SCRIPTS))
generator = importlib.import_module("generate_knees_out_postures")
resampler = importlib.import_module("resample_height_postures")
playback = importlib.import_module("playback_height_postures")
sys.path.pop(0)


@pytest.fixture(scope="module")
def model_data():
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg())
  reference, addresses = playback.load_python_frames(SCRIPTS / "postures_x2_recorded.py", solver.model)
  corrected, _ = playback.load_python_frames(SCRIPTS / "postures_x2_knees_out.py", solver.model)
  return solver, reference, corrected, addresses


def test_shipped_corrections_preserve_recording_and_clear_knees(model_data):
  solver, reference, corrected, addresses = model_data
  assert [f.height_label for f in reference] == [f.height_label for f in corrected]
  for original, new in zip(reference, corrected):
    solver.reference(original)
    ref_rotation = solver.data.qpos[3:7].copy()
    ref_knees = solver.data.xpos[solver.knees] - solver.data.xpos[solver.pelvis]
    for name in resampler.WAIST_JOINTS:
      assert new.positions[name] == original.positions[name]
    for name, value in new.positions.items():
      assert solver.model.joint(name).range[0] <= value <= solver.model.joint(name).range[1]
    height, spread = playback.apply_grounded_frame(
      solver.model, solver.data, new, addresses, solver.feet,
      *solver.ankles, solver.pelvis)
    knees, points, _ = solver.geometry()
    knees -= solver.data.xpos[solver.pelvis]
    blend = solver.cfg.blend(original.height_label)
    assert height == pytest.approx(original.height_label, abs=0.0002)
    assert spread < 0.0002
    rotation_error = 2 * np.arccos(np.clip(abs(np.dot(ref_rotation, solver.data.qpos[3:7])), 0, 1))
    assert np.rad2deg(rotation_error) < 0.05
    assert np.all(np.array([1, -1]) * (knees[:, 1] - ref_knees[:, 1])
                  >= solver.cfg.knee_outward * blend - 0.0002)
    assert np.all(ref_knees[:, 0] - knees[:, 0] >= solver.cfg.knee_retraction * blend - 0.0002)
    assert all(c.dist >= -0.001 for c in solver.data.contact[:solver.data.ncon])
    hull = generator.ConvexHull(points[:, :, :2].reshape(-1, 2))
    com = solver.data.subtree_com[solver.pelvis, :2]
    assert np.max(hull.equations[:, :2] @ com + hull.equations[:, 2]) < 0
    for joint in ("hip_pitch", "knee", "ankle_pitch"):
      for side in ("left", "right"):
        name = f"{side}_{joint}_joint"
        assert abs(new.positions[name] - original.positions[name]) <= solver.cfg.max_pitch_change
    if blend == 0:
      assert new.positions == original.positions
  deltas = np.array([[new.positions[name] - original.positions[name]
                      for name in generator.LOWER_BODY_JOINTS]
                     for original, new in zip(reference, corrected)])
  # The source has a sharp pitch change at its lowest sample; constrain the
  # additional correction rather than smoothing away the recorded motion.
  assert np.max(np.abs(np.diff(deltas, axis=0))) < 0.10


def test_solver_uses_recorded_waist_and_exports_it_exactly(model_data):
  _, reference, _, _ = model_data
  original = next(f for f in reference if f.height_label == 0.4)
  positions = dict(original.positions, waist_pitch_joint=0.123456789)
  frame = playback.PostureFrame(0.4, 0.4, positions)
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg())
  posture, metrics = solver.solve(frame)
  namespace = {}
  exec(generator.format_corrected_postures({0.4: posture}, Path("recorded.py")), namespace)

  assert namespace["HEIGHT_POSTURES"][0.4]["waist_pitch_joint"] == 0.123456789
  assert min(metrics["knee_outward_m"]) >= 0.10 - 0.0002
  assert min(metrics["knee_retraction_m"]) >= 0.03 - 0.0002
  assert metrics["pelvis_rotation_change_deg"] < 0.05


def test_zero_correction_retains_recorded_posture(model_data):
  _, reference, _, _ = model_data
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg(
    knee_outward=0, knee_retraction=0, toe_out_deg=0, foot_outward=0, smoothness=0))
  original = next(f for f in reference if f.height_label == 0.4)
  posture, _ = solver.solve(original)
  assert posture == original.positions


def test_infeasible_correction_is_rejected(model_data):
  _, reference, _, _ = model_data
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg(knee_outward=1.0))
  original = next(f for f in reference if f.height_label == 0.4)
  with pytest.raises(RuntimeError, match="No feasible recorded-posture correction"):
    solver.solve(original)


@pytest.mark.parametrize("settings", [
  {"knee_outward": -0.1}, {"toe_out_deg": float("nan")},
  {"start_height": 0.4, "full_splay_height": 0.5}, {"max_pitch_change": 0},
])
def test_invalid_correction_settings_fail(settings):
  with pytest.raises(ValueError):
    generator.CorrectionCfg(**settings)


def test_incorrect_recorded_height_is_rejected(model_data):
  solver, reference, _, _ = model_data
  frame = playback.PostureFrame(0.1, 0.1, reference[0].positions)
  with pytest.raises(ValueError, match="not grounded at its labelled height"):
    solver.reference(frame)


def test_generation_leaves_recorded_reference_unchanged(tmp_path, model_data):
  _, reference, _, _ = model_data
  original = next(f for f in reference if f.height_label == 0.64)
  reference_path = tmp_path / "recorded.py"
  text = generator.format_corrected_postures({0.64: original.positions}, reference_path)
  reference_path.write_text(text)

  table, _ = generator.generate(reference_path, generator.CorrectionCfg())

  assert reference_path.read_text() == text
  assert table[0.64] == original.positions
