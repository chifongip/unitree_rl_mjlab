"""Fixed-foot X2 squat generation, export, and discrete-height regression checks."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
generator = importlib.import_module("generate_knees_out_postures")
playback = importlib.import_module("playback_height_postures")
sys.path.pop(0)


@pytest.fixture(scope="module")
def model_data():
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg())
  reference, _ = playback.load_python_frames(SCRIPTS / "postures_x2_recorded.py", solver.model)
  corrected, _ = playback.load_python_frames(SCRIPTS / "postures_x2_knees_out.py", solver.model)
  solver.prepare(reference)
  return solver, reference, corrected


def test_shipped_squat_preserves_fixed_feet_waist_and_support(model_data):
  solver, reference, corrected = model_data
  assert [f.height_label for f in corrected] == [f.height_label for f in reference]
  for frame in corrected:
    _, original = solver.reference_at(frame.height_label)
    for name in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"):
      assert frame.positions[name] == original[name]
    m = solver.measure(frame)
    assert solver.acceptable(m), m
    assert m["foot_width_m"] == pytest.approx(.374, abs=.0002)
    assert m["toe_out_deg"] == pytest.approx([30., 30.], abs=.05)
    q = np.array([frame.positions[n] for n in generator.LOWER_BODY_JOINTS])
    np.testing.assert_allclose(q[:6], generator.MIRROR * q[6:], atol=1e-10)
  assert corrected[-1].positions != reference[-1].positions


def test_solver_recovers_lowest_pose_and_exports_exact_waist(model_data):
  solver, reference, _ = model_data
  solution = solver.solve(.22, [])
  assert solver.acceptable(solution.metrics)
  assert abs(solution.metrics["pelvis_fore_aft_shift_m"]) <= solver.cfg.max_pelvis_shift
  namespace = {}
  exec(generator.format_corrected_postures({.22: solution.frame.positions}, Path("reference.py")), namespace)
  for name in ("waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"):
    assert namespace["HEIGHT_POSTURES"][.22][name] == reference[0].positions[name]


def test_infeasible_stance_is_rejected(model_data):
  _, reference, _ = model_data
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg(
    stance_width=2., max_pelvis_shift=0))
  solver.prepare(reference)
  with pytest.raises(RuntimeError, match="No feasible fixed-foot posture"):
    solver.solve(.64, [])


@pytest.mark.parametrize("limit", [.006, .007])
def test_offset_search_includes_non_grid_limit(model_data, limit):
  _, reference, _ = model_data
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg(max_pelvis_shift=limit))
  solver.prepare(reference)
  solution = solver.solve(.22, [])
  assert solver.acceptable(solution.metrics)
  assert solution.dx == limit


def test_offset_search_refines_before_rejecting(model_data, monkeypatch):
  _, reference, _ = model_data
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg(max_pelvis_shift=.007))
  solver.prepare(reference)
  acceptable = solver.acceptable
  # Restrict acceptance to a feasible slice missed by the coarse grid and bounds.
  monkeypatch.setattr(solver, "acceptable", lambda m: acceptable(m)
    and abs(m["pelvis_fore_aft_shift_m"] - .00625) < .00005)
  solution = solver.solve(.22, [])
  assert acceptable(solution.metrics)
  assert abs(solution.dx - .00625) < .00005


def test_additional_joints_survive_generation_and_export(tmp_path, model_data):
  _, reference, _ = model_data
  positions = dict(reference[-1].positions, left_elbow_joint=-.2, right_elbow_joint=-.2)
  path = tmp_path / "full_reference.py"
  path.write_text(generator.format_corrected_postures({.64: positions}, path))
  table, diagnostics = generator.generate(path, generator.CorrectionCfg())
  namespace = {}
  exec(generator.format_corrected_postures(table, path), namespace)
  solver = generator.RecordedPostureCorrector(generator.CorrectionCfg())
  frames, _ = playback.load_python_frames(path, solver.model)
  solver.prepare(frames)
  for name in ("left_elbow_joint", "right_elbow_joint"):
    assert table[.64][name] == namespace["HEIGHT_POSTURES"][.64][name] == -.2
  frame = playback.PostureFrame(.64, .64, namespace["HEIGHT_POSTURES"][.64])
  assert solver.acceptable(solver.measure(frame))
  assert diagnostics[0]["float32_validation"]["checks"] == 1


@pytest.mark.parametrize("settings", [
  {"knee_outward": -.1}, {"toe_out_deg": float("nan")},
  {"start_height": .4, "full_splay_height": .5}, {"stance_width": 0},
  {"max_pelvis_shift": -.01},
])
def test_invalid_settings(settings):
  with pytest.raises(ValueError):
    generator.CorrectionCfg(**settings)


def test_invalid_reference_rejected(model_data):
  solver, reference, _ = model_data
  with pytest.raises(ValueError, match="not grounded"):
    solver.reference(playback.PostureFrame(.1, .1, reference[0].positions))
  positions = dict(reference[-1].positions, waist_roll_joint=.01)
  with pytest.raises(ValueError, match="symmetric"):
    solver.reference(playback.PostureFrame(.64, .64, positions))


def test_generation_preserves_source_and_solves_standing(tmp_path, model_data):
  _, reference, _ = model_data
  path = tmp_path / "reference.py"
  text = generator.format_corrected_postures({.64: reference[-1].positions}, path)
  path.write_text(text)
  table, diagnostics = generator.generate(path, generator.CorrectionCfg())
  assert path.read_text() == text
  assert table[.64] != reference[-1].positions
  assert diagnostics[0]["float32_validation"]["checks"] == 1


def test_failed_generation_does_not_replace_outputs(tmp_path, model_data):
  import subprocess
  _, reference, _ = model_data
  source = tmp_path / "source.py"
  source.write_text(generator.format_corrected_postures({.64: reference[-1].positions}, source))
  output, report = tmp_path / "output.py", tmp_path / "report.json"
  output.write_text("existing output")
  report.write_text("existing report")
  completed = subprocess.run([
    sys.executable, str(SCRIPTS / "generate_knees_out_postures.py"),
    "--reference", str(source), "--output", str(output), "--report", str(report),
    "--stance-width", "2", "--max-pelvis-shift", "0", "--overwrite",
  ], capture_output=True, text=True)
  assert completed.returncode != 0
  assert "No feasible fixed-foot posture" in completed.stderr
  assert output.read_text() == "existing output"
  assert report.read_text() == "existing report"
