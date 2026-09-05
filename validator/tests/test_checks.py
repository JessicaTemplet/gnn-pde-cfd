"""Plain-assertion tests for the validator's check logic, using tiny
synthetic tensors rather than the real (slow-to-load) shipped checkpoints.
No test framework dependency, matching the rest of this repo -- run directly:

    python validator/tests/test_checks.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from validator.checks.accuracy import check_accuracy
from validator.checks.stability import check_stability


def test_stability_passes_on_bounded_trajectory():
    reference = torch.randn(4, 10, 8)
    preds = reference + 0.01 * torch.randn(4, 10, 8)
    result = check_stability(preds, reference, divergence_scale_factor=20.0)
    assert result.passed
    assert result.n_diverged == 0
    assert result.n_nan_inf == 0
    print("PASS: test_stability_passes_on_bounded_trajectory")


def test_stability_flags_nan():
    reference = torch.randn(2, 5, 4)
    preds = reference.clone()
    preds[0, -1, 0] = float("nan")
    result = check_stability(preds, reference)
    assert not result.passed
    assert result.n_nan_inf == 1
    print("PASS: test_stability_flags_nan")


def test_stability_flags_blowup_without_nan():
    """Mirrors the real hyperbolic v3 failure: no literal NaN/Inf, but a
    final magnitude far beyond the reference scale."""
    reference = torch.ones(1, 3, 4) * 3.0
    preds = reference.clone()
    preds[0, -1, :] = 1e6
    result = check_stability(preds, reference, divergence_scale_factor=20.0)
    assert not result.passed
    assert result.n_diverged == 1
    assert result.n_nan_inf == 0
    print("PASS: test_stability_flags_blowup_without_nan")


def test_accuracy_zero_error_on_identical_trajectory():
    reference = torch.rand(3, 6, 5) + 0.1
    result = check_accuracy(reference.clone(), reference, dt=0.1)
    assert result.final_rel_err < 1e-5
    assert result.passed
    print("PASS: test_accuracy_zero_error_on_identical_trajectory")


def test_accuracy_threshold_failure_is_reported():
    reference = torch.ones(2, 4, 3)
    preds = reference * 2.0  # 100% relative error at every step
    result = check_accuracy(preds, reference, dt=0.1, max_rel_l2_final=0.5)
    assert not result.passed
    assert len(result.threshold_failures) == 1
    assert "exceeds threshold 0.5" in result.threshold_failures[0]
    print("PASS: test_accuracy_threshold_failure_is_reported")


def test_accuracy_checkpoint_threshold_at_specific_time():
    reference = torch.ones(1, 11, 2)
    preds = reference.clone()
    preds[:, 5:, :] = 3.0  # error appears halfway through, at t=0.5
    result = check_accuracy(preds, reference, dt=0.1, max_rel_l2_at={0.5: 0.1})
    assert not result.passed
    assert any("t=0.5" in msg for msg in result.threshold_failures)
    print("PASS: test_accuracy_checkpoint_threshold_at_specific_time")


if __name__ == "__main__":
    test_stability_passes_on_bounded_trajectory()
    test_stability_flags_nan()
    test_stability_flags_blowup_without_nan()
    test_accuracy_zero_error_on_identical_trajectory()
    test_accuracy_threshold_failure_is_reported()
    test_accuracy_checkpoint_threshold_at_specific_time()
    print("\nAll validator check tests passed.")
