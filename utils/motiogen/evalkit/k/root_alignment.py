"""Explicit geometric root anchors; no evaluation reference enters fitting.

The bilateral hip midpoint is a geometric landmark, not a centre of mass or
an anatomical pelvis origin. Translation never supplies missing rotations.
"""
import numpy as np


def hip_midpoint(points, names):
    return .5 * (points[:, names.index('left_hip')] + points[:, names.index('right_hip')])


def anchor_target(points, names, source_midpoint):
    """Exporter-only root placement on unchanged target offsets and rotations."""
    delta = np.asarray(source_midpoint) - hip_midpoint(points, names)
    if not np.isfinite(delta).all():
        raise ValueError('Hip-midpoint export requires finite bilateral source observations')
    return points + delta[:, None], delta


def align_fitted_root(model_path, times, labels, coordinates, source, names, prior=()):
    """Translate the model to observed hip midpoints, preserving every joint angle.

    Missing/prior hip observations keep the prior result and an invalid mask.
    A measured translation Jacobian handles the model's actual coordinate axes;
    FK verifies that every output landmark undergoes the same rigid translation.
    """
    import opensim as osim
    from .fitting3 import positions
    required = ('left_hip', 'right_hip')
    original = coordinates.copy()
    fitted, fit_names = positions(model_path, times, labels, original)
    valid = np.zeros(len(times), dtype=bool)
    shift = np.zeros((len(times), 3))
    receipt = {'anchor': 'bilateral_hip_midpoint', 'source': 'observed_input_only',
               'scope': 'geometric translation; angles retained; force eligibility unchanged'}
    if any(n not in names or n in prior for n in required):
        receipt.update(status='unavailable', reason='missing_or_prior_hip_observation')
        return original, fitted, valid, shift, receipt
    wanted = hip_midpoint(source, names)
    actual = hip_midpoint(fitted, fit_names)
    valid = np.isfinite(wanted).all(1) & np.isfinite(actual).all(1)
    root_names = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    model = osim.Model(str(model_path)); state = model.initSystem()
    cs = model.getCoordinateSet()
    if any(n not in labels or not cs.contains(n) or cs.get(n).getLocked(state)
           or cs.get(n).isConstrained(state) or cs.get(n).getMotionType() != 2
           for n in root_names):
        receipt.update(status='unavailable', reason='root_translation_coordinates_unavailable')
        return original, fitted, np.zeros(len(times), bool), shift, receipt
    columns = [labels.index(n) for n in root_names]
    jacobian = np.empty((len(times), 3, 3))
    epsilon = 1e-4
    for axis, column in enumerate(columns):
        perturbed = original.copy(); perturbed[:, column] += epsilon
        moved, _ = positions(model_path, times, labels, perturbed)
        jacobian[:, :, axis] = (hip_midpoint(moved, fit_names) - actual) / epsilon
    valid &= np.isfinite(jacobian).all((1, 2))
    valid &= np.abs(np.linalg.det(jacobian)) > 1e-8
    shift[valid] = wanted[valid] - actual[valid]
    correction = np.zeros_like(shift)
    correction[valid] = np.linalg.solve(jacobian[valid], shift[valid, :, None])[..., 0]
    result = original.copy(); result[:, columns] += correction
    final, final_names = positions(model_path, times, labels, result)
    if final_names != fit_names:
        raise ValueError('Root alignment changed landmark ordering')
    np.testing.assert_allclose(final, fitted + shift[:, None], atol=1e-8, rtol=0)
    np.testing.assert_allclose(hip_midpoint(final, fit_names)[valid], wanted[valid], atol=1e-8, rtol=0)
    receipt.update(status='computed', valid_frames=int(valid.sum()), total_frames=len(times),
                   maximum_shift_m=float(np.linalg.norm(shift, axis=1).max()),
                   reason='finite observed bilateral hips; missing frames retain prior translation')
    return result, final, valid, shift, receipt
