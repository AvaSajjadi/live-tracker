"""Deterministic alert rules; no LLM call required for event timing."""
import numpy as np


def threshold_events(probabilities, fps=30.0, threshold=0.75, min_duration_s=2.0):
    """Find contiguous high-probability intervals. End indices are exclusive."""
    p = np.asarray(probabilities, dtype=float)
    active = np.isfinite(p) & (p >= threshold)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    min_frames = int(np.ceil(min_duration_s * fps))
    return [{'start_frame': int(start), 'end_frame': int(end),
             'start_s': float(start / fps), 'end_s': float(end / fps),
             'duration_s': float((end - start) / fps),
             'mean_probability': float(np.mean(p[start:end]))}
            for start, end in zip(starts, ends) if end - start >= min_frames]
