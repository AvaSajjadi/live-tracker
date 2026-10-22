# 30 fps seconds ↔ frames rule (Contract v1, shared surface item 2)

**Status:** final. This is Ava's rule: round half up at both window boundaries. The code is
`src/live_tracker/timebase.py`; the tests are in `tests/test_timebase.py`.

## Rule

Frames are canonical inside the code, and seconds appear only at the tool boundary. Frame `f` (0-based within its
session) has the timestamp `f / 30`, the same value as `time_s` in INTERFACES.md §2 and in Igor's score export.

```
frame(t)          = floor(t · 30 + 0.5)                  nearest frame, halves round up
window [t0_s, t1_s) = frames frame(t0_s) … frame(t1_s) − 1
```

The expression is evaluated literally in IEEE float64 (`math.floor(t * 30 + 0.5)`). That is the same arithmetic as
NumPy or plain Python, so every implementation that writes the formula this way gets the same frames for the same
inputs. No tolerance or exact-fraction step is added.

**Equivalent reading:** frame `f` is in `[t0_s, t1_s)` exactly when `30·t0_s < f + 0.5 ≤ 30·t1_s`. In other words,
the halfway point between frame `f` and frame `f + 1` falls in `(t0_s, t1_s]`.

## Examples

| t (s) | 30·t | frame(t) |
|---|---|---|
| 0.0 | 0 | 0 |
| 1.0 | 30 | 30 |
| 0.01 | 0.3 | 0 |
| 0.017 | 0.51 | 1 |
| 0.05 | 1.5 | **2** (exact half: up) |
| 1.02 | 30.6 | 31 |
| 2.98 | 89.4 | 89 |

| window | frames |
|---|---|
| `[1.0, 2.0)` | `[30, 60)` |
| `[0.05, 0.1)` | `[2, 3)` |
| `[0.0, 0.05)` | `[0, 2)` |
| `[0.01, 0.02)` | `[0, 1)` |
| `[0.02, 0.03)` | empty: refused (`bad_window`) |

## Properties (tested)

- **Round trip:** `frame(f/30) == f`. All frames below 200,000 are checked, plus 5,000 random frames up to 10⁷.
- **No guard needed for timestamp error.** Floating-point error in a timestamp is far below half a frame. The
  monitor's `end_s = t0 + k·(1/30)` and values written with only 6 decimals both map back to their own frame.
- **Monotone:** a later time never maps to an earlier frame. So adjacent windows `[a, b)` and `[b, c)` partition
  `[a, c)`, including when `b` is an exact half frame.
- **Matches the monitor:** on a monitored synthetic session, `start_s` maps to the bout's first frame, `fired_s` to
  its firing frame, and the exclusive `end_s` to `last_frame + 1`. So `[start_s, end_s)` is exactly the bout's
  frames.
- **Literal formula:** the code gives exactly `floor(t·30 + 0.5)` in float64 on 26,000 random and grid times.

## Validation (error code `bad_window`)

A window is refused when:
- a boundary is not a finite number (or is a bool), or `30·t` overflows;
- `t0_s < 0`;
- `t1_s <= t0_s`;
- both ends round to the same frame, so the window has no frame (e.g. `[0.02, 0.03)`);
- `frame(t1_s)` is past the session's `n_frames`. A `t1_s` that rounds to `n_frames` itself is allowed.

## Windows from an anchor and durations

`compare_windows` takes `pre_s`, `baseline_s` and `baseline_gap_s` in seconds. It forms each window's **absolute
endpoints in seconds first**, then converts **each endpoint** with `frame(t)`:

```
pre      = [frame(start_s − pre_s),                         frame(start_s))
baseline = [frame(start_s − pre_s − gap_s − baseline_s),    frame(start_s − pre_s − gap_s))
```

The subtraction is done in float64 seconds. The durations are never rounded to frame counts on their own and
subtracted from a frame, because `frame(a − d)` can differ from `frame(a) − frame(d)`:

| case | correct (endpoints) | wrong (subtract counts) |
|---|---|---|
| `start_s` 12, `pre_s` 2, gap 0, `baseline_s` 0.05 | baseline `[frame(9.95), frame(10.0))` = `[299, 300)` | `[300 − 2, 300)` = `[298, 300)` |
| gap 0.05 s | baseline end `frame(9.95)` = 299 | `300 − frame(0.05)` = 298 |
| `pre_s` 1.95 | pre start `frame(10.05)` = 302 | `360 − frame(1.95)` = 301 |

**Validation:**
- `pre_s` and `baseline_s` must be finite and > 0 s; `baseline_gap_s` must be finite and ≥ 0 s. Otherwise
  `bad_window`.
- A window whose two converted endpoints are equal (or reversed) has no frame, and the request is `bad_window`. For
  example, `baseline_s` 0.01 gives `[frame(9.99), frame(10.0))` = `[300, 300)`.
- A baseline starting before 0 s excludes that bout (`window_before_session_start`).

With a zero gap, the baseline's end frame is the pre window's start frame, so the two windows are adjacent, sharing
no frame and missing none.

`timebase.duration_frames` converts a lone duration by the same rule. It is not used to build windows.

## Note for Ava's scripts

`scripts/dryrun_usable_windows.py` and `scripts/compare_quality_policies.py` (on `main`) use `int(round(t * 30))`.
Python's `round` rounds exact halves to the even integer: `round(2.5) == 2`, while this rule gives 3. Every time those
scripts convert is a whole number of frames (ground-truth bout edges and the 1/2/5 s grid), so their published counts
are unaffected. Where a time could fall on an exact half, `timebase.boundary_frame` or `window_frames` give the agreed
result. Syed has not changed these scripts; they are Ava's.
