"""Deterministic, causal replay of a recorded session (new file, D3-1).

The data model behind pages/2_Replay.py (D3-2). No Streamlit here, no model,
no network. A replay is a recorded session played back, not a live camera.

What it is built from (existing contracts only):
- score rows, INTERFACES.md §2 (session, frame, time_s, <scores>,
  tracking_ok), e.g. fake_scores.build_scores (SYNTHETIC) or a verified drop;
  tracking_ok and scores are parsed by the monitor's own rules;
- the researcher's approved rules (rules.RuleSet / Rule, status approved),
  always executed in rule_id order;
- monitor.Monitor, run once over the session's rows. Its journal records
  (event_fired / event_closed / event_answered) and the §4 projection
  (event_log.project_event) are used as they are: no new event schema;
- tracking warnings from Ava's features.tracking_warning_events, through a
  separate typed adapter (tracking_warnings), because the journal accepts only
  alert / ask / log actions (event_log.ACTIONS) and is not changed here;
- optionally the pose (long table: session, frame, mouse, keypoint, x, y,
  score) for skeleton geometry (skeleton.py), with per-frame CAUSAL quality
  flags computed here (causal_pose_flags).

Causal visibility (what the researcher may know at frame f, i.e. after the
rows up to and including f have been played):
- an event becomes visible on its fired_frame, exactly as it was at firing
  (project_event(fired)); its closure (end_s, end_reason, bout_* statistics)
  on the row at which the monitor detected it: the first recorded row after
  last_frame. A bout still open at the session's last row closes with
  stream_end, visible only in the end state (after the last row);
- a researcher answer is visible from the replay position where it was given
  (answer(..., at_frame=f)), recorded by this layer next to the journal (the
  journal's answer record has no time and is not changed). An answer whose
  position is not known (e.g. written to the journal by something else) is
  timing-unknown and appears only in the end state;
- a tracking warning becomes visible once its stretch has lasted the
  warning's minimum duration (and, for a frozen tracker, once enough identical
  frames were seen to call it frozen); its end on the row of the first good
  frame. Whether it was still going when the session ended is part of the end
  state only: during playback an unfinished warning is just "ongoing";
- keypoint quality on frame f uses frames up to f only: a stuck tracker is
  reported as frozen from the frame on which frozen_min_frames identical
  positions had been seen, never retroactively; a short gap that a later
  interpolation fills is missing on the frame itself;
- seeking is a pure function of the frame: seeking back hides everything that
  had not happened yet; the same frame always gives the same state.

Identity: one monitor run per session. run_id = replay_run_id(...) binds the
session, the frame rate, the approved rules (their monitor semantics) and the
canonical score rows exactly as fed to the monitor (column names, types and
values, in feed order). Event IDs (E-<run_id>-<n>) therefore never collide
across sessions, rule sets, score schemas or score files, and replaying the
same session with other rules is a new run, not a JournalConflictError.
Warning IDs are W-<session digest>-<flag>-<mouse>-<start frame> (Ava's T-IDs
restart per call; they are kept as source_event_id).

Tracking gaps: a frame with tracking_ok False is a tracking gap, reported as
such (tracking_note); positions and motion there are unknown, never
"stationary", and no keypoint of that frame is drawn as tracked.
"""

from __future__ import annotations

import bisect
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import event_log as el
from .calms21 import TORSO_KEYPOINTS
from .fake_scores import SYNTHETIC_PREFIX
from .features import tracking_warning_events
from .monitor import DEFAULT_FPS, Monitor, MonitorInputError, _parse_score, _parse_tracking, normalize_rule
from .quality import SUGGESTED_MIN_SCORE
from .rules import RuleSet
from .skeleton import mouse_skeleton

__all__ = ["PLAYBACK_SPEEDS", "REPLAY_BANNER", "FrameState", "ReplayError", "ReplaySession", "TrackingWarning",
           "build_replay", "build_replays", "causal_pose_flags", "replay_run_id", "run_paths", "tracking_warnings"]

REPLAY_BANNER = "Replay of a recorded session, not a live camera"
SYNTHETIC_NOTE = "SYNTHETIC test data: not CalMS21, not predictions of any model"
PLAYBACK_SPEEDS = (1.0, 2.0, 5.0)
TRACKING_GAP_NOTE = ("Tracking unavailable on this frame: positions and motion are unknown "
                     "(a tracking gap, not the mouse standing still)")
# Ava's flags plus low_conf (opt-in in quality.flag_tracking; on real CalMS21 coordinates are never missing,
# tracking problems show up as low confidence). PROVISIONAL until Ava confirms the warning contract.
WARNING_FLAGS = ("missing", "frozen", "out_of_frame", "low_conf")
WARNING_KINDS = {"missing": "missing_coordinates", "low_conf": "low_confidence", "frozen": "frozen_tracker",
                 "out_of_frame": "out_of_frame", "jump": "implausible_jump"}
FROZEN_MIN_FRAMES = 15             # quality.flag_tracking's default (placeholder)
FROZEN_TOL_PX = 1e-6               # quality.flag_tracking's default
IDENTITY_VERSION = "replay-run/2"
ANSWER_RECORD = "replay_answer_position"
LOCK_TIMEOUT_S = 5.0               # bounded wait for a run's lock (answers, refresh, building)
_END_ONLY = ("session_end", "stream_end")      # closures known only after the session's last row
_TOL = 1e-9


class ReplayError(ValueError):
    """The replay cannot be built or the request is not allowed (message says why)."""


# ---- identity ----

def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rule_semantics(r):
    """What the monitor executes for a rule (description and status do not change events)."""
    return {"rule_id": r.rule_id, "behavior": r.behavior, "min_duration_s": float(r.min_duration_s).hex(),
            "ask_threshold": float(r.ask_threshold).hex(), "alert_threshold": float(r.alert_threshold).hex(),
            "alert_mode": r.alert_mode}


def _cell(v):
    """A typed, exact, JSON-safe encoding of one score-table cell."""
    if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
        return ["null"]
    if isinstance(v, (bool, np.bool_)):
        return ["bool", bool(v)]
    if isinstance(v, (int, np.integer)):
        return ["int", int(v)]
    if isinstance(v, (float, np.floating)):
        return ["float", float(v).hex()]
    if isinstance(v, str):
        return ["str", v]
    return ["other", type(v).__name__, repr(v)]


def _rows_digest(records, columns):
    """SHA-256 of the rows exactly as the monitor is fed them: column names,
    then each row's typed cells in feed order."""
    h = hashlib.sha256(json.dumps(["columns", list(columns)]).encode("utf-8"))
    for rec in records:
        h.update(b"\n" + json.dumps([_cell(rec[c]) for c in columns], separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _approved(rules):
    rules = rules.active() if isinstance(rules, RuleSet) else list(rules)
    if not rules:
        raise ReplayError("no approved rule: confirm a rule on the Rules page first")
    try:
        checked = [normalize_rule(r) for r in rules]
    except (TypeError, ValueError) as exc:
        raise ReplayError(str(exc)) from None
    if len({r.rule_id for r in checked}) != len(checked):
        raise ReplayError("duplicate rule_id")
    return tuple(sorted(checked, key=lambda r: r.rule_id))     # one canonical execution order


def replay_run_id(session, rules, rows_digest, fps=DEFAULT_FPS):
    """Deterministic monitor run_id (letters, digits, _; <= 64) for one session,
    one approved rule set (its execution semantics, in rule_id order), one
    frame rate and one canonical score-row digest."""
    identity = {"version": IDENTITY_VERSION, "session": str(session), "fps": float(fps).hex(),
                "rules": [_rule_semantics(r) for r in sorted(rules, key=lambda r: r.rule_id)],
                "rows_sha256": rows_digest}
    digest = hashlib.sha256(el.canonical_bytes(identity)).hexdigest()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(session)).strip("_")[-20:] or "session"
    return f"rp_{slug}_{_sha(str(session))[:8]}_{digest[:24]}"


# ---- causal keypoint quality ----

def causal_pose_flags(pose, *, min_score=SUGGESTED_MIN_SCORE, frozen_min_frames=FROZEN_MIN_FRAMES,
                      frozen_tol_px=FROZEN_TOL_PX, frame_size=None, max_speed_px_s=None):
    """Per-frame quality flags that use frames up to and including each row
    only (the definitions of quality.flag_tracking, made causal; Ava's module is
    not changed). missing: no coordinates, or an interpolated (imputed) value,
    which did not exist on that frame; low_conf: score below min_score;
    frozen: this frame closes a run of >= frozen_min_frames identical
    positions; jump / out_of_frame only when max_speed_px_s / frame_size are
    given. Retrospective flag columns in `pose` are ignored."""
    cols = ["session", "mouse", "keypoint", "frame", "time_s", "x", "y"] + \
        [c for c in ("score", "imputed") if c in pose.columns]
    d = pose[cols].sort_values(["session", "mouse", "keypoint", "frame"], kind="stable").reset_index(drop=True)
    x, y = d["x"].to_numpy(float).copy(), d["y"].to_numpy(float).copy()
    if "imputed" in d.columns:
        imputed = d["imputed"].fillna(False).to_numpy(bool)
        x[imputed], y[imputed] = np.nan, np.nan
    missing = ~(np.isfinite(x) & np.isfinite(y))
    out = d[["session", "mouse", "keypoint", "frame", "time_s"]].copy()
    out["x"], out["y"] = np.where(missing, np.nan, x), np.where(missing, np.nan, y)
    out["score"] = d["score"].to_numpy(float) if "score" in d.columns else np.nan
    out["missing"] = missing
    out["low_conf"] = (~missing & (out["score"].to_numpy(float) < min_score)) if min_score is not None else False
    out["out_of_frame"] = False
    if frame_size is not None:
        w, h = frame_size
        out["out_of_frame"] = ~missing & ((x < 0) | (x > w) | (y < 0) | (y > h))
    frozen = np.zeros(len(d), dtype=bool)
    jump = np.zeros(len(d), dtype=bool)
    key = (d["session"].astype(str) + "\x00" + d["mouse"].astype(str) + "\x00" + d["keypoint"].astype(str)).to_numpy()
    t = d["time_s"].to_numpy(float)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(d)]):
        step = np.hypot(np.diff(x[a:b]), np.diff(y[a:b]))           # NaN across missing frames
        still = np.r_[False, np.nan_to_num(step, nan=np.inf) <= frozen_tol_px]
        run = 0
        for i, s in enumerate(still):                                # identical frames seen so far, causally
            run = run + 1 if s else 0
            frozen[a + i] = run + 1 >= frozen_min_frames
        if max_speed_px_s is not None:
            dt = np.diff(t[a:b])
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = np.where(dt > 0, step / dt, np.nan)
            jump[a:b] = np.r_[False, np.nan_to_num(speed, nan=0.0) > max_speed_px_s]
    out["frozen"], out["jump"] = frozen, jump
    return out


# ---- tracking warnings (typed adapter over Ava's tracking_warning_events) ----

@dataclass(frozen=True)
class TrackingWarning:
    warning_id: str                 # W-<session digest>-<flag>-<mouse>-<start frame>: unique across sessions
    source_event_id: str            # Ava's T-ID (restarts per call; never used as an identity)
    session: str
    flag: str                       # quality flag: missing, low_conf, frozen, out_of_frame, ...
    kind: str                       # WARNING_KINDS: missing_coordinates is not low_confidence
    coordinates_present: bool       # False only for missing coordinates
    mouse: str | None
    start_s: float
    end_s: float                    # Ava's end: the next good frame, or the last frame if still going
    ongoing_at_session_end: bool    # END-STATE fact: the stretch had not ended when the session ended
    min_duration_s: float
    message: str                    # Ava's text, unchanged
    knowable_after_s: float = 0.0   # frozen: time from start_s until enough identical frames were seen

    def to_log_entry(self, end_known=True, at_end=True):
        """A §4-shaped entry (Ava's tracking_warning shape, with this adapter's
        identity and kind in evidence). Not written to the monitor journal.
        During playback (at_end False) an unfinished warning is only ongoing:
        its end, duration and session-end outcome are unknown (None)."""
        outcome = self.ongoing_at_session_end if at_end else (False if end_known else None)
        return {"event_id": self.warning_id, "session": self.session, "rule_id": None, "behavior": None,
                "start_s": self.start_s, "end_s": self.end_s if end_known else None, "confidence": None,
                "action": "tracking_warning", "researcher_answer": None,
                "evidence": {"mouse": self.mouse, "flag": self.flag, "kind": self.kind,
                             "coordinates_present": self.coordinates_present,
                             "duration_s": self.end_s - self.start_s if end_known else None,
                             "end_known": end_known, "ongoing_at_session_end": outcome,
                             "min_duration_s": self.min_duration_s, "source_event_id": self.source_event_id,
                             "message": self.message}}


def tracking_warnings(flags_df, session, *, flags=WARNING_FLAGS, keypoints=TORSO_KEYPOINTS, min_duration_s=0.5,
                      fps=DEFAULT_FPS, frozen_min_frames=FROZEN_MIN_FRAMES):
    """Typed warnings of one session from a quality.flag_tracking table
    (Ava's tracking_warning_events, unchanged, on that session's rows only).
    frozen_min_frames must be the value the flags were computed with."""
    d = flags_df[flags_df["session"] == session]
    if d.empty:
        return ()
    flags = tuple(f for f in flags if f in d.columns)
    events = tracking_warning_events(d, flags=flags, keypoints=keypoints, min_duration_s=min_duration_s)
    k = d[d["keypoint"].isin(keypoints)] if keypoints is not None and "keypoint" in d.columns else d
    last_t = float(d["time_s"].max())
    token = _sha(str(session))[:10]
    out = []
    for ev in events:
        flag, mouse = ev["evidence"]["flag"], ev["evidence"]["mouse"]
        rows = k[k["mouse"].astype(str) == mouse] if mouse is not None and "mouse" in k.columns else k
        at_end = rows[np.isclose(rows["time_s"].to_numpy(float), last_t, rtol=0, atol=_TOL)]
        ongoing = bool(abs(ev["end_s"] - last_t) <= _TOL and at_end[flag].any())
        start_frame = int(round(ev["start_s"] * fps))
        safe_mouse = re.sub(r"[^A-Za-z0-9]+", "_", str(mouse))
        out.append(TrackingWarning(
            f"W-{token}-{flag}-{safe_mouse}-{start_frame}", ev["event_id"], session, flag,
            WARNING_KINDS.get(flag, flag), flag != "missing", mouse, float(ev["start_s"]), float(ev["end_s"]),
            ongoing, float(min_duration_s), ev["evidence"]["message"],
            (frozen_min_frames - 1) / fps if flag == "frozen" else 0.0))
    if len({w.warning_id for w in out}) != len(out):
        raise ReplayError(f"{session}: two tracking warnings share an identity")
    return tuple(out)


# ---- frame state ----

@dataclass(frozen=True)
class FrameState:
    """Everything the researcher may know at one replay position (causal)."""
    session: str
    run_id: str
    index: int                      # row position in the session (-1 before the first row, n at the end)
    frame: int | None               # None before the first row and in the end state
    time_s: float | None
    at_end: bool                    # after the last row (stream_end closures, session-end outcomes)
    synthetic: bool
    tracking_ok: bool | None
    tracking_note: str | None       # TRACKING_GAP_NOTE on a tracking gap
    scores: dict                    # behavior -> score at this frame (None if missing)
    events: tuple                   # §4 events visible now, in firing order
    new_event_ids: tuple            # events that fired on this frame
    open_event_ids: tuple           # visible events whose closure is not yet known
    warnings: tuple                 # §4-shaped warning entries visible now
    active_warning_ids: tuple       # visible warnings whose end is not yet known
    skeletons: tuple = ()           # skeleton.MouseSkeleton per mouse (empty without pose)


@dataclass
class _Timeline:
    fired: dict
    closed: dict | None
    fired_index: int
    closed_index: int | None        # row index where the closure is known; None = end state only


@dataclass
class ReplaySession:
    session: str
    run_id: str
    rules: tuple
    synthetic: bool
    frames: tuple
    times: tuple
    tracking_ok: tuple
    scores: dict                    # behavior -> tuple of scores
    journal: el.EventJournal = field(repr=False)
    _monitor: Monitor = field(repr=False)
    _timeline: list = field(repr=False)
    _warnings: list = field(repr=False)       # (TrackingWarning, visible index, end index or None)
    _pose: dict = field(repr=False, default_factory=dict)   # frame -> {mouse: {keypoint: row}}
    _min_keypoint_score: float = SUGGESTED_MIN_SCORE
    _answers_path: Path | None = None
    _answer_index: dict = field(repr=False, default_factory=dict)   # event_id -> row index of the answer

    banner = REPLAY_BANNER

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def fps(self):
        """The frame rate the monitor ran at."""
        return self._monitor.fps

    @property
    def source_note(self):
        return SYNTHETIC_NOTE if self.synthetic else "recorded classifier scores"

    def index_of(self, frame):
        """Row index of the last row with frame <= `frame` (-1 before the first row)."""
        return bisect.bisect_right(self.frames, frame) - 1

    def state_at(self, frame):
        """The state after the rows up to and including `frame` were played."""
        return self.state_at_index(self.index_of(frame))

    def end_state(self):
        return self.state_at_index(self.n_frames)

    def state_at_index(self, i):
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not -1 <= i <= self.n_frames:
            raise ReplayError(f"index must be an integer in [-1, {self.n_frames}]")
        i = int(i)
        at_end = i == self.n_frames
        row = None if i < 0 or at_end else i
        events, new, open_ = [], [], []
        for t in self._timeline:
            if not (at_end or t.fired_index <= i):
                continue
            eid = t.fired["event_id"]
            closed_known = t.closed is not None and (at_end or (t.closed_index is not None and t.closed_index <= i))
            answered = self.journal.get("event_answered", eid)
            if answered is not None and not at_end:
                given = self._answer_index.get(eid)            # None: timing unknown -> end state only
                if given is None or given > i:
                    answered = None
            events.append(el.project_event(t.fired, t.closed if closed_known else None, answered))
            if not at_end and t.fired_index == i:
                new.append(eid)
            if not closed_known:
                open_.append(eid)
        warnings, active = [], []
        for w, visible, end in self._warnings:
            if not (at_end or (visible is not None and visible <= i)):
                continue
            known = at_end or (end is not None and end <= i)
            warnings.append(w.to_log_entry(end_known=known, at_end=at_end))
            if not known:
                active.append(w.warning_id)
        ok = None if row is None else self.tracking_ok[row]
        skeletons = ()
        if row is not None and self.frames[row] in self._pose:
            skeletons = tuple(mouse_skeleton(m, kps, min_score=self._min_keypoint_score, frame_tracking_ok=ok)
                              for m, kps in sorted(self._pose[self.frames[row]].items()))
        return FrameState(
            session=self.session, run_id=self.run_id, index=i, frame=None if row is None else self.frames[row],
            time_s=None if row is None else self.times[row], at_end=at_end, synthetic=self.synthetic,
            tracking_ok=ok, tracking_note=TRACKING_GAP_NOTE if ok is False else None,
            scores={} if row is None else {b: v[row] for b, v in self.scores.items()},
            events=tuple(copy.deepcopy(events)), new_event_ids=tuple(new), open_event_ids=tuple(open_),
            warnings=tuple(warnings), active_warning_ids=tuple(active), skeletons=skeletons)

    def advance(self, index, wall_elapsed_s, speed=1.0):
        """Playback at recorded speed: the row index reached after
        `wall_elapsed_s` seconds of wall time from row `index` at `speed`
        (time_s based; never past the last row; a pure function)."""
        if speed not in PLAYBACK_SPEEDS:
            raise ReplayError(f"speed must be one of {PLAYBACK_SPEEDS}")
        if not (isinstance(wall_elapsed_s, (int, float)) and math.isfinite(wall_elapsed_s) and wall_elapsed_s >= 0):
            raise ReplayError("wall_elapsed_s must be a finite number >= 0")
        if self.n_frames == 0:
            return -1
        if index < 0 and wall_elapsed_s == 0:
            return -1                     # nothing played yet
        start = min(max(int(index), 0), self.n_frames - 1)
        target = self.times[start] + wall_elapsed_s * speed
        return max(start, bisect.bisect_right(self.times, target + _TOL) - 1)

    def answer_status(self, event_id):
        """The journal's current answer of an event of this run and where it was
        given: {"researcher_answer", "answered_frame" (None = timing unknown)}."""
        if not any(t.fired["event_id"] == event_id for t in self._timeline):
            raise ReplayError(f"{event_id!r} is not an event of this run")
        answered = self.journal.get("event_answered", event_id)
        index = self._answer_index.get(event_id)
        return {"researcher_answer": answered["researcher_answer"] if answered else None,
                "answered_frame": None if index is None or answered is None else self.frames[index]}

    def refresh(self):
        """Re-read the journal and answer positions from disk (another browser
        tab or process may have answered), under the run lock. In-memory
        sessions have nothing to re-read."""
        if self.journal.path is None:
            return
        with run_lock(self.journal.path):
            self._refresh_locked()

    def _refresh_locked(self):
        try:
            self.journal.recover()
        except el.JournalIntegrityError as exc:
            raise ReplayError(f"the journal {self.journal.path.name} is inconsistent: {exc}") from None
        except el.EventPersistenceError as exc:
            raise ReplayError(f"the journal {self.journal.path.name} cannot be re-read: {exc}") from None
        for eid, index in _load_answer_positions(self._answers_path, self.run_id, self.frames, self._timeline,
                                                 self.journal).items():
            self._answer_index.setdefault(eid, index)

    def answer(self, event_id, answer, *, at_frame):
        """The researcher's final answer to an "ask" event of this run that is
        visible at replay frame `at_frame`. Written once to the journal (the
        same answer again is a no-op that keeps the first position; a
        different one is refused); the position is recorded by this layer."""
        state = self.state_at(at_frame)
        event = next((e for e in state.events if e["event_id"] == event_id), None)
        if event is None:
            raise ReplayError(f"{event_id!r} is not an event of this run visible at frame {at_frame}")
        if event["action"] != "ask":
            raise ReplayError(f"{event_id!r} is a {event['action']!r} event; only 'ask' events take an answer")
        if answer not in el.ANSWERS:
            raise ReplayError(f"answer must be one of {el.ANSWERS}")
        # one atomic step under the run lock: re-read, check, append the answer, record its position
        with run_lock(self.journal.path):
            if self.journal.path is not None:
                self._refresh_locked()
            current = self.journal.get("event_answered", event_id)
            if current is not None:       # decided before anything is written
                if current["researcher_answer"] != answer:
                    raise ReplayError(f"{event_id!r} already has a different answer; answers are final")
                return False              # the same answer again: a no-op
            try:
                written = self._monitor.answer(event_id, answer)
            except el.JournalConflictError:
                raise ReplayError(f"{event_id!r} already has a different answer; answers are final") from None
            except el.EventPersistenceError as exc:
                raise ReplayError(f"the answer could not be saved ({exc}); nothing was recorded") from None
            if written:                   # a new answer: its replay position is known exactly
                self._answer_index[event_id] = state.index
                if self._answers_path is not None:
                    try:
                        _append_line(self._answers_path, {"record": ANSWER_RECORD, "run_id": self.run_id,
                                                          "event_id": event_id, "researcher_answer": answer,
                                                          "answered_frame": state.frame})
                    except OSError as exc:
                        raise ReplayError(f"the answer was saved, but its replay position could not be recorded "
                                          f"({exc.strerror or exc}); after a restart it shows as timing-unknown") \
                            from None
            return written

    def log_entries(self):
        """End-of-session §4 entries of this run (events, then warnings), for the summary."""
        end = self.end_state()
        return end.events + end.warnings


# ---- the per-run lock ----

def lock_path(journal_path):
    return Path(f"{journal_path}.lock")


# ---- one non-blocking exclusive lock attempt, per platform ----
# POSIX: fcntl.flock(LOCK_EX | LOCK_NB), BlockingIOError when another holder has it.
# Windows (no fcntl): msvcrt.locking(LK_NBLCK) on byte 0; contention is reported as one of
# investigation_data.WINDOWS_LOCK_CONTENTION (the errors the Stage 9 cache lock already treats as contention).
# Any other error propagates. The msvcrt module is a parameter so the Windows path is testable anywhere.

def _windows_try_lock(fd, msvcrt_module):
    from .investigation_data import WINDOWS_LOCK_CONTENTION
    os.lseek(fd, 0, os.SEEK_SET)                    # locking() works from the current position
    try:
        msvcrt_module.locking(fd, msvcrt_module.LK_NBLCK, 1)
        return True
    except OSError as exc:
        if exc.errno in WINDOWS_LOCK_CONTENTION:
            return False
        raise


def _windows_unlock(fd, msvcrt_module):
    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt_module.locking(fd, msvcrt_module.LK_UNLCK, 1)


if os.name == "nt":  # pragma: no cover - Windows
    import msvcrt

    def _try_lock(fd):
        return _windows_try_lock(fd, msvcrt)

    def _unlock(fd):
        _windows_unlock(fd, msvcrt)
else:
    import fcntl

    def _try_lock(fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _unlock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def run_lock(journal_path, timeout_s=None):
    """An exclusive advisory lock (fcntl.flock on POSIX, msvcrt.locking on Windows) on <journal>.lock, shared by
    every replay instance and process using that journal: building (the
    monitor's appends), refreshing and answering all hold it, so no reader
    repairs a line another writer is still writing and no two answers
    interleave. Waits at most timeout_s (LOCK_TIMEOUT_S), then ReplayError.
    Released on every exit path (the descriptor is closed). None: no lock (an
    in-memory journal has no other writer)."""
    if journal_path is None:
        yield
        return
    timeout_s = LOCK_TIMEOUT_S if timeout_s is None else timeout_s
    path = lock_path(journal_path)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ReplayError(f"the run lock {path.name} cannot be opened ({exc.strerror})") from None
    try:
        deadline = time.monotonic() + timeout_s
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise ReplayError(f"the replay journal is busy (another answer or build holds {path.name} for "
                                  f"more than {timeout_s:g} s); nothing was written, try again") from None
            time.sleep(0.01)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


# ---- answer positions (next to the journal; the journal schema is unchanged) ----

def _append_line(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_answer_positions(path, run_id, frames, timeline, journal):
    """{event_id: row index} from an answers file, checked against this run and
    its journal. A line whose answer the journal does not hold (e.g. a crash
    before the journal write) is ignored; a contradiction is refused."""
    if path is None or not Path(path).exists():
        return {}
    fired_at = {t.fired["event_id"]: t.fired_index for t in timeline}
    out = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            raise ReplayError(f"{Path(path).name} line {n}: not JSON") from None
        if not isinstance(rec, dict) or set(rec) != {"record", "run_id", "event_id", "researcher_answer",
                                                       "answered_frame"} or rec["record"] != ANSWER_RECORD:
            raise ReplayError(f"{Path(path).name} line {n}: not an answer-position record")
        if rec["run_id"] != run_id:
            continue
        eid, frame = rec["event_id"], rec["answered_frame"]
        if eid not in fired_at or type(frame) is not int or frame not in frames:
            raise ReplayError(f"{Path(path).name} line {n}: unknown event or frame for this run")
        index = frames.index(frame)
        if index < fired_at[eid]:
            raise ReplayError(f"{Path(path).name} line {n}: answer position before the event fired")
        answered = journal.get("event_answered", eid)
        if answered is None:
            continue
        if answered["researcher_answer"] != rec["researcher_answer"]:
            raise ReplayError(f"{Path(path).name} line {n}: contradicts the journal's answer")
        out.setdefault(eid, index)                 # the first recorded position
    return out


# ---- building ----

def _session_records(scores, session, behaviors):
    rows = scores[scores["session"] == session]
    if rows.empty:
        raise ReplayError(f"no score rows for session {session!r}")
    missing = [c for c in ("frame", "time_s", "tracking_ok", *behaviors) if c not in rows.columns]
    if missing:
        raise ReplayError(f"{session}: score rows lack columns {missing}")
    rows = rows.sort_values("frame", kind="stable").reset_index(drop=True)
    if rows["frame"].duplicated().any():
        raise ReplayError(f"{session}: duplicate frame numbers in the score rows")
    columns = list(rows.columns)
    return rows.to_dict("records"), columns


def _pose_index(pose, session, **quality):
    if pose is None:
        return {}
    p = causal_pose_flags(pose[pose["session"] == session], **quality)
    out = {}
    for rec in p.to_dict("records"):
        out.setdefault(int(rec["frame"]), {}).setdefault(str(rec["mouse"]), {})[str(rec["keypoint"])] = rec
    return out


def _run_id_for(scores, session, rules, fps):
    records, columns = _session_records(scores, session, sorted({r.behavior for r in rules}))
    return records, replay_run_id(session, rules, _rows_digest(records, columns), fps)


def run_paths(scores, session, rules, directory, fps=DEFAULT_FPS):
    """(run_id, journal path, answers path) of one session run kept in `directory`."""
    _, run_id = _run_id_for(scores, session, _approved(rules), fps)
    directory = Path(directory)
    return run_id, directory / f"{run_id}.journal.jsonl", directory / f"{run_id}.answers.jsonl"


def build_replay(scores, session, rules, *, warnings=(), pose=None, journal_path=None, answers_path=None,
                 fps=DEFAULT_FPS, min_keypoint_score=SUGGESTED_MIN_SCORE, frozen_min_frames=FROZEN_MIN_FRAMES,
                 frame_size=None, max_speed_px_s=None):
    """A ReplaySession for one session: the approved rules (in rule_id order)
    run once through the monitor over that session's rows. journal_path: one
    journal file per session run (None keeps it in memory); answers_path: where
    this layer records answer positions (None: in memory only, so a rebuilt
    session sees journal answers as timing-unknown)."""
    rules = _approved(rules)
    records, run_id = _run_id_for(scores, session, rules, fps)
    synthetic = str(session).startswith(SYNTHETIC_PREFIX)
    if any(w.session != session for w in warnings):
        raise ReplayError("tracking warnings of another session were given")
    try:
        with run_lock(journal_path):      # the monitor appends to (or confirms) the shared journal
            monitor = Monitor(list(rules), fps=fps, journal_path=journal_path, run_id=run_id)
            for row in records:
                monitor.step(row)
            events = monitor.finish()
    except (el.JournalIntegrityError, el.EventPersistenceError, ValueError) as exc:
        raise ReplayError(f"{session}: the monitor could not replay this session ({type(exc).__name__}: {exc})") \
            from None
    frames = tuple(int(r["frame"]) for r in records)
    times = tuple(float(r["time_s"]) for r in records)
    try:                                   # the monitor's own parsing (already accepted every row above)
        tracking = tuple(_parse_tracking(r["tracking_ok"]) for r in records)
        behaviors = sorted({r.behavior for r in rules})
        scores_by = {b: tuple(_parse_score(b, r[b]) for r in records) for b in behaviors}
    except MonitorInputError as exc:
        raise ReplayError(f"{session}: {exc}") from None
    timeline = []
    for e in events:                      # this monitor instance's own events: only this run
        fired = monitor.journal.get("event_fired", e["event_id"])
        closed = monitor.journal.get("event_closed", e["event_id"])
        fired_index = frames.index(fired["fired_frame"])
        closed_index = None
        if closed is not None and closed["end_reason"] not in _END_ONLY:
            closed_index = bisect.bisect_right(frames, closed["last_frame"])   # the first row after the bout
            if closed_index >= len(frames):
                closed_index = None
        timeline.append(_Timeline(fired, closed, fired_index, closed_index))
    timeline.sort(key=lambda t: (t.fired_index, t.fired["event_id"]))
    warned = []
    for w in sorted(warnings, key=lambda w: (w.start_s, w.warning_id)):
        visible = bisect.bisect_left(times, w.start_s + max(w.min_duration_s, w.knowable_after_s) - _TOL)
        end = None if w.ongoing_at_session_end else bisect.bisect_left(times, w.end_s - _TOL)
        warned.append((w, visible if visible < len(times) else None,
                       end if end is not None and end < len(times) else None))
    with run_lock(journal_path):          # positions are written under the same lock as answers
        answers = _load_answer_positions(answers_path, run_id, frames, timeline, monitor.journal)
    pose_index = _pose_index(pose, session, min_score=min_keypoint_score, frozen_min_frames=frozen_min_frames,
                             frame_size=frame_size, max_speed_px_s=max_speed_px_s)
    return ReplaySession(session, run_id, rules, synthetic, frames, times, tracking, scores_by, monitor.journal,
                         monitor, timeline, warned, pose_index, min_keypoint_score,
                         None if answers_path is None else Path(answers_path), answers)


def build_replays(scores, rules, *, warnings=(), pose=None, journal_dir=None, fps=DEFAULT_FPS):
    """{session: ReplaySession} for every session in the score table, each its
    own monitor run (journal_dir: <run_id>.journal.jsonl and
    <run_id>.answers.jsonl per run)."""
    approved = _approved(rules)
    out = {}
    for session in sorted(scores["session"].unique()):
        journal = answers = None
        if journal_dir is not None:
            _, journal, answers = run_paths(scores, session, approved, journal_dir, fps)
        out[session] = build_replay(scores, session, rules, warnings=[w for w in warnings if w.session == session],
                                    pose=pose, journal_path=journal, answers_path=answers, fps=fps)
    return out
