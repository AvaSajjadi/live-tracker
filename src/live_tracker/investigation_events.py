"""EXPERIMENTAL (Investigation mode, Stage 3): the real `query_events` tool (new file).

Stage 4 adds get_window_features and check_tracking_quality
(investigation_windows.py), Stage 5 compare_windows (investigation_compare.py);
they read the cache through the same AnnotationEventSource.snapshot() and are
reached through the same CalMS21InvestigationTools.

Read-only. Returns behavioral episodes as Stage 2 contract results
(investigation_contracts, CONTRACT_VERSION), from event sources plugged in by
name:

    validation_annotation   AnnotationEventSource: CalMS21 validation-session
                            annotations, read only through the Stage 1 verified
                            cache (investigation_data.verified_sessions) and
                            Igor's calms21.load_calms21. No other file is opened.
    classifier_prediction   not available yet -> explicit "source_unavailable"
    monitor_event           not available yet -> explicit "source_unavailable"

A request for a source that is not plugged in is refused with an error result;
annotations are never substituted for predictions or monitor events. Adding a
source later means implementing EventSource and passing it to
CalMS21InvestigationTools; the agent-facing interface does not change.

Authorization and data access share ONE manifest snapshot
---------------------------------------------------------
AnnotationEventSource reads the split manifest once, at construction, and
keeps its bytes. The EvidencePolicy used for authorization must describe
exactly that snapshot (checked at construction and on every query), and the
tool refuses a source bound to a different policy object. On every query:
1. the live manifest file must still hash to the snapshot (else refused),
2. the policy must still equal the snapshot,
3. Stage 1's verifier is given a private, frozen copy of the snapshot, so the
   cache index must be bound to that same manifest, and every session the
   index lists is checked against it BEFORE any session file is opened.
Checks 1 and 2 are repeated just before the result is returned (finalize), so
a manifest or policy that changed while the query computed refuses it.

Session coverage is explicit
----------------------------
The sessions a query may search are the manifest's validation split (the
authorized annotation split), not whatever the cache happens to hold. With
coverage="complete" (the default) every tool refuses unless the verified
cache holds every eligible validation session. coverage="partial" must be
chosen explicitly; its results then say, in methodology (eligible_sessions,
available_sessions, session_coverage) and limitations, which eligible
sessions were unavailable. A partial cache is never reported as complete.

One immutable metadata snapshot per query
-----------------------------------------
All provenance (index digest, source path / sha256, per-session file sha256)
comes from the single index that the verifier checked. Before the result is
returned, the index file is re-read under the cache lock and compared in full
(not only the session entries); any difference refuses the query.

Every backend episode is checked before totals or truncation
-------------------------------------------------------------
All episodes (not only the returned page) go through the Stage 2 contract
validator (duration minima, timestamps, source / behavior, policy, provenance,
stable IDs, uniqueness, session membership) before n_episodes_total is
counted or the list is truncated. A malformed episode anywhere refuses the
whole query.

Episode definition (provisional; listed in every result's methodology)
------------------------------------------------------------------------
- a maximal run of consecutive frames annotated with the requested behavior;
- if merge_gap_frames is given, runs separated by at most that many frames are
  merged into ONE SPAN [first start, last end). A merged span contains the gap
  frames, which are NOT annotated as the behavior: it is not continuous
  behavior;
- every episode carries annotated_frames: the number of frames in its span
  annotated with the behavior, i.e. the sum of the lengths of the original
  annotation runs it contains (each frame counted once, gaps excluded). For an
  unmerged episode it equals end_frame - start_frame;
- min_duration_frames is measured on the span (after merging), so merged gaps
  count toward it;
- episodes are ordered by (session, start_frame); when more than max_episodes
  match, the first max_episodes in that order are returned and the result is
  marked truncated (this favors sessions that sort first).
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import investigation_contracts as ic
from .calms21 import FPS, load_calms21
from .investigation_data import (INDEX_NAME, CacheError, ExtractionRefused, cache_lock, load_manifest,
                                 verified_sessions)

__all__ = ["AnnotationEventSource", "CalMS21InvestigationTools", "EventQuery", "EventSource",
           "SourceUnavailable", "behavior_runs", "merge_runs"]

IMPLEMENTATION = "live-tracker-calms21-events"
IMPLEMENTATION_VERSION = "0.5-experimental"


class SourceUnavailable(RuntimeError):
    """The event source cannot answer (cache missing / altered, manifest
    changed, session not available). Reported as a 'source_unavailable' error."""


# ---- episode construction (pure, no I/O) ----

def behavior_runs(is_behavior):
    """[(start, end_exclusive)] of maximal True runs."""
    a = np.asarray(is_behavior, dtype=bool)
    if a.size == 0:
        return []
    edges = np.diff(np.r_[0, a.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def merge_runs(runs, merge_gap_frames):
    """Merge runs separated by at most merge_gap_frames frames (None: no merging).
    Returns [(start, end, annotated_frames)]: the span and how many of its
    frames belong to the original runs (the rest are merged gap frames)."""
    if not runs:
        return []
    out = [[runs[0][0], runs[0][1], runs[0][1] - runs[0][0]]]
    for start, end in runs[1:]:
        if merge_gap_frames is not None and start - out[-1][1] <= merge_gap_frames:
            out[-1][1] = end
            out[-1][2] += end - start
        else:
            out.append([start, end, end - start])
    return [tuple(r) for r in out]


# ---- the pluggable source interface ----

@dataclass
class EventQuery:
    """What a source returns for one validated request."""
    episodes: list                       # contract episode items, any order
    sessions_searched: list
    data: list = field(default_factory=list)          # provenance data items
    methodology: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    finalize: object = None              # optional callable run just before the result is returned


class EventSource(Protocol):
    source: str                          # one of investigation_contracts.SOURCES
    policy: ic.EvidencePolicy            # the policy this source is bound to

    def query(self, request: dict) -> EventQuery: ...


def _method(name, value, note=None):
    return {"name": name, "value": value, "status": "provisional", "decided_by": None, "note": note}


ANNOTATION_LIMITATIONS = [
    "Episodes come from human annotations (annotator 1) of CalMS21 Task 1 validation sessions; "
    "they are labels, not measurements, and annotation boundaries are themselves uncertain.",
    "Episodes within one session are not independent observations.",
    "Frames just before an annotated attack are mostly annotated 'investigation' "
    "(investigation confound, not resolved).",
    "CalMS21 Task 1 has no 'approach' label; approach can only be measured from motion features.",
    "Held-out test annotations are never read; training-split annotations are not used.",
]


COVERAGE_MODES = ("complete", "partial")


def behavior_prevalence(summary, behavior):
    """The split-wide count of sessions with the behavior, from the verified
    index's split label summary (computed from the source at extraction)."""
    if not summary or any(behavior not in v["label_runs"] for v in summary.values()):
        return None
    with_behavior = sum(v["label_runs"][behavior] > 0 for v in summary.values())
    runs = sum(v["label_runs"][behavior] for v in summary.values())
    return (f"In the manifest's validation split, {with_behavior} of {len(summary)} sessions contain any "
            f"annotated {behavior} ({runs} runs; split label summary computed from the source at extraction); "
            f"with sessions as the unit of analysis, at most {with_behavior} sessions can contribute.")


def coverage_metadata(coverage):
    """(methodology items, limitations) recording a snapshot's session coverage;
    every tool that reads the cache adds them, so partial coverage is disclosed
    downstream of query_events too."""
    if not coverage:
        return [], []
    methodology = [
        _method("eligible_sessions", coverage["eligible"],
                "validation sessions in the split manifest (the authorized annotation split)"),
        _method("available_sessions", coverage["available"], "eligible sessions held by the verified cache"),
        _method("session_coverage", "complete" if coverage["complete"] else
                f"partial: {len(coverage['missing'])} eligible sessions unavailable",
                ("unavailable (not searched): " + ", ".join(coverage["missing"]))[:ic.MAX_TEXT]
                if coverage["missing"] else "every eligible validation session is available"),
    ]
    limitations = [] if coverage["complete"] else [
        f"PARTIAL COVERAGE: only {coverage['available']} of {coverage['eligible']} eligible validation sessions "
        "are in the verified cache; the unavailable sessions were not read and nothing is said about them."]
    return methodology, limitations


def _policy_mapping(manifest):
    return {sid: split for split in ("train", "validation", "test") for sid in manifest[split]}


def _canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class VerifiedSnapshot:
    """Verified sessions and provenance from one immutable cache-index snapshot."""
    index: dict
    sessions: dict            # session id -> parsed {group: {session: record}}
    data: list                # provenance data items
    synthetic: bool
    finalize: object          # callable: refuses if the binding or the index changed since the snapshot
    coverage: dict = field(default_factory=dict)   # eligible / available / missing validation sessions


def session_labels(sid, data):
    """Per-frame label names of one verified session (Igor's loader), refused
    unless frames are 0..n-1 and times are frame / FPS."""
    _, labels = load_calms21(data)
    labels = labels.sort_values("frame").reset_index(drop=True)
    frames = labels["frame"].to_numpy()
    if not np.array_equal(frames, np.arange(len(frames))) or \
            not np.allclose(labels["time_s"].to_numpy(), frames / FPS, rtol=0, atol=1e-9):
        raise SourceUnavailable(f"{sid}: frames / times are not 0..n-1 at {FPS} fps")
    return labels["label"].to_numpy()


class AnnotationEventSource:
    """CalMS21 validation annotations from the Stage 1 verified cache, bound to
    one manifest snapshot and to the EvidencePolicy that describes it."""

    source = "validation_annotation"

    def __init__(self, cache_dir, manifest_path, policy, coverage="complete"):
        ic._require_policy(policy)
        if policy.annotation_split != "validation":
            raise ic.ForbiddenEvidence("policy", "only validation annotations are read")
        if coverage not in COVERAGE_MODES:
            raise ValueError(f"coverage must be one of {COVERAGE_MODES}")
        self.coverage = coverage
        self.cache_dir = Path(cache_dir)
        self.manifest_path = Path(manifest_path)
        self._manifest_bytes = self.manifest_path.read_bytes()        # the snapshot
        self._manifest_sha = hashlib.sha256(self._manifest_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:                     # validate the snapshot itself
            frozen = Path(tmp) / "manifest.json"
            frozen.write_bytes(self._manifest_bytes)
            manifest, sha = load_manifest(frozen)
        assert sha == self._manifest_sha
        self._mapping = _policy_mapping(manifest)
        self._eligible = sorted(manifest["validation"])     # the authorized sessions, from the manifest
        if policy.split_of != self._mapping:
            raise ic.ForbiddenEvidence("policy", "the policy does not match this source's manifest")
        self.policy = policy

    def _check_binding(self):
        """Before reading anything: live manifest == snapshot == policy."""
        try:
            live = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise SourceUnavailable(f"split manifest unreadable ({type(exc).__name__})") from None
        if live != self._manifest_sha:
            raise SourceUnavailable("the split manifest changed after this tool was created; "
                                    "create a new tool from the current manifest")
        if self.policy.split_of != self._mapping or self.policy.annotation_split != "validation":
            raise SourceUnavailable("the evidence policy no longer matches the manifest snapshot")

    def snapshot(self, sessions):
        """Verified data for `sessions` (None: every verified session) from ONE
        immutable index snapshot: binding checks, Stage 1 verification against the
        frozen manifest snapshot, provenance items, and a finalize() that refuses
        if the manifest / policy binding or the index changed before the result
        is returned. Shared by every tool
        that reads the cache."""
        self._check_binding()
        with tempfile.TemporaryDirectory() as tmp:  # Stage 1 verifies against the frozen snapshot
            frozen = Path(tmp) / "manifest.json"
            frozen.write_bytes(self._manifest_bytes)
            try:
                verified, index = verified_sessions(self.cache_dir, frozen)
            except (CacheError, ExtractionRefused, OSError, ValueError) as exc:
                raise SourceUnavailable(f"validated CalMS21 cache unavailable ({type(exc).__name__})") from None
        index = copy.deepcopy(index)                 # the one immutable snapshot for this query
        available = {sid: data for sid, data in verified}
        missing_eligible = sorted(set(self._eligible) - set(available))
        if missing_eligible and self.coverage == "complete":
            raise SourceUnavailable(
                f"the validated cache holds {len(self._eligible) - len(missing_eligible)} of "
                f"{len(self._eligible)} eligible validation sessions (missing {missing_eligible[:3]}"
                f"{' ...' if len(missing_eligible) > 3 else ''}); extract them with "
                "`python -m live_tracker.investigation_data --all`, or choose coverage='partial' explicitly")
        coverage = {"mode": self.coverage, "eligible": len(self._eligible),
                    "available": len(self._eligible) - len(missing_eligible), "missing": missing_eligible,
                    "complete": not missing_eligible}
        wanted = sorted(available) if sessions is None else sorted(sessions)
        missing = [sid for sid in wanted if sid not in available]
        if missing:
            raise SourceUnavailable(f"{len(missing)} requested session(s) are not in the validated "
                                    f"cache: {missing[:3]}")
        synthetic = any("SYNTHETIC" in sid for sid in available)
        kind = "synthetic_fixture" if synthetic else "calms21_validation_cache"
        label = "SYNTHETIC test fixture, not CalMS21 data" if synthetic else \
            "verified CalMS21 Task 1 validation-session cache (Stage 1)"
        data = [{"kind": kind, "path": str(self.cache_dir / INDEX_NAME), "sha256": _canonical_sha(index),
                 "description": f"{label}: extraction index (sha256 of its canonical JSON, as verified)"},
                {"kind": kind, "path": index["source"]["path"], "sha256": index["source"]["sha256"],
                 "description": f"{label}: source file the cache was extracted from (read once, read-only)"}]
        for sid in wanted:
            data.append({"kind": kind, "path": str(self.cache_dir / index["sessions"][sid]["file"]),
                         "sha256": index["sessions"][sid]["sha256"], "description": f"{label}: {sid}"})

        def finalize():
            """Just before the result is returned: the manifest / policy binding
            still holds, and the whole index, re-read under the lock, equals the
            snapshot."""
            self._check_binding()
            try:
                with cache_lock(self.cache_dir):
                    live = json.loads((self.cache_dir / INDEX_NAME).read_bytes())
            except (OSError, ValueError) as exc:
                raise SourceUnavailable(f"validated CalMS21 cache changed while reading "
                                        f"({type(exc).__name__})") from None
            if live != index:
                raise SourceUnavailable("validated CalMS21 cache changed while reading "
                                        "(extraction index differs from the verified snapshot)")

        return VerifiedSnapshot(index=index, sessions={sid: available[sid] for sid in wanted},
                                data=data[:20], synthetic=synthetic, finalize=finalize, coverage=coverage)

    def query(self, request):
        snap = self.snapshot(request["sessions"])
        index, wanted = snap.index, sorted(snap.sessions)
        episodes = []
        for sid in wanted:
            entry = index["sessions"][sid]
            labels = session_labels(sid, snap.sessions[sid])
            rec = next(iter(snap.sessions[sid].values()))[sid]
            annotator = f"annotator-id_{rec['metadata'].get('annotator-id', 'unknown')}"
            runs = merge_runs(behavior_runs(labels == request["behavior"]), request["merge_gap_frames"])
            for start, end, annotated in runs:
                if request["min_duration_frames"] is not None and end - start < request["min_duration_frames"]:
                    continue
                episodes.append({
                    "episode_id": ic.episode_id(self.source, request["behavior"], sid, start),
                    "session": sid, "source": self.source,
                    "behavior": request["behavior"], "start_frame": start, "end_frame": end,
                    "start_s": start / FPS, "end_s": end / FPS, "fps": FPS,
                    "annotated_frames": annotated,
                    "provenance": {"split": "validation", "annotator": annotator[:40],
                                   "file_sha256": entry["sha256"]}})

        span_note = ("measured on the episode span [start, end); with merging, merged gap frames "
                     "count toward it although they are not annotated as the behavior")
        methodology = [
            _method("event_source", "validation_annotation",
                    "human annotations of validation sessions; not classifier predictions"),
            _method("episode_definition", "maximal run of frames annotated with the behavior"),
            _method("merge_gap_frames", request["merge_gap_frames"],
                    "null: runs are never merged. Otherwise runs at most this many frames apart "
                    "become one span that includes the non-behavior gap frames; it is not "
                    "continuous behavior. annotated_frames counts only the annotated frames"),
            _method("min_duration_frames", request["min_duration_frames"],
                    "null: no minimum duration. Otherwise " + span_note),
            _method("ordering_and_truncation", "sorted by (session, start_frame); first max_episodes kept"),
            _method("fps", FPS, "calms21.FPS"),
            *coverage_metadata(snap.coverage)[0],
        ]
        limitations = list(ANNOTATION_LIMITATIONS)
        prevalence = behavior_prevalence(index.get("split_label_summary"), request["behavior"])
        if prevalence:
            limitations.insert(1, prevalence)
        limitations = coverage_metadata(snap.coverage)[1] + limitations
        if snap.synthetic:
            limitations.insert(0, "SYNTHETIC test fixture: not CalMS21 data, not evidence.")
        if request["merge_gap_frames"]:
            limitations.append("Merged episodes span unannotated gap frames: end_frame - start_frame "
                               "is elapsed span, annotated_frames is the observed behavior.")
        return EventQuery(episodes=episodes, sessions_searched=wanted, data=snap.data,
                          methodology=methodology, limitations=limitations, finalize=snap.finalize)


# ---- the InvestigationTools implementation ----

def _code_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class CalMS21InvestigationTools(ic.UnavailableTools):
    """InvestigationTools with real query_events (Stage 3), get_window_features
    and check_tracking_quality (Stage 4, investigation_windows) and
    compare_windows (Stage 5, investigation_compare)."""

    implementation = IMPLEMENTATION
    implementation_version = IMPLEMENTATION_VERSION

    def __init__(self, policy, sources):
        ic._require_policy(policy)
        if policy.annotation_split != "validation":
            raise ic.ForbiddenEvidence("policy", "this stage only reads validation annotations")
        self.policy = policy
        self.sources = {}
        for src in sources:
            if src.source not in ic.SOURCES:
                raise ic.ContractError("sources", f"unknown event source {src.source!r}")
            if src.source in self.sources:
                raise ic.ContractError("sources", f"two sources for {src.source!r}")
            if getattr(src, "policy", None) is not policy:
                raise ic.ForbiddenEvidence("sources", f"source {src.source!r} is bound to another "
                                           "evidence policy")
            self.sources[src.source] = src

    @classmethod
    def for_validation_cache(cls, cache_dir, manifest_path, policy, coverage="complete"):
        """The standard setup: annotations from the verified cache, bound to the
        same manifest snapshot as `policy`. coverage="complete" refuses a cache
        missing any eligible validation session; "partial" must be explicit."""
        return cls(policy, [AnnotationEventSource(cache_dir, manifest_path, policy, coverage=coverage)])

    def _provenance(self, data=()):
        return {"implementation": IMPLEMENTATION, "implementation_version": IMPLEMENTATION_VERSION,
                "contract_version": ic.CONTRACT_VERSION, "code_sha256": _code_sha256(),
                "data": list(data)}

    def _error(self, request, code, message):
        return ic.make_result("query_events", request, "error", None, provenance=self._provenance(),
                              errors=[{"code": code, "message": message[:ic.MAX_TEXT]}])

    def _check_all(self, request, q, episodes):
        """Validate EVERY backend episode with the Stage 2 contract validator,
        in pages of at most MAX_EPISODES, before totals or truncation."""
        if len({e.get("episode_id") if isinstance(e, dict) else None for e in episodes}) != len(episodes):
            raise ic.ContractError("episodes", "duplicate episode ids from the backend")
        searched = list(q.sessions_searched)
        for i in range(0, max(len(episodes), 1), ic.MAX_EPISODES):
            page = episodes[i:i + ic.MAX_EPISODES]
            page_req = dict(request, max_episodes=max(len(page), 1))
            with_eps = {e["session"] for e in page}
            m = {"episodes": page, "n_episodes_total": len(page), "truncated": False,
                 "sessions_searched": searched,
                 "sessions_without_episodes": [s for s in searched if s not in with_eps]}
            res = ic.make_result("query_events", page_req, "ok" if page else "no_data", m,
                                 methodology=q.methodology, limitations=q.limitations,
                                 provenance=self._provenance(q.data))
            ic.validate_result("query_events", page_req, res, self.policy)

    def get_window_features(self, request):
        from .investigation_windows import get_window_features
        return get_window_features(self, request)

    def check_tracking_quality(self, request):
        from .investigation_windows import check_tracking_quality
        return check_tracking_quality(self, request)

    def compare_windows(self, request):
        from .investigation_compare import compare_windows
        return compare_windows(self, request)

    def query_events(self, request):
        ic.validate_request("query_events", request, self.policy)  # nothing is read before this
        source = self.sources.get(request["source"])
        if source is None:
            return self._error(request, "source_unavailable",
                               f"event source {request['source']!r} is not available in this "
                               "implementation; no other source is substituted")
        try:
            q = source.query(copy.deepcopy(request))
        except SourceUnavailable as exc:
            return self._error(request, "source_unavailable", str(exc))
        episodes = copy.deepcopy(list(q.episodes))
        self._check_all(request, q, episodes)          # every episode, before anything is counted
        episodes.sort(key=lambda e: (e["session"], e["start_frame"]))
        kept = episodes[:request["max_episodes"]]
        with_eps = {e["session"] for e in episodes}
        measurements = {
            "episodes": kept,
            "n_episodes_total": len(episodes),
            "truncated": len(episodes) > len(kept),
            "sessions_searched": list(q.sessions_searched),
            "sessions_without_episodes": [s for s in q.sessions_searched if s not in with_eps],
        }
        result = ic.make_result("query_events", request, "ok" if episodes else "no_data", measurements,
                                methodology=q.methodology, limitations=q.limitations,
                                provenance=self._provenance(q.data))
        ic.validate_result("query_events", request, result, self.policy)  # self-check
        if q.finalize is not None:
            try:
                q.finalize()
            except SourceUnavailable as exc:
                return self._error(request, "source_unavailable", str(exc))
        return result
