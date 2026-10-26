"""Synthetic recorded-pose display regressions; no datasets or provider access."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from live_tracker import skeleton_player as sp, uploads, flow
from live_tracker.calms21 import KEYPOINTS
from live_tracker.skeleton import mouse_skeleton
from video_fixtures import fake_mp4


def rows(n=4):
    return pd.DataFrame([dict(session="SYNTHETIC", frame=f, mouse=m, keypoint=k,
                              x=100.+f+i, y=200.+i, score=.99)
                         for f in range(n) for m in "ab" for i,k in enumerate(KEYPOINTS)])


def frames(tmp_path, df):
    p=tmp_path/"poses.parquet"; df.to_parquet(p,index=False)
    return sp.pose_frames(p,session="SYNTHETIC")


def test_coordinates_precision_dimensions_and_two_identities(tmp_path):
    df=rows();df.loc[0,"x"]=123.456789
    out=frames(tmp_path,df)
    assert out[0]["a"][0]["x"] == 123.456789
    assert set(out[0]) == {"a","b"}
    assert all(len(m)==7 for f in out for m in f.values())
    assert sp.payload(out)["size"] == [1024,570]
    assert sp.payload(out)["keypoints"] == list(KEYPOINTS)


@pytest.mark.parametrize("change,status",[
    ({"x":np.nan},"missing"),({"x":np.inf},"missing"),({"y":-1.},"unreliable"),
    ({"x":1024.},"unreliable"),({"score":.1},"low_confidence"),
    ({"score":np.nan},"unreliable"),({"score":np.inf},"unreliable"),
    ({"imputed":True},"imputed"),({"tracking_ok":False},"unreliable"),
    ({"jump":True},"unreliable"),({"score_observed":False},"unreliable"),
    ({"score_observed":True},"tracked"),
])
def test_existing_quality_semantics_are_preserved(tmp_path,change,status):
    df=rows()
    for k,v in change.items():df.loc[0,k]=v
    assert frames(tmp_path,df)[0]["a"][0]["status"] == status


def test_missing_frame_and_keypoint_never_reuse_a_previous_position(tmp_path):
    df=rows();df=df[~((df.frame==1)|((df.frame==2)&(df.mouse=="a")&(df.keypoint=="nose")))]
    out=frames(tmp_path,df)
    assert len(out)==4 and out[1]=={}
    assert out[2]["a"][0]["status"]=="missing" and out[2]["a"][0]["x"] is None


@pytest.mark.parametrize("change",[
    lambda d:pd.concat([d,d.iloc[:1]]),lambda d:d.assign(mouse="c"),
    lambda d:d.assign(keypoint="invented"),lambda d:d.assign(frame=-1),
    lambda d:d.assign(frame=.5),lambda d:d.assign(frame=np.inf),
    lambda d:d.assign(session="another"),lambda d:d.drop(columns="session"),
    lambda d:d.iloc[:0],lambda d:d.assign(frame=2_000_000),
])
def test_ambiguous_source_refused(tmp_path,change):
    with pytest.raises(ValueError):frames(tmp_path,change(rows()))


def test_mixed_sessions_refused(tmp_path):
    df=rows();df.loc[0,"session"]="another"
    with pytest.raises(ValueError,match="session"):frames(tmp_path,df)


@pytest.mark.parametrize("fps",[0,-1,float("inf"),float("nan"),True])
def test_bad_rate_refused(fps):
    with pytest.raises(ValueError):sp.payload([],fps)


def test_unknown_confidence_is_not_connected_as_anatomy():
    sk=mouse_skeleton("a",{k:dict(x=10.,y=20.) for k in KEYPOINTS})
    assert sk.n_tracked==0 and not sk.edges and len(sk.omitted_edges)==8


def test_new_uploads_keep_missing_score_provenance_without_changing_backend_values(tmp_path):
    df=rows(30).drop(columns="score")
    normalized=uploads.normalize_keypoints(df)
    assert normalized.score.eq(1).all()  # previous monitoring semantics unchanged
    assert not normalized.score_observed.any()
    assert frames(tmp_path,normalized)[0]["a"][0]["status"]=="unreliable"


def test_partially_absent_scores_remain_distinguishable_after_upload(tmp_path):
    df=rows(30);df.loc[0,"score"]=np.nan
    normalized=uploads.normalize_keypoints(df)
    out=frames(tmp_path,normalized)
    assert out[0]["a"][0]["status"]=="unreliable"
    assert out[0]["a"][1]["status"]=="tracked"


def test_disclosures_and_provenance_use_text_not_executable_html():
    marker="</script><script>throw Error('untrusted')</script>"
    page=sp.html([],video_url="/synthetic.mp4",session=marker)
    assert marker not in page
    assert "PROVISIONAL connections" in page
    assert "not confirmed official CalMS21" in page
    assert "No display interpolation" in page
    assert "confidence" in page and "recorded" in page


def test_replay_uses_selected_session_and_discloses_its_role(tmp_path,monkeypatch):
    video=tmp_path/"video.mp4";video.write_bytes(fake_mp4())
    poses=tmp_path/"p.parquet";rows().to_parquet(poses,index=False)
    calls=[];monkeypatch.setattr(sp,"render",lambda *a,**k:calls.append((a,k)))
    at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"pages/2_Replay.py"))
    at.session_state[flow.SOURCE_KEY]=uploads.SessionSource("video_keypoints","Synthetic",str(video),str(poses),"SYNTHETIC")
    at.run();assert not at.exception
    assert len(calls)==1 and calls[0][1]["session"]=="SYNTHETIC"
    assert calls[0][1]["original_video"]==str(video)
    assert any("does not run monitoring" in c.value for c in at.caption)


def test_wrong_session_never_mounts_a_player(tmp_path,monkeypatch):
    video=tmp_path/"video.mp4";video.write_bytes(fake_mp4())
    poses=tmp_path/"p.parquet";rows().to_parquet(poses,index=False)
    calls=[];monkeypatch.setattr(sp,"render",lambda *a,**k:calls.append((a,k)))
    at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"pages/2_Replay.py"))
    at.session_state[flow.SOURCE_KEY]=uploads.SessionSource("video_keypoints","Synthetic",str(video),str(poses),"WRONG")
    at.run();assert not at.exception and not calls and at.error


def test_reupload_cannot_turn_substituted_confidence_into_measurement(tmp_path):
    normalized=uploads.normalize_keypoints(rows(30).drop(columns="score"))
    again=uploads.normalize_keypoints(normalized)
    assert again.score.eq(1).all() and not again.score_observed.any()
    assert frames(tmp_path,again)[0]["a"][0]["status"]=="unreliable"


@pytest.mark.parametrize("flag,status",[("imputed","imputed"),("jump","unreliable"),
                                      ("low_conf","low_confidence"),("tracking_ok","unreliable")])
def test_supplied_quality_survives_upload_for_display(tmp_path,flag,status):
    df=rows(30);df[flag]=True if flag=="tracking_ok" else False
    df.loc[0,flag]=False if flag=="tracking_ok" else True
    normalized=uploads.normalize_keypoints(df)
    assert flag in normalized
    assert frames(tmp_path,normalized)[0]["a"][0]["status"]==status


def test_invalid_text_coordinates_are_absent_not_drawn(tmp_path):
    df=rows();df["x"]=df.x.astype(str);df.loc[0,"x"]="not a measurement"
    assert frames(tmp_path,df)[0]["a"][0]["status"]=="missing"


def test_mixed_numeric_frame_spellings_cannot_overwrite_each_other(tmp_path):
    df=rows();df["frame"]=df.frame.astype(str)
    df=pd.concat([df,df.iloc[:1].assign(frame="00")],ignore_index=True)
    with pytest.raises(ValueError,match="Duplicate"):frames(tmp_path,df)


def test_standalone_source_text_roundtrips_without_script_injection():
    import json
    marker="</script><script>untrusted</script>"
    page=sp.html([],session=marker)
    data=json.loads(page.rsplit("data:",1)[1].split("});</script>")[0])
    assert data["session"]==marker and marker not in page


def test_legacy_normalized_uploads_never_claim_substitutes_are_measured(tmp_path):
    p=tmp_path/"legacy.parquet";rows().to_parquet(p,index=False)
    out=sp.pose_frames(p,session="SYNTHETIC",require_observed=True)
    assert all(p["status"]=="unreliable" for f in out for m in f.values() for p in m)
    df=rows();df["score_observed"]=True;df.to_parquet(p,index=False)
    assert sp.pose_frames(p,session="SYNTHETIC",require_observed=True)[0]["a"][0]["status"]=="tracked"
