"""Optional real-decoder browser regression. All media is generated in tmp_path.

Run with LIVE_TRACKER_PLAYWRIGHT (installed Node module) and
LIVE_TRACKER_BROWSER_EXE (Brave executable). No downloads or external requests.
The same JS is also verified inside the actual Replay Streamlit page via MCP.
"""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest
from PIL import Image, ImageDraw

from live_tracker import skeleton_player as sp
from live_tracker.calms21 import KEYPOINTS


def test_real_decoded_frames_seeks_quality_and_fallback(tmp_path):
    playwright=os.environ.get("LIVE_TRACKER_PLAYWRIGHT")
    executable=os.environ.get("LIVE_TRACKER_BROWSER_EXE")
    node=shutil.which("node");ffmpeg=shutil.which("ffmpeg")
    if not (playwright and executable and node and ffmpeg):
        pytest.skip("Explicit installed Playwright/browser paths, node and ffmpeg required; never downloaded")
    clip=tmp_path/"synthetic.mp4"
    enc=subprocess.Popen([ffmpeg,"-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","1024x570",
        "-r","30","-i","-","-an","-c:v","libx264","-crf","15","-pix_fmt","yuv420p","-movflags","+faststart",str(clip)],stdin=subprocess.PIPE)
    rows=[]
    try:
        for f in range(90):
            im=Image.new("RGB",(1024,570),"#343434");draw=ImageDraw.Draw(im)
            for bit in range(8):
                x=40+bit*40;draw.rectangle((x,25,x+30,55),fill="white" if f&(1<<bit) else "black")
            for m in "ab":
                for i,k in enumerate(KEYPOINTS):
                    if f==12:continue
                    r=dict(session="SYNTHETIC",frame=f,mouse=m,keypoint=k,x=float((120 if f%2==0 else 800)+i*14),
                           y=float((140 if m=="a" else 360)+i*10),score=.99)
                    if i==0 and m=="a" and f==18:r["score"]=.1
                    if i==0 and m=="a" and f==19:r["score"]=float("nan")
                    if i==0 and m=="a" and f==17:r["x"]=float("nan")
                    rows.append(r)
            enc.stdin.write(im.tobytes())
    finally:
        enc.stdin.close()
    assert enc.wait(timeout=30)==0
    poses=tmp_path/"poses.parquet";pd.DataFrame(rows).to_parquet(poses,index=False)
    data_url="data:video/mp4;base64,"+base64.b64encode(clip.read_bytes()).decode()
    frames=sp.pose_frames(poses,session="SYNTHETIC")
    normal=sp.html(frames,30,video_url=data_url)
    (tmp_path/"normal.html").write_text(normal)
    (tmp_path/"short.html").write_text(sp.html(frames[:35],30,video_url=data_url))
    (tmp_path/"timing.html").write_text(sp.html(frames,24,video_url=data_url))
    (tmp_path/"size.html").write_text(normal.replace('"size": [1024, 570]', '"size": [640, 480]'))
    (tmp_path/"fallback.html").write_text(normal.replace("<script>","<script>HTMLVideoElement.prototype.requestVideoFrameCallback=undefined;</script><script>",1))
    script=tmp_path/"check.cjs";script.write_text(JS)
    completed=subprocess.run([node,str(script),str(tmp_path)],capture_output=True,text=True,timeout=60,
        env={**os.environ,"LIVE_TRACKER_PLAYWRIGHT":playwright,"LIVE_TRACKER_BROWSER_EXE":executable})
    assert completed.returncode==0, completed.stdout+completed.stderr
    evidence=json.loads(completed.stdout)
    assert evidence["mismatches"]==0 and evidence["playback_samples"]>=30
    assert evidence["end_frame"]==89 and evidence["fallback"]=="approximate"
    (tmp_path/"browser-evidence.json").write_text(completed.stdout)


JS = r"""
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.LIVE_TRACKER_PLAYWRIGHT);
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.LIVE_TRACKER_BROWSER_EXE,headless:true});
 try {
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.route('http://**/*',route=>route.abort());await page.route('https://**/*',route=>route.abort());
 const fs=require('node:fs');const video=fs.readFileSync(path.join(process.argv[2],'synthetic.mp4'));
 await page.route('http://replay.invalid/**',async route=>{
   const u=new URL(route.request().url());
   if(u.pathname==='/clip.mp4'){
     const range=route.request().headers()['range'];
     if(range){const parts=/bytes=(\d+)-(\d*)/.exec(range);const start=Number(parts[1]),end=parts[2]?Math.min(Number(parts[2]),video.length-1):video.length-1;
       await route.fulfill({status:206,headers:{'Content-Type':'video/mp4','Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${video.length}`},body:video.subarray(start,end+1)});
     }else await route.fulfill({status:200,contentType:'video/mp4',headers:{'Accept-Ranges':'bytes'},body:video});
   }else {const content=fs.readFileSync(path.join(process.argv[2],u.pathname.slice(1)),'utf8').replace(/data:video\/mp4;base64,[A-Za-z0-9+/=]+/g,'http://replay.invalid/clip.mp4');
     await route.fulfill({status:200,contentType:'text/html',body:content});}
 });
 const open=async(name)=>{await page.goto('http://replay.invalid/'+name+'.html');
   await page.locator('video').evaluate(v=>new Promise(resolve=>{if(v.readyState>=2)resolve();else v.addEventListener('loadeddata',resolve,{once:true});}));
   // Headless Brave may defer the first paused composition. This is not a
   // substitute for the separately checked, initially paused Streamlit player.
   await page.locator('video').evaluate(async v=>{await v.play();});
   await page.waitForFunction(()=>document.querySelector('canvas').dataset.frame!==undefined);
   await page.locator('video').evaluate(v=>{v.pause();v.currentTime=.01;});
   await page.waitForFunction(()=>document.querySelector('canvas').dataset.frame==='0');};
 const sample=()=>page.evaluate(()=>{
   const cv=document.querySelector('canvas'),v=document.querySelector('video');
   const copy=document.createElement('canvas');copy.width=1024;copy.height=570;copy.getContext('2d').drawImage(v,0,0);
   const decode=c=>{const g=c.getContext('2d');let n=0;for(let b=0;b<8;b++)if(g.getImageData(55+b*40,40,1,1).data[0]>128)n|=1<<b;return n;};
   return {frame:Number(cv.dataset.frame),pixels:decode(cv),videoPixels:decode(copy),clock:cv.dataset.clock,
     note:document.querySelector('.note').textContent,quality:document.querySelector('.quality').textContent};
 });
 const seek=async(t)=>{await page.locator('video').evaluate((v,t)=>new Promise(r=>{
   v.addEventListener('seeked',r,{once:true});v.currentTime=t;
 }),t);await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));};
 await open('normal');await page.locator('.mode').selectOption('overlay');
 let initial=await sample();assert.equal(initial.frame,0);assert.equal(initial.pixels,0);
 const seeks=[];
 for(const t of [31/30,31.1/30,31.49/30,31.99/30,32/30,32.01/30,33/30,5/30,0,89/30]){
   await seek(t);const s=await sample();seeks.push({t,...s});assert.equal(s.frame,s.pixels);assert.equal(s.frame,s.videoPixels);
 }
 // Real user controls must settle on adjacent frames, including after ending.
 await page.locator('.next').click();await page.waitForFunction(()=>document.querySelector('canvas').dataset.frame==='89');
 await page.locator('.prev').click();await page.waitForFunction(()=>document.querySelector('canvas').dataset.frame==='88');
 await page.locator('.next').click();await page.waitForFunction(()=>document.querySelector('canvas').dataset.frame==='89');
 // Rapid motion alternates between opposite sides of the image each frame.
 await seek(.01);
 const playback=await page.locator('video').evaluate(v=>new Promise(async(resolve)=>{
   const cv=document.querySelector('canvas'),g=cv.getContext('2d');const samples=[];
   const tick=(now,meta)=>{let pixels=0;for(let b=0;b<8;b++)if(g.getImageData(55+b*40,40,1,1).data[0]>128)pixels|=1<<b;
     samples.push({frame:Number(cv.dataset.frame),pixels});if(!v.ended)v.requestVideoFrameCallback(tick);};
   v.requestVideoFrameCallback(tick);v.addEventListener('ended',()=>resolve(samples),{once:true});await v.play();
 }));
 const end=await sample();assert.equal(end.frame,89);assert.equal(end.pixels,89);
 const mismatches=playback.filter(s=>s.frame!==s.pixels);assert.equal(mismatches.length,0);
 await seek(32.5/30);const paused=await sample();
 await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));assert.deepEqual(await sample(),paused);
 await page.locator('.mode').selectOption('side');
 for(const [f,status] of [[12,'No pose recorded'],[17,'missing: 1'],[18,'low_confidence: 1'],[19,'unreliable: 1']]){
   await seek((f+.5)/30);assert.equal((await sample()).frame,f);
   assert.ok((await page.locator(status.startsWith('No')?'.note':'.quality').innerText()).includes(status));
   if(f===18 || f===19)assert.equal(await page.locator('canvas').evaluate((c,f)=>c.getContext('2d').getImageData(f%2?800:120,140,1,1).data[3],f),0);
 }
 await open('short');await page.locator('.mode').selectOption('overlay');await seek(40.5/30);
 assert.equal((await sample()).frame,40);assert.ok((await sample()).note.includes('No pose recorded'));
 await open('timing');await seek(1.5/30);assert.match(await page.locator('.note').innerText(),/Timing mismatch/);
 // Different declared image dimensions must not rescale poses silently.
 await page.goto('http://replay.invalid/size.html');
 await page.locator('video').evaluate(v=>v.play());
 await page.waitForFunction(()=>document.querySelector('.note').textContent.includes('Coordinate mismatch'));
 assert.equal(await page.locator('canvas').getAttribute('data-frame'),null);
 await open('fallback');assert.equal(await page.locator('.mode option[value=overlay]').evaluate(e=>e.disabled),true,
   JSON.stringify(await page.evaluate(()=>({method:typeof document.querySelector('video').requestVideoFrameCallback,note:document.querySelector('.timing').textContent,disabled:document.querySelector('.mode option[value=overlay]').disabled}))));
 assert.match(await page.locator('.timing').innerText(),/APPROXIMATE/);
 await seek(31.8/30);assert.equal((await sample()).frame,31);const fallback=(await sample()).clock;
 assert.equal(errors.length,0,errors.join('\n'));
 console.log(JSON.stringify({initial,seeks,playback_samples:playback.length,mismatches:mismatches.length,end_frame:end.frame,fallback}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
"""
