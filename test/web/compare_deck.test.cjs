// Exercise visibility and audio selection independently of rendering the DOM.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const saved = new Map();
const context = {window:{PureSoundAudio:{DEFAULT_SPECTROGRAM:{}, clamp:(value,low,high)=>Math.min(high,Math.max(low,value))}}, localStorage:{setItem:(k,v)=>saved.set(k,v)}, requestAnimationFrame:()=>{}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../../puresound/web/static/compare-deck.js'),'utf8'),context);
const Deck = context.window.PureSoundCompareDeck;
function deck() {
  const d = Object.create(Deck.prototype);
  Object.assign(d, {
    tracks:['input','aligned','source-a'].map(id=>({id,buffer:{},lane:{hidden:false,classList:{toggle(){}}}})),
    visibleTracks:new Set(['input','aligned']), selectedId:'aligned', playing:true,
    visibilityKey:'test-visible', listeningEnabled:true, listening:true, viewMode:'wave',viewBeforeListening:'both',
    root:{classList:{toggle(){}},querySelectorAll:()=>[]},
    renderTrackButtons(){}, updateControls(){}, updateLimiter(){}, draw(){},
    applyGains(seconds){this.gainTransition=seconds},
    pause(){this.playing=false}, setView(mode){this.viewMode=mode},
  });
  return d;
}
test('hiding the audible track switches to a checked track without stopping playback',()=>{
  const d=deck();d.setTrackVisible('aligned',false);
  assert.equal(d.selectedId,'input');assert.equal(d.playing,true);
  assert.equal(d.gainTransition,0.004);
  assert.equal(d.track('aligned').lane.hidden,true);
  assert.deepEqual(JSON.parse(saved.get('test-visible')),['input']);
  d.setTrackVisible('input',false);
  assert.deepEqual([...d.visibleTracks],['input']);
});
test('selecting a hidden source reveals only that source and remembers the combination',()=>{
  const d=deck();d.select('source-a');
  assert.deepEqual([...d.visibleTracks],['input','aligned','source-a']);
  assert.equal(d.track('source-a').lane.hidden,false);
  assert.deepEqual(JSON.parse(saved.get('test-visible')),['input','aligned','source-a']);
  d.setTrackVisible('missing',true);assert.equal(d.visibleTracks.size,3);
});
test('listen and inspect keep the same comparison and restore the chosen analysis view',()=>{
  const d=deck();d.setListening(false);assert.equal(d.viewMode,'both');
  d.viewMode='spec';d.setListening(true);assert.equal(d.viewMode,'wave');
  d.setListening(false);assert.equal(d.viewMode,'spec');
  assert.deepEqual([...d.visibleTracks],['input','aligned']);assert.equal(d.playing,true);
});
test('hiding the current track waits safely when the remaining track has not loaded',()=>{
  const d=deck();d.track('input').buffer=null;d.setTrackVisible('aligned',false);
  assert.equal(d.playing,false);assert.equal(d.selectedId,'input');
  assert.deepEqual([...d.visibleTracks],['input']);
});
test('a gesture the browser cancels leaves no listener behind and seeks nowhere',()=>{
  const listeners=new Map();
  context.window.addEventListener=(type,fn)=>listeners.set(type,[...(listeners.get(type)||[]),fn]);
  context.window.removeEventListener=(type,fn)=>listeners.set(type,(listeners.get(type)||[]).filter(item=>item!==fn));
  const d=deck();d.seeks=[];
  Object.assign(d,{area:{getBoundingClientRect:()=>({left:0,width:100})},seekTo(seconds){this.seeks.push(seconds)}});
  d.tracks.forEach(track=>{track.buffer={duration:4}});
  d.pointerDown({button:0,clientX:20,target:{closest:()=>null}});
  listeners.get('pointercancel')[0]({});
  for(const type of ['pointermove','pointerup','pointercancel'])assert.equal(listeners.get(type).length,0,type);
  assert.deepEqual(d.seeks,[]);
});
