const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements={};
const element=id=>elements[id]??=( {value:id==='opacity'?'0.35':id==='bird'?'all':'',checked:true,disabled:true,
 textContent:'',appendChild(){},getContext(){return {drawImage(){},globalAlpha:1};}} );
const pending=new Map();
class MockImage {set src(value){if(value.startsWith('frames/'))pending.set(value,this);else queueMicrotask(()=>this.onload());}}
const context=vm.createContext({document:{getElementById:element,createElement:()=>({}),addEventListener(){}},
 Image:MockImage,console});
const config={gt:2250,start:2150,end:2350,width:10,height:10,pilot:'test',layers:[]};
const html=fs.readFileSync(__dirname+'/review_frame_offset.html','utf8');
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1].replace('__CONFIG__',JSON.stringify(config)),context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 pending.get('frames/002250.jpg').onload();await tick();
 assert(element('status').textContent.includes('Displayed video 2250'));
 element('next').onclick();element('next').onclick();
 assert(element('choose').disabled);
 pending.get('frames/002252.jpg').onload();await tick();
 pending.get('frames/002251.jpg').onload();await tick();
 assert(element('status').textContent.includes('Displayed video 2252'));
 element('choose').onclick();assert(element('selection').textContent.includes('offset +2'));
 element('reset').onclick();pending.get('frames/002250.jpg').onload();await tick();
 assert(element('status').textContent.includes('Offset +0'));
 console.log('Viewer checks passed: loading guard, stale-image rejection, candidate recording, reset.');
})().catch(error=>{console.error(error);process.exitCode=1;});
