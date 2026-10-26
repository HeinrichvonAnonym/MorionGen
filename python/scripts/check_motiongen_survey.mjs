// Actual Chrome CDP inspection of the compiled report, not a DOM-only mock.
import fs from 'node:fs';
const root='/home/heinrich/MorionGen/develop_logs/test_runs';
const tabs=await(await fetch('http://127.0.0.1:9223/json/list')).json();
const tab=tabs.find(t=>t.url.startsWith('http://127.0.0.1:4174/'));
if(!tab)throw Error('Survey preview is not open');
const socket=new WebSocket(tab.webSocketDebuggerUrl),pending=new Map(),errors=[];
let next=0;
socket.onmessage=e=>{const r=JSON.parse(e.data);if(r.method==='Runtime.exceptionThrown')errors.push(r.params.exceptionDetails);if(r.id){const p=pending.get(r.id);pending.delete(r.id);r.error?p.reject(r.error):p.resolve(r.result)}};
await new Promise(resolve=>socket.onopen=resolve);
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const settle=()=>new Promise(r=>setTimeout(r,250));
await call('Runtime.enable');await call('Page.enable');await call('Page.reload',{ignoreCache:true});
await new Promise(r=>setTimeout(r,1800));
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
const checks={};
checks.initial=await evaluate(`({title:document.querySelector('h1')?.innerText,body:document.body.innerText.slice(0,2200),svg:document.querySelectorAll('svg').length})`);
async function pick(id,value){await evaluate(`(()=>{let e=document.getElementById(${JSON.stringify(id)});if(!e)throw Error('Missing control ${id}');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('change',{bubbles:true}));})()`);await settle()}
const block=id=>`document.querySelector('[data-component-id="${id}"]')`;
checks.methodFilters=[];
for(const value of ['run','physics','all']){await pick('method-filter',value);checks.methodFilters.push(await evaluate(`({value:${JSON.stringify(value)},rows:${block('method-matrix')}?.querySelectorAll('tbody tr').length})`))}
await evaluate(`document.getElementById('all-diagrams').click()`);await settle();
checks.diagrams=await evaluate(`({count:document.querySelectorAll('.survey-diagrams svg,.survey-feedback svg').length,labels:[...document.querySelectorAll('.survey-diagrams svg,.survey-feedback svg')].map(s=>s.getAttribute('aria-label'))})`);
await evaluate(`document.getElementById('all-diagrams').click()`);await pick('diagram-method','PRIMAL');
await evaluate(`document.querySelector('.survey-diagrams').scrollIntoView({block:'start'});window.scrollBy(0,-70)`);await settle();
fs.writeFileSync(`${root}/20261024_survey_diagram.png`,Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
checks.groups=[];
for(const method of ['dart','primal'])for(const clock of ['paused','realtime']){await pick('rep-method',method);await pick('rep-clock',clock);checks.groups.push(await evaluate(`({method:${JSON.stringify(method)},clock:${JSON.stringify(clock)},table:${block('replication-table')}?.innerText,marks:document.querySelectorAll('.recharts-bar-rectangle').length})`))}
await pick('rep-metric','jerk_rms');checks.jerk=await evaluate(`${block('replication-table')}?.innerText`);
await pick('rep-metric','mean_error_m');
checks.sensitivity=[];
for(const window of ['5','10']){await pick('sensitivity-window',window);checks.sensitivity.push(await evaluate(`({window:${JSON.stringify(window)},text:${block('sensitivity-table')}?.innerText})`))}
await pick('motion-scenario','standing_reverse');
checks.motion=await evaluate(`({count:document.querySelectorAll('.survey-motion-pair svg').length,frame:document.getElementById('motion-frame')?.value})`);
await evaluate(`document.getElementById('motion-play')?.click()`);await new Promise(r=>setTimeout(r,600));
checks.motion.frameAfter=await evaluate(`document.getElementById('motion-frame')?.value`);
await evaluate(`document.getElementById('motion-play')?.click()`);
await pick('archive-select','20261009_kingston_portable_bundle.md');checks.archive=await evaluate(`document.querySelector('.survey-archive-text')?.textContent.length`);await pick('archive-select','');
await pick('rep-method','dart');await pick('rep-clock','realtime');
await evaluate(`document.querySelector('[data-component-id="scenario-chart"]').scrollIntoView({block:'start'});window.scrollBy(0,-70)`);await settle();
fs.writeFileSync(`${root}/20261024_survey_chart.png`,Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await evaluate(`(()=>{let s=document.querySelector('[data-component-id="replication-table"] .component-native-actions select');if(s){s.value='0';s.dispatchEvent(new Event('change',{bubbles:true}))}})()`);await settle();
checks.source=await evaluate(`({dialogs:[...document.querySelectorAll('[role="dialog"]')].map(d=>d.innerText.slice(0,4500)),sidebar:[...document.querySelectorAll('aside')].map(d=>d.innerText.slice(0,4500))})`);
await evaluate(`[...document.querySelectorAll('aside button')].find(b=>b.innerText==='Data preview')?.click()`);await settle();
checks.sourcePreview=await evaluate(`[...document.querySelectorAll('aside')].map(d=>d.innerText.slice(0,2500))`);
await call('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
await call('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});await settle();
await evaluate(`window.scrollTo(0,0)`);await settle();
fs.writeFileSync(`${root}/20261024_survey_desktop.png`,Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await settle();
checks.mobile=await evaluate(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,bodyWidth:document.body.scrollWidth})`);
checks.mobile.authoredOverflow=await evaluate(`([...document.querySelectorAll('.motiongen-survey > *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1).map(e=>({tag:e.tagName,id:e.id,width:e.getBoundingClientRect().width})))`);
fs.writeFileSync(`${root}/20261024_survey_mobile.png`,Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
checks.errors=errors;fs.writeFileSync(`${root}/20261024_survey_browser_checks.json`,JSON.stringify(checks,null,2));
console.log(JSON.stringify(checks,null,2));socket.close();
