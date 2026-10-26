import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL,fileURLToPath} from 'node:url';
const offlineIndex=process.argv.indexOf('--offline');
const offlinePath=offlineIndex>=0?process.argv[offlineIndex+1]:null;
if(offlineIndex>=0&&!offlinePath)throw Error('--offline requires an exported HTML path');
const reportRoot='/home/heinrich/MorionGen/develop_logs/notes/motiongen_survey_revision/dist';
const build=JSON.parse(fs.readFileSync(reportRoot+'/data-app-build.json','utf8'));
const hashFile=path=>crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const htmlBefore=hashFile(reportRoot+'/index.html');
if(htmlBefore!==build.html.sha256)throw Error('HTML does not match build manifest');
if(build.snapshot&&hashFile(reportRoot+'/'+build.snapshot.path)!==build.snapshot.sha256)throw Error('Data snapshot does not match build manifest');
const offlineHash=offlinePath?hashFile(offlinePath):null;
if(offlinePath){
 const exported=fs.readFileSync(offlinePath,'utf8');
 const identity=exported.match(/<meta name="data-app-snapshot-sha256" content="([a-f0-9]{64})"/);
 if(identity?.[1]!==build.sourceSnapshotSha256)throw Error('Offline export does not identify the current reviewed snapshot');
}
const tabs=await(await fetch('http://127.0.0.1:9223/json/list')).json();
const offlineUrl=offlinePath?pathToFileURL(offlinePath).href+'?view=1':null;
let tab=tabs.find(t=>offlinePath?t.url===offlineUrl:t.url.includes('/motiongen_survey_revision/'));
if(offlinePath&&!tab)tab=await(await fetch('http://127.0.0.1:9223/json/new?'+encodeURIComponent(offlineUrl),{method:'PUT'})).json();
if(!tab) throw Error('Revision preview missing');
const socket=new WebSocket(tab.webSocketDebuggerUrl), pending=new Map(),errors=[];
let next=0;
socket.onmessage=e=>{const r=JSON.parse(e.data);if(r.method==='Runtime.exceptionThrown')errors.push(r.params.exceptionDetails);if(r.id){const p=pending.get(r.id);if(!p)return;pending.delete(r.id);clearTimeout(p.timer);r.error?p.reject(r.error):p.resolve(r.result)}};
socket.onclose=e=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(new Error('Browser inspection transport closed: '+e.code+' '+e.reason));}pending.clear();};
socket.onerror=()=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(new Error('Browser inspection transport error'));}pending.clear();};
await new Promise(r=>socket.onopen=r);
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++next;const timer=setTimeout(()=>{pending.delete(id);reject(new Error('Browser inspection timed out: '+method));},30000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}))});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
// Old data-URL exception stacks can exceed the browser transport's message
// limit. Clear only prior logs; all exceptions from this fresh reload are kept.
await call('Runtime.discardConsoleEntries');
await call('Runtime.enable'); await call('Page.enable'); await call('Page.bringToFront'); await call('Page.reload',{ignoreCache:true});
for(let i=0;i<100;i++){
 if(await evaluate(`Boolean(document.getElementById('native-method-filter'))`))break;
 if(i===99)throw Error('Report controls did not load within 20 seconds');
 await new Promise(r=>setTimeout(r,200));
}
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
const checks={initial:await evaluate(`({title:document.querySelector('h1')?.innerText,rows:document.querySelectorAll('[data-component-id="pilot-comparison"] tbody tr').length,methods:document.querySelectorAll('.method-diagrams').length})`),filters:[]};
const reviewedSnapshot=JSON.parse(fs.readFileSync(reportRoot+'/'+build.snapshot.path,'utf8'));
if(reviewedSnapshot.buildStatus==='complete'){
 checks.finalWording=await evaluate(`!['本报告正在重做','其余方法仍按预登记矩阵扩样','这是仍在扩样的中期读数'].some(s=>document.body.innerText.includes(s))`);
 if(!checks.finalWording)throw Error('Completed delivery retains stale in-progress narrative');
}
checks.build={htmlSha256:htmlBefore,snapshotSha256:build.snapshot?.sha256||build.sourceSnapshotSha256,checkedAt:new Date().toISOString()};
if(offlinePath)checks.build.offline={path:offlinePath,sha256:offlineHash,scope:'Exported HTML checked through file://, not an HTTP sidecar build'};
checks.acceptance=await evaluate(`({matrix:document.querySelectorAll('[data-component-id="native-matrix-coverage"] tbody tr').length,invariants:document.querySelectorAll('[data-component-id="native-invariant-checks"] tbody tr').length,body:document.querySelectorAll('[data-component-id="dart-body-compatibility"] tbody tr').length})`);
if(checks.acceptance.matrix!==11||checks.acceptance.invariants!==4||checks.acceptance.body!==3)throw Error('Matrix coverage or output-level audit missing');
checks.acceptance.preprocessing=await evaluate(`document.querySelectorAll('[data-component-id="native-preprocessing-checks"] tbody tr').length`);
if(checks.acceptance.preprocessing!==8)throw Error('Eight-method preprocessing audit missing');
checks.acceptance.reconstruction=await evaluate(`document.querySelectorAll('[data-component-id="native-metric-reconstruction"] tbody tr').length`);
if(checks.acceptance.reconstruction!==7)throw Error('New seven-method metric reconstruction missing');
checks.acceptance.spatialCoordinates=await evaluate(`document.querySelectorAll('[data-component-id="spatial-coordinate-checks"] tbody tr').length`);
if(checks.acceptance.spatialCoordinates!==1)throw Error('MotionLCM native coordinate audit missing');
checks.acceptance.choisObject=await evaluate(`document.querySelectorAll('[data-component-id="chois-independent-object-checks"] tbody tr').length`);
if(checks.acceptance.choisObject!==8)throw Error('CHOIS independent object and anatomical elbow metric evidence missing');
checks.acceptance.choisCheckpoint=await evaluate(`document.querySelectorAll('[data-component-id="chois-checkpoint-load"] tbody tr').length`);
if(checks.acceptance.choisCheckpoint!==1)throw Error('CHOIS actual loaded checkpoint tensor audit missing');
checks.acceptance.choisContact=await evaluate(`document.querySelectorAll('[data-component-id="chois-contact-confusion-checks"] tbody tr').length`);
if(checks.acceptance.choisContact!==6)throw Error('CHOIS full-mesh GT contact confusion audit missing');
checks.acceptance.ardyHistory=await evaluate(`document.querySelectorAll('[data-component-id="ardy-common-history-checks"] tbody tr').length`);
if(checks.acceptance.ardyHistory!==1)throw Error('ARDY common history-span audit missing');
checks.acceptance.ardyTextProfiles=await evaluate(`document.querySelectorAll('[data-component-id="ardy-text-setup-profiles"] tbody tr').length`);
if(checks.acceptance.ardyTextProfiles!==(reviewedSnapshot.queries.ardy_text_setup?.rows.length||0))throw Error('ARDY native vs cached startup profiles missing');
checks.acceptance.environments=await evaluate(`document.querySelectorAll('[data-component-id="native-runtime-environments"] tbody tr').length`);
if(checks.acceptance.environments!==8)throw Error('Eight-method runtime version evidence missing');
checks.acceptance.executionContracts=await evaluate(`document.querySelectorAll('[data-component-id="execution-design-contracts"] tbody tr').length`);
if(checks.acceptance.executionContracts!==5)throw Error('Execution engineering contracts missing');
checks.acceptance.sourceMethods=await evaluate(`document.querySelectorAll('[data-component-id="source-cohort-methods"] tbody tr').length`);
if(checks.acceptance.sourceMethods!==7)throw Error('Frozen-source review must cover seven new inference methods');
checks.acceptance.changedSourceEvents=await evaluate(`document.querySelectorAll('[data-component-id="source-replay-changed-events"] tbody tr').length`);
if(checks.acceptance.changedSourceEvents!==reviewedSnapshot.queries.source_replay_comparisons.rows.filter(r=>r.method==='closd'&&!r.exact_same_coordinates).length)throw Error('Changed CLoSD replay event records missing');
checks.acceptance.epochEffects=await evaluate(`document.querySelectorAll('[data-component-id="execution-epoch-effects"] tbody tr').length`);
if(checks.acceptance.epochEffects!==reviewedSnapshot.queries.execution_epoch_sensitivity.rows.length)throw Error('Cross-reboot timing sensitivity missing');
checks.acceptance.trajectoryCoverage=await evaluate(`document.querySelectorAll('[data-component-id="trajectory-coverage"] tbody tr').length`);
if(checks.acceptance.trajectoryCoverage!==1)throw Error('Supplementary trajectory coverage missing');
checks.acceptance.clockPairs=await evaluate(`document.querySelectorAll('[data-component-id="native-clock-pairs"] tbody tr').length`);
if(checks.acceptance.clockPairs!==4)throw Error('Four online methods need cross-clock trajectory accounting');
checks.acceptance.clockObservationBounds=await evaluate(`document.querySelectorAll('[data-component-id="native-clock-observation-bounds"] tbody tr').length`);
if(checks.acceptance.clockObservationBounds!==2)throw Error('DART/CLoSD completion timestamp precision audit missing');
checks.acceptance.degenerateIntervalLabel=await evaluate(`document.querySelector('[data-component-id="native-capability-rates"]').innerText.includes('样本退化')`);
const expectedDegeneracy=JSON.parse(fs.readFileSync(reportRoot+'/'+build.snapshot.path,'utf8')).queries.native_capability_rates.rows.some(r=>r.clock==='paused'&&r.interval_degenerate);
if(checks.acceptance.degenerateIntervalLabel!==expectedDegeneracy)throw Error('Empirical-degeneracy labels must match the reviewed values, without requiring any particular outcome');
const expectedPrimalTiming=JSON.parse(fs.readFileSync(reportRoot+'/'+build.snapshot.path,'utf8')).queries.primal_timing?.rows.length||0;
checks.acceptance.primalTiming=await evaluate(`document.querySelectorAll('[data-component-id="primal-timing-decomposition"] tbody tr').length`);
if(checks.acceptance.primalTiming!==expectedPrimalTiming)throw Error('PRIMAL pacing evidence rows missing');
checks.acceptance.primalPrimaryTiming=await evaluate(`Array.from(document.querySelectorAll('[data-component-id="primal-primary-timing"] tbody tr')).map(tr=>Array.from(tr.cells).map(td=>td.textContent))`);
for(const [i,clock] of ['paused','realtime'].entries()){
 const rows=reviewedSnapshot.queries.primal_primary_timing.rows.filter(r=>r.clock===clock);
 if(checks.acceptance.primalPrimaryTiming[i]?.[1]!==`${rows.filter(r=>r.passed).length} / ${rows.length}`||rows.length!==480)throw Error('Full PRIMAL timing decomposition incomplete');
}
for(const value of ['omnicontrol','closd','all']){
 await evaluate(`(()=>{const s=document.getElementById('native-method-filter');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'${value}');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 await new Promise(r=>setTimeout(r,200));
 checks.filters.push(await evaluate(`({value:'${value}',rows:document.querySelectorAll('[data-component-id="pilot-comparison"] tbody tr').length,text:document.querySelector('[data-component-id="pilot-comparison"]')?.innerText})`));
}
const out='/home/heinrich/MorionGen/develop_logs/test_runs/20261024_native_pilots'+(offlinePath?'/offline':'');
fs.mkdirSync(out,{recursive:true});
checks.diagrams=await evaluate(`({count:document.querySelectorAll('.method-diagrams svg').length,labels:[...document.querySelectorAll('.method-diagrams svg')].map(x=>x.getAttribute('aria-label'))})`);
if(checks.filters[0].rows!==4||checks.filters[1].rows!==0)throw Error('Method filter leaked unrelated rows');
await evaluate(`(()=>{const s=document.getElementById('effect-metric');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'jerk_rms');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,200));
checks.effects=await evaluate(`({rows:document.querySelectorAll('[data-component-id="native-effect-table"] tbody tr').length,marks:document.querySelectorAll('[data-component-id="native-effect-chart"] .recharts-bar-rectangle').length,text:document.querySelector('[data-component-id="native-effect-table"]')?.innerText})`);
if(!checks.effects.rows||!checks.effects.marks)throw Error('Factorial effect chart/filter failed');
checks.effects.taskRows=await evaluate(`document.querySelectorAll('[data-component-id="native-task-effects"] tbody tr').length`);
checks.effects.omissionRows=await evaluate(`document.querySelectorAll('[data-component-id="native-omission-sensitivity"] tbody tr').length`);
if(!checks.effects.taskRows||checks.effects.omissionRows!==checks.effects.rows)throw Error('Task spread or omission sensitivity missing');
checks.effects.seedRows=await evaluate(`document.querySelectorAll('[data-component-id="native-seed-sensitivity"] tbody tr').length`);
if(checks.effects.seedRows!==checks.effects.rows)throw Error('Seed sensitivity must cover every selected factorial contrast');
checks.effects.simpleRows=await evaluate(`document.querySelectorAll('[data-component-id="native-simple-effects"] tbody tr').length`);
if(checks.effects.simpleRows!==20)throw Error('All 2x2/2x2x2 conditional effects must be inspectable for jerk');
await evaluate(`(()=>{const t=document.querySelector('[data-component-id="native-simple-effects"]');t.closest('details').open=true;t.scrollIntoView({block:'start'});})()`);
fs.writeFileSync(out+'/preview_conditional_effects.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await evaluate(`document.querySelector('[data-component-id="native-effect-chart"]').scrollIntoView({block:'start'})`);
fs.writeFileSync(out+'/preview_effects.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
checks.tradeoff=await evaluate(`({marks:document.querySelectorAll('[data-component-id="native-benefit-cost"] .recharts-scatter-symbol').length,capabilityRows:document.querySelectorAll('[data-component-id="native-capability-rates"] tbody tr').length})`);
if(!checks.tradeoff.marks||!checks.tradeoff.capabilityRows)throw Error('Benefit-cost marks or capability denominators absent');
await evaluate(`document.querySelector('[data-component-id="native-benefit-cost"]').scrollIntoView({block:'start'})`);
fs.writeFileSync(out+'/preview_tradeoff.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
checks.migrated=await evaluate(`({rawRows:document.querySelectorAll('[data-component-id="historical-raw-ability"] tbody tr').length,kitRows:document.querySelectorAll('[data-component-id="kit-reference"] tbody tr').length,tmrRows:document.querySelectorAll('[data-component-id="historical-tmr"] tbody tr').length,timingRows:document.querySelectorAll('[data-component-id="native-timing-table"] tbody tr').length})`);
if(checks.migrated.rawRows!==3||checks.migrated.kitRows!==9||!checks.migrated.tmrRows||!checks.migrated.timingRows)throw Error('Retained 1007 analytical sections missing');
checks.uses=await evaluate(`(()=>{const selects=[...document.querySelectorAll('.survey-guide select')];const s=selects[0];if(!s)return null;Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'primal');s.dispatchEvent(new Event('change',{bubbles:true}));return {options:s.options.length,migrationRows:document.querySelectorAll('[data-component-id="survey-migration-ledger"] tbody tr').length};})()`);
await new Promise(r=>setTimeout(r,150));
checks.uses.text=await evaluate(`document.querySelector('[data-component-id="native-uses-comparison"]')?.innerText`);
if(checks.uses.options!==8||checks.uses.migrationRows!==12||!checks.uses.text.includes('primal'))throw Error('Eight-method uses or migration ledger failed');
checks.paperAndWeights=await evaluate(`({papers:document.querySelectorAll('[data-component-id="published-architecture-evidence"] tbody tr').length,kimodoWeights:document.querySelectorAll('[data-component-id="kimodo-weight-checks"] tbody tr').length})`);
if(checks.paperAndWeights.papers!==2||checks.paperAndWeights.kimodoWeights!==19)throw Error('Published vs local evidence or Kimodo weight audit missing');
checks.paperAndWeights.kimodoReplay=await evaluate(`document.querySelectorAll('[data-component-id="kimodo-recorrection"] tbody tr').length`);
if(checks.paperAndWeights.kimodoReplay!==5)throw Error('Kimodo raw-to-correction reconstruction missing');
checks.kimodo=await evaluate(`(()=>{const s=document.getElementById('kimodo-family-filter');if(!s)return null;const selected=s.options[1].value;Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,selected);s.dispatchEvent(new Event('change',{bubbles:true}));return {options:s.options.length,selected};})()`);
await new Promise(r=>setTimeout(r,200));
checks.kimodo.rows=await evaluate(`document.querySelectorAll('[data-component-id="kimodo-correction-table"] tbody tr').length`);
if(!checks.kimodo.rows)throw Error('Kimodo family filtering returned no evidence');
if(checks.diagrams.count!==16) throw Error('All eight methods require two diagrams');
checks.methodPages=[];
for(const method of ['motionlcm','omnicontrol','chois','ardy','kimodo','dart','primal','closd']){
 await evaluate(`(()=>{const s=document.getElementById('native-method-filter');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'${method}');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 await new Promise(r=>setTimeout(r,150));
 const page={method,diagrams:await evaluate(`document.querySelectorAll('.method-diagrams svg').length`),domains:[]};
 if(page.diagrams!==2)throw Error('Missing method diagram: '+method);
 page.sourceLinks=await evaluate(`document.querySelector('.method-diagrams').querySelectorAll('a[href*="/blob/"]').length`);
 if(page.sourceLinks<2)throw Error('Fixed-commit diagram evidence links absent: '+method);
 if(method==='ardy'){
  const complete=reviewedSnapshot.queries.matrix_acceptance.rows.filter(r=>r.method==='ardy').every(r=>r.accepted_cells===r.expected_cells);
  page.completeFindings=await evaluate(`Boolean(document.querySelector('[data-component-id="ardy-complete-findings"]'))`);
  if(page.completeFindings!==complete)throw Error('Complete ARDY interpretation visibility disagrees with its matrix');
  if(complete){
   page.historyInteraction=await evaluate(`document.querySelector('[data-component-id="ardy-complete-findings"]').innerText.includes('方向相反')`);
   if(!page.historyInteraction)throw Error('ARDY history/correction interaction interpretation missing');
   const clock=reviewedSnapshot.queries.native_clock_pairs.rows.find(r=>r.method==='ardy');
   const expected=clock.paired_conditions+' 个跨时钟配对中，'+clock.exact_same_coordinates+' 个关节轨迹逐元素相同';
   page.clockNarrative=await evaluate(`document.querySelector('[data-component-id="ardy-complete-findings"]').innerText.includes(${JSON.stringify(expected)})`);
   if(!page.clockNarrative)throw Error('ARDY narrative counts summary rows instead of paired trajectories');
  }
 }
 if(method==='omnicontrol'){
  page.trajectorySummary=await evaluate(`[...document.querySelectorAll('[data-component-id="trajectory-condition-summary"] tbody tr')].map(r=>r.cells[1].innerText.trim())`);
  const expected=[[0,0],[0,1],[1,0],[1,1]].map(([realism,spatial])=>{const group=reviewedSnapshot.queries.trajectory_runs.rows.filter(r=>r.condition.realism===realism&&r.condition.spatial===spatial);return group.filter(r=>r.values.all_anchors_success_5cm).length+' / '+group.length;});
  if(JSON.stringify(page.trajectorySummary)!==JSON.stringify(expected))throw Error('Trajectory all-anchor success summary disagrees with source rows');
  page.trajectoryErrorCurves=await evaluate(`document.querySelectorAll('[data-component-id="trajectory-anchor-errors"] .recharts-line-curve').length`);
  if(page.trajectoryErrorCurves!==3)throw Error('Two spatial-on error curves and the threshold are required');
  await evaluate(`document.querySelector('[data-component-id="trajectory-anchor-errors"]').scrollIntoView({block:'start'})`);
  fs.writeFileSync(out+'/preview_trajectory_anchor_errors.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
  await evaluate(`(()=>{const s=document.getElementById('trajectory-error-display');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'all');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await new Promise(r=>setTimeout(r,150));
  if(await evaluate(`document.querySelectorAll('[data-component-id="trajectory-anchor-errors"] .recharts-line-curve').length`)!==5)throw Error('All four trajectory error curves plus threshold must be switchable');
 }
 if(method==='closd'){
  const nativeRuns=reviewedSnapshot.queries.native_runs.rows.filter(r=>r.method==='closd');
  const expectedResetRows=['paused','realtime'].flatMap(clock=>[0,1].map(condition=>{
   const rows=nativeRuns.filter(r=>r.clock===clock&&r.condition.execution_context===condition);
   return ['new_goal_success_30cm_hold_05s','new_goal_success_before_next_reset','new_goal_success_reset_free_episode'].map(metric=>rows.filter(r=>r.values[metric]===1).length+' / '+rows.length);
  }));
  page.resetSuccess=await evaluate(`[...document.querySelectorAll('[data-component-id="closd-reset-success"] tbody tr')].map(r=>[...r.cells].slice(1).map(c=>c.innerText.trim()))`);
  if(JSON.stringify(page.resetSuccess)!==JSON.stringify(expectedResetRows))throw Error('CLoSD reset-aware success table disagrees with accepted runs');
  page.resetCases=await evaluate(`document.querySelectorAll('[data-component-id="closd-reset-events"] tbody tr').length`);
  if(page.resetCases!==nativeRuns.filter(r=>r.reset_semantics.reset_frames.length).length)throw Error('CLoSD reset cases missing');
  const currentEffect=reviewedSnapshot.queries.native_effects.rows.find(r=>r.method==='closd'&&r.clock==='realtime'&&r.metric==='new_goal_final_error_m'&&r.factors.join()==='execution_context');
  const expected='实时条件为 '+(currentEffect.mean_equal_group_weight*100).toFixed(2);
  page.currentNarrative=await evaluate(`document.body.innerText.includes(${JSON.stringify(expected)})`);
  if(!page.currentNarrative)throw Error('CLoSD authored narrative retains a stale numeric result');
  const clock=reviewedSnapshot.queries.native_clock_pairs.rows.find(r=>r.method==='closd');
  const expectedClock=clock.paired_conditions+' 个跨时钟配对中，'+clock.exact_same_coordinates+' 个轨迹逐元素相同，'+clock.different_coordinates+' 个不同';
  page.clockNarrative=await evaluate(`document.body.innerText.includes(${JSON.stringify(expectedClock)})`);
  if(!page.clockNarrative)throw Error('CLoSD narrative counts summary rows instead of paired trajectories');
 }
 if(method==='closd'&&reviewedSnapshot.queries.closd_marker_cases?.rows.length){
  const expectedCases=reviewedSnapshot.queries.closd_marker_cases.rows.length;
  page.markerDiagnostic=await evaluate(`(()=>{const s=document.querySelector('select[aria-label="CLoSD marker diagnostic case"]');if(!s)return null;const chosen=s.options[s.options.length-1].value;Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,chosen);s.dispatchEvent(new Event('change',{bubbles:true}));return {cases:s.options.length,chosen};})()`);
  await new Promise(r=>setTimeout(r,150));
  page.markerDiagnostic.rows=await evaluate(`document.querySelectorAll('[data-component-id="closd-marker-replay-checks"] tbody tr').length`);
  page.markerDiagnostic.curves=await evaluate(`document.querySelectorAll('[data-component-id="closd-marker-trajectory-difference"] .recharts-line-curve').length`);
  if(page.markerDiagnostic.cases!==expectedCases||page.markerDiagnostic.rows!==4||page.markerDiagnostic.curves!==2)throw Error('Marker diagnostic case filtering or evidence chart failed');
  const fullPath=await evaluate(`document.querySelector('[data-component-id="closd-marker-trajectory-difference"] .recharts-line-curve').getAttribute('d')`);
  await evaluate(`(()=>{const s=document.querySelector('select[aria-label="CLoSD marker diagnostic time window"]');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'update');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await new Promise(r=>setTimeout(r,200));
  page.markerDiagnostic.windowChanged=await evaluate(`document.querySelector('[data-component-id="closd-marker-trajectory-difference"] .recharts-line-curve').getAttribute('d')!==${JSON.stringify(fullPath)}`);
  if(!page.markerDiagnostic.windowChanged)throw Error('Marker update-window control did not change plotted data');
  page.markerDiagnostic.forceCurves=await evaluate(`document.querySelectorAll('[data-component-id="closd-marker-contact-force"] .recharts-line-curve').length`);
  if(page.markerDiagnostic.forceCurves!==4)throw Error('Four marker force traces missing');
  await evaluate(`document.querySelector('[data-component-id="closd-marker-trajectory-difference"]').scrollIntoView({block:'start'})`);
  fs.writeFileSync(out+'/preview_closd_marker.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
 }
 if(method!=='kimodo'){
  page.findingRows=await evaluate(`document.querySelectorAll('[data-component-id="${method}-finding-estimates"] tbody tr').length`);
  if(!page.findingRows)throw Error('Method-specific component interpretation absent: '+method);
  if(['ardy','dart','primal','closd'].includes(method)){
   await evaluate(`(()=>{const s=document.querySelector('select[aria-label="${method} findings clock"]');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'realtime');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
   await new Promise(r=>setTimeout(r,100));
   page.realtimeFindingRows=await evaluate(`document.querySelectorAll('[data-component-id="${method}-finding-estimates"] tbody tr').length`);
   if(!page.realtimeFindingRows)throw Error('Real-time component interpretation absent: '+method);
   await evaluate(`(()=>{const s=document.querySelector('select[aria-label="${method} findings clock"]');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'paused');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  }
 }
 for(const domain of ['A','B','E','H']){
  const changed=await evaluate(`(()=>{const s=document.querySelector('select[aria-label="${method} metric domain"]');if(!s)return false;Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'${domain}');s.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`);
  if(changed){await new Promise(r=>setTimeout(r,100));page.domains.push(await evaluate(`({domain:'${domain}',rows:document.querySelectorAll('[data-component-id="${method}-case-metrics"] tbody tr').length})`));}
  if(domain==='A'&&method==='chois'){
   page.contactPersistence=await evaluate(`['geometric_hand_contact_longest_span_seconds','geometric_hand_contact_eligible_transitions'].every(k=>document.querySelector('[data-component-id="chois-case-metrics"]').innerText.includes(k))`);
   if(!page.contactPersistence)throw Error('CHOIS sampled same-hand contact persistence absent from A view');
  }
  if(domain==='H'&&method==='primal'&&reviewedSnapshot.queries.native_runs.rows.some(r=>r.method==='primal'&&r.task===0&&r.seed===20&&r.clock==='paused'&&r.values.nonwait_generation_rtf!=null)){
   page.nonwaitColumns=await evaluate(`['paced_wait_seconds','nonwait_generation_seconds','nonwait_generation_rtf'].every(k=>document.querySelector('[data-component-id="primal-case-metrics"]').innerText.includes(k))`);
   if(!page.nonwaitColumns)throw Error('PRIMAL primary wait/nonwait measurements missing from H view');
  }
  if(domain==='H'&&['ardy','dart','closd'].includes(method)){
   page.latencyCurves=await evaluate(`document.querySelectorAll('[data-component-id="${method}-request-latency"] .recharts-line-curve').length`);
   if(page.latencyCurves!==(method==='ardy'?4:2))throw Error('Complete paired request latency curves missing: '+method);
   await evaluate(`document.querySelector('[data-component-id="${method}-request-latency"]').scrollIntoView({block:'start'})`);
   fs.writeFileSync(out+'/preview_'+method+'_latency.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
  }
  if(method==='closd'&&domain==='E'){
   page.forceRows=await evaluate(`document.querySelectorAll('[data-component-id="closd-force-sensitivity"] tbody tr').length`);
   if(page.forceRows!==6)throw Error('Both CLoSD conditions need three force thresholds');
   await evaluate(`document.querySelector('[data-component-id="closd-force-sensitivity"]').scrollIntoView({block:'start'})`);
   fs.writeFileSync(out+'/preview_closd_forces.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
  }
 }
 const clip=await evaluate(`(()=>{const r=document.querySelector('.method-diagrams').getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1};})()`);
 fs.writeFileSync(out+'/preview_'+method+'_both_diagrams.png',Buffer.from((await call('Page.captureScreenshot',{format:'png',clip,captureBeyondViewport:true})).data,'base64'));
 checks.methodPages.push(page);
}
await evaluate(`(()=>{const s=document.getElementById('native-method-filter');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'all');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,150));
await evaluate(`document.querySelector('[data-component-id="paired-motion-player"]').scrollIntoView({block:'start'})`);
await evaluate(`document.getElementById('paired-motion-play').click()`);
await new Promise(r=>setTimeout(r,400));
checks.paired=await evaluate(`({time:Number(document.getElementById('paired-motion-time').value),skeletons:document.querySelectorAll('[data-component-id="paired-motion-player"] svg[role="img"]').length,curvePaths:document.querySelectorAll('[data-component-id="paired-motion-curve"] .recharts-line-curve').length})`);
await evaluate(`document.getElementById('paired-motion-play').click()`);
if(checks.paired.time<=0||checks.paired.skeletons!==2||checks.paired.curvePaths<2)throw Error('Paired animation/curve rendering failed');
fs.writeFileSync(out+'/preview_paired_motion.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await evaluate(`(()=>{const s=document.getElementById('paired-curve-metric');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'jerk_component_rms_m_s3');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,150));
checks.paired.jerkPaths=await evaluate(`document.querySelectorAll('[data-component-id="paired-motion-curve"] .recharts-line-curve').length`);
if(checks.paired.jerkPaths!==2)throw Error('Raw four-frame jerk curves absent from paired viewer');
await evaluate(`document.querySelector('[data-component-id="paired-motion-curve"]').scrollIntoView({block:'start'})`);
fs.writeFileSync(out+'/preview_paired_jerk.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
checks.kimodoAnimation=await evaluate(`(()=>{const s=document.getElementById('paired-case-filter');const options=[...s.options].filter(o=>o.value.startsWith('kimodo /'));if(options.length!==4)return {cases:options.length};Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,options[0].value);s.dispatchEvent(new Event('change',{bubbles:true}));return {cases:options.length};})()`);
await new Promise(r=>setTimeout(r,200));
await evaluate(`document.getElementById('paired-motion-play').click()`);
await new Promise(r=>setTimeout(r,400));
checks.kimodoAnimation.time=await evaluate(`Number(document.getElementById('paired-motion-time').value)`);
await evaluate(`document.getElementById('paired-motion-play').click()`);
if(checks.kimodoAnimation.cases!==4||checks.kimodoAnimation.time<=0)throw Error('Kimodo paired animation failed');
await evaluate(`document.querySelector('[data-component-id="paired-motion-player"]').scrollIntoView({block:'start'})`);
fs.writeFileSync(out+'/preview_kimodo_motion.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await evaluate(`(()=>{const s=document.getElementById('paired-case-filter');const choice=[...s.options].find(o=>o.value.startsWith('chois /'));if(!choice)throw Error('CHOIS pair absent');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,choice.value);s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,200));
checks.choisGeometry=await evaluate(`({points:document.querySelectorAll('.native-object-points circle').length,handLinks:document.querySelectorAll('.native-hand-surface-links line').length,firstX:document.querySelector('.native-object-points circle')?.getAttribute('cx')})`);
await evaluate(`document.getElementById('paired-motion-play').click()`);
await new Promise(r=>setTimeout(r,400));
await evaluate(`document.getElementById('paired-motion-play').click()`);
checks.choisGeometry.moved=await evaluate(`document.querySelector('.native-object-points circle')?.getAttribute('cx')!==${JSON.stringify(checks.choisGeometry.firstX)}`);
await evaluate(`(()=>{const s=document.getElementById('paired-curve-metric');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'left_hand_object_distance_m');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,200));
checks.choisGeometry.distanceCurves=await evaluate(`document.querySelectorAll('[data-component-id="paired-motion-curve"] .recharts-line-curve').length`);
if(checks.choisGeometry.points!==640||checks.choisGeometry.handLinks!==4||!checks.choisGeometry.moved||checks.choisGeometry.distanceCurves!==2)throw Error('CHOIS object motion or native distance curves failed');
await evaluate(`document.querySelector('[data-component-id="paired-motion-player"]').scrollIntoView({block:'start'})`);
fs.writeFileSync(out+'/preview_chois_object_pair.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await evaluate(`(()=>{const s=document.getElementById('paired-case-filter');const choice=[...s.options].find(o=>o.value.startsWith('chois / contact_counterexample_'));if(!choice)throw Error('Full-matrix contact counterexample absent');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,choice.value);s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,200));
await evaluate(`(()=>{const s=document.getElementById('paired-curve-metric');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'left_hand_object_distance_m');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await evaluate(`document.getElementById('paired-motion-play').click()`);
await new Promise(r=>setTimeout(r,400));
await evaluate(`document.getElementById('paired-motion-play').click()`);
checks.choisContactCounterexample=await evaluate(`({points:document.querySelectorAll('.native-object-points circle').length,curves:document.querySelectorAll('[data-component-id="paired-motion-curve"] .recharts-line-curve').length,time:Number(document.getElementById('paired-motion-time').value),disclosed:document.body.innerText.includes('事后接触反例')})`);
if(checks.choisContactCounterexample.points!==640||checks.choisContactCounterexample.curves!==2||checks.choisContactCounterexample.time<=0||!checks.choisContactCounterexample.disclosed)throw Error('Contact counterexample visualization/disclosure failed');
await evaluate(`document.querySelector('[data-component-id="paired-motion-player"]').scrollIntoView({block:'start'})`);
fs.writeFileSync(out+'/preview_chois_contact_counterexample.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await evaluate(`(()=>{const s=document.getElementById('paired-case-filter');const choice=[...s.options].find(o=>o.value.startsWith('omnicontrol / trajectory_'));if(!choice)throw Error('Native trajectory pair missing');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,choice.value);s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,200));
await evaluate(`(()=>{const s=document.getElementById('paired-curve-metric');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,'target_distance_m');s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,300));
checks.trajectoryAnimation=await evaluate(`({targets:document.querySelectorAll('.native-constraint-mark').length,points:document.querySelectorAll('[data-component-id="paired-motion-curve"] .recharts-scatter-symbol').length,conditions:document.querySelector('[data-component-id="paired-motion-player"]').innerText,disclosed:document.body.innerText.includes('九个不同时间的腕部目标')})`);
if(checks.trajectoryAnimation.targets!==18||checks.trajectoryAnimation.points!==18||!checks.trajectoryAnimation.disclosed)throw Error('Native trajectory must display nine actual constraint times per condition');
checks.trajectoryAnimation.defaultChangedComponents=await evaluate(`Number(document.querySelector('[data-paired-condition-changes]')?.getAttribute('data-paired-condition-changes'))`);
if(checks.trajectoryAnimation.defaultChangedComponents!==1)throw Error('Default trajectory comparison must change only one component');
const defaultTrajectoryRight=await evaluate(`document.querySelector('select[aria-label="paired 右 condition"]').value`);
await evaluate(`(()=>{const left=JSON.parse(document.querySelector('select[aria-label="paired 左 condition"]').value),right=document.querySelector('select[aria-label="paired 右 condition"]');const option=[...right.options].find(o=>{const c=JSON.parse(o.value);return Object.keys(left).filter(k=>left[k]!==c[k]).length===2;});Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(right,option.value);right.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,150));
checks.trajectoryAnimation.multiComponentWarning=await evaluate(`document.querySelector('[data-paired-condition-changes="2"]')?.innerText.includes('不能将差异归给单个组件')`);
if(!checks.trajectoryAnimation.multiComponentWarning)throw Error('Multi-component viewer contrast needs a visible attribution warning');
await evaluate(`(()=>{const s=document.querySelector('select[aria-label="paired 右 condition"]');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,${JSON.stringify(defaultTrajectoryRight)});s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
await new Promise(r=>setTimeout(r,150));
await evaluate(`document.getElementById('paired-motion-play').click()`);
await new Promise(r=>setTimeout(r,350));
await evaluate(`document.getElementById('paired-motion-play').click()`);
checks.trajectoryAnimation.time=await evaluate(`Number(document.getElementById('paired-motion-time').value)`);
if(checks.trajectoryAnimation.time<=0)throw Error('Trajectory playback did not advance');
await evaluate(`document.querySelector('[data-component-id="paired-motion-player"]').scrollIntoView({block:'start'})`);
fs.writeFileSync(out+'/preview_omnicontrol_trajectory.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
const diagnosticCases=await evaluate(`[...document.getElementById('paired-case-filter').options].map(o=>o.value).filter(v=>v.includes(' / diagnostic_'))`);
if(diagnosticCases.length!==7)throw Error('Seven-method diagnostic examples missing');
checks.diagnostics=[];
for(const choice of diagnosticCases){
 await evaluate(`(()=>{const s=document.getElementById('paired-case-filter');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,${JSON.stringify(choice)});s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 await new Promise(r=>setTimeout(r,150));
 await evaluate(`(()=>{const s=document.getElementById('paired-motion-time');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(s,String(Number(s.max)/2));s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 await new Promise(r=>setTimeout(r,150));
 const check=await evaluate(`({skeletons:document.querySelectorAll('[data-component-id="paired-motion-player"] svg[role="img"]').length,curves:document.querySelectorAll('[data-component-id="paired-motion-curve"] .recharts-line-curve').length,time:Number(document.getElementById('paired-motion-time').value),disclosed:document.body.innerText.includes('诊断案例，不是随机代表样本')})`);
 checks.diagnostics.push({choice,...check});
 if(check.skeletons!==2||check.curves!==2||check.time<=0||!check.disclosed)throw Error('Diagnostic case failed: '+choice);
 await evaluate(`document.querySelector('[data-component-id="paired-motion-player"]').scrollIntoView({block:'start'})`);
 fs.writeFileSync(out+'/preview_diagnostic_'+choice.split(' / ')[0]+'.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
}
await evaluate(`document.querySelector('[data-component-id="ardy-backbone"]').scrollIntoView({block:'start'});window.scrollBy(0,-65)`);
await new Promise(r=>setTimeout(r,200));
fs.writeFileSync(out+'/preview_ardy_diagram.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
const tree=await call('Page.getFrameTree');
const legacy=tree.frameTree.childFrames?.find(f=>f.frame.url==='about:srcdoc');
if(legacy){
 const world=await call('Page.createIsolatedWorld',{frameId:legacy.frame.id,worldName:'motiongen-archive-check'});
 const result=await call('Runtime.evaluate',{contextId:world.executionContextId,returnByValue:true,
  expression:`({sections:[...document.querySelectorAll('main section')].map(s=>s.id),compareOptions:document.querySelector('#compareA')?.options.length,caseDetails:document.querySelectorAll('details').length,videos:document.querySelectorAll('.play').length})`});
 checks.legacy=result.result.value;
}
if(!checks.legacy){
 const targets=await(await fetch('http://127.0.0.1:9223/json/list')).json();
 const frame=targets.find(t=>t.type==='iframe'&&t.parentId===tab.id);
 if(frame){
  const child=new WebSocket(frame.webSocketDebuggerUrl);
  await new Promise(r=>child.onopen=r);
  const result=await new Promise((resolve,reject)=>{child.onmessage=e=>{const r=JSON.parse(e.data);if(r.id===1)resolve(r)};
   child.send(JSON.stringify({id:1,method:'Runtime.evaluate',params:{returnByValue:true,expression:`({sections:[...document.querySelectorAll('main section')].map(s=>s.id),compareOptions:document.querySelector('#compareA')?.options.length,caseDetails:document.querySelectorAll('details').length,videos:document.querySelectorAll('.play').length,firstVideo:document.querySelector('.play')?.dataset.sourceVideo||document.querySelector('.play')?.dataset.video,base:document.baseURI})`}}));});
  checks.legacy=result.result?.result?.value;
  const playback=await new Promise(resolve=>{child.onmessage=e=>{const r=JSON.parse(e.data);if(r.id===2)resolve(r)};
   child.send(JSON.stringify({id:2,method:'Runtime.evaluate',params:{returnByValue:true,awaitPromise:true,expression:`(async()=>{
    const buttons=[...document.querySelectorAll('.play')];const checks=[];
    for(const button of buttons.slice(0,5)){
     const video=document.getElementById('motionVideo');video.muted=true;button.click();
     await new Promise(resolve=>{if(video.readyState>=2)return resolve();video.addEventListener('loadeddata',resolve,{once:true});setTimeout(resolve,5000)});
     await video.play().catch(()=>{});await new Promise(r=>setTimeout(r,300));
     checks.push({title:button.dataset.title,readyState:video.readyState,duration:video.duration,currentTime:video.currentTime,error:video.error?.message||null});
     video.pause();
    }
    document.querySelector('#videoModal .close').click();
    const images=await Promise.all([...document.images].map(async image=>{image.loading='eager';await image.decode().catch(()=>{});return {source:image.dataset.sourceImage,embedded:image.src.startsWith('data:image/png;base64,'),width:image.naturalWidth,height:image.naturalHeight}}));
    return {checks,images,embeddedVideos:buttons.filter(b=>b.dataset.video.startsWith('data:video/mp4;base64,')).length,urls:buttons.map(b=>new URL(b.dataset.sourceVideo||b.dataset.video,document.baseURI).href)};
   })()`}}));});
  checks.legacyPlayback=playback.result?.result?.value;
  child.close();
  if(checks.legacyPlayback){
   if(checks.legacyPlayback.embeddedVideos!==25)throw Error('All original playback controls need embedded original media');
   if(checks.legacyPlayback.images.length!==29||checks.legacyPlayback.images.some(i=>!i.embedded||i.width<=0||i.height<=0))throw Error('Original gallery or GT/raw/correction images failed to decode');
   checks.legacyVideoUrls=await Promise.all([...new Set(checks.legacyPlayback.urls)].map(async url=>({url,status:url.startsWith('file:')?(fs.existsSync(fileURLToPath(url))?200:404):(await fetch(url,{method:'HEAD'})).status})));
   if(checks.legacyPlayback.checks.some(c=>c.error||c.readyState<2||c.currentTime<=0))throw Error('Legacy video playback failed');
   if(checks.legacyVideoUrls.some(c=>c.status!==200))throw Error('Legacy video URL failed');
  }
 }
}
await evaluate('window.scrollTo(0,0)');
fs.writeFileSync(out+'/preview_desktop.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
await new Promise(r=>setTimeout(r,200));
checks.mobile=await evaluate(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})`);
fs.writeFileSync(out+'/preview_mobile.png',Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
checks.errors=errors;
checks.status='checks_recorded';
fs.writeFileSync(out+'/preview_checks.json',JSON.stringify(checks,null,2));
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
// The gallery repeats five case videos: 25 playback buttons, 20 unique files.
if(!checks.legacy||checks.legacy.compareOptions!==5||checks.legacy.videos!==25||checks.legacyVideoUrls?.length!==20)throw Error('Original report controls or 20 unique videos absent');
if(!checks.legacyPlayback||checks.legacyPlayback.checks.length!==5)throw Error('Actual legacy video playback not exercised');
if(checks.mobile.scrollWidth>checks.mobile.width)throw Error('Mobile horizontal overflow');
if(errors.length)throw Error('Browser runtime exceptions: '+JSON.stringify(errors));
if(hashFile(reportRoot+'/index.html')!==htmlBefore)throw Error('Report changed during browser checks');
if(offlinePath&&hashFile(offlinePath)!==offlineHash)throw Error('Offline export changed during browser checks');
checks.status='passed';
fs.writeFileSync(out+'/preview_checks.json',JSON.stringify(checks,null,2));
console.log(JSON.stringify(checks));socket.close();
