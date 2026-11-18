/* Presentation only. Frozen result files are read, never changed. */
(() => {
  const headline = document.querySelector('h1');
  const summary = headline.nextElementSibling;
  const formatTable = document.querySelector('table');
  const bar = document.createElement('aside');
  bar.style.cssText = 'padding:10px 14px;background:#e8f3ef;border-radius:6px;margin:14px 0';
  bar.innerHTML = '<label><input id="k4-auto" type="checkbox" checked>自动更新（15 秒）</label> <button id="k4-refresh">立即更新</button> <span id="k4-live-state"></span>';
  summary.after(bar);
  const status = bar.querySelector('#k4-live-state');
  const phase = document.createElement('p');
  phase.id='k4-phase';phase.style.cssText='margin:8px 0 0;font-size:14px';bar.append(phase);
  if (!/^https?:$/.test(location.protocol)) {
    status.textContent = '离线快照：显示文件中保存的结果';
    bar.querySelector('#k4-auto').disabled = true;
    bar.querySelector('#k4-refresh').disabled = true;
    return;
  }
  const openKeys = new Set();
  function tagDetails() {
    document.querySelectorAll('#rows > details').forEach(root => {
      const key = root.querySelector('summary').textContent.split(' · ').slice(0,2).join('_');
      root.dataset.liveKey = key;
      root.querySelectorAll('details').forEach((node,index) => {node.dataset.liveKey = key+'|'+index;});
    });
  }
  const baseRender = render;
  render = function() {
    tagDetails();
    document.querySelectorAll('#rows details').forEach(node => {
      if (node.open) openKeys.add(node.dataset.liveKey); else openKeys.delete(node.dataset.liveKey);
    });
    const x=scrollX,y=scrollY;
    baseRender();tagDetails();
    document.querySelectorAll('#rows details').forEach(node => {node.open = openKeys.has(node.dataset.liveKey);});
    scrollTo(x,y);
  };
  tagDetails();
  let busy=false;
  async function get(url) {
    const response=await fetch(url,{cache:'no-store'});
    if (!response.ok) throw Error('HTTP '+response.status);
    return response.json();
  }
  async function refresh() {
    if (busy) return;
    busy=true;
    try {
      const data=await get('/api/k4/progress');
      const index=new Map(rows.map(row=>[row.window+'_'+row.method,row]));
      let changed=false;
      for (const meta of data.rows) {
        const key=meta.window+'_'+meta.method, old=index.get(key);
        if (!old) throw Error('Unknown registered path');
        if (meta.status!==old.status || meta.passed!==old.passed || (meta.status==='completed'&&!old.metrics)) {
          const detail=meta.status==='completed'?await get('/api/k4/record?key='+encodeURIComponent(key)):meta;
          Object.assign(old,detail);changed=true;
        }
      }
      summary.textContent=`已评分 ${data.evaluated}/${data.expected}；通过 ${data.passed}/${data.expected}；执行失败 ${data.execution_failed}；待完成 ${data.pending}（其中计算中 ${data.running}）。`;
      formatTable.innerHTML='<tr><th>格式</th><th>通过／总数</th><th>已评分</th><th>执行失败</th><th>待完成</th></tr>'+Object.entries(data.by_format).sort().map(([method,c])=>`<tr><td>${esc(method)}</td><td>${c.passed}/${c.total}</td><td>${c.evaluated}</td><td>${c.execution_failed}</td><td>${c.pending}</td></tr>`).join('');
      if (changed) render();
      status.textContent=`已同步 ${new Date(data.checked_at).toLocaleTimeString()} · worker 上限 ${data.worker_limit}`;
      const states={waiting_for_registered_runs:'等待前序实验及独立验收',running:'执行中',validation_stage_failed:'本阶段未通过，后续保留待执行',completed:'本版任务执行结束'};
      const messages=[];
      if (data.recorded_state) messages.push('调度记录：'+(states[data.recorded_state]||data.recorded_state));
      if (data.stage) messages.push('当前阶段：'+data.stage);
      if (data.waiting_on?.length) messages.push('等待：'+data.waiting_on.join('、'));
      messages.push(data.latest_result_at?'最近新增结果：'+new Date(data.latest_result_at).toLocaleString():'尚无已完成结果');
      phase.textContent=messages.join('；');
      if (!data.pending) {clearInterval(timer);status.textContent+=' · 全部路径已结束';}
    } catch (error) {
      status.textContent='更新暂不可用；保留上次结果，可稍后重试';
    } finally {busy=false;}
  }
  bar.querySelector('#k4-refresh').onclick=refresh;
  const timer=setInterval(()=>{if(bar.querySelector('#k4-auto').checked)refresh();},15000);
  refresh();
})();
