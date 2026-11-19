(()=>{
 if(!/^https?:$/.test(location.protocol))return;
 const status=document.getElementById('live-status');let busy=false;
 async function refresh(){
  if(busy)return;busy=true;
  try{
   const response=await fetch('/api/k4/ablation_status',{cache:'no-store'});
   if(!response.ok)throw new Error('HTTP '+response.status);
   const data=await response.json();
   document.getElementById('k4-ablation').outerHTML=data.panel;
   document.getElementById('replay-live').innerHTML=data.replay;
   const age=data.updated_unix_s?Math.max(0,Math.round(Date.now()/1000-data.updated_unix_s)):null;
   status.textContent='接口刷新：'+new Date().toLocaleTimeString()+'；消融账本距今 '+(age===null?'未知':age+' 秒')+'。'+(age>120?'账本暂无新增，接口仍正常。':'');
  }catch(e){status.textContent='实时接口暂不可用，保留上次数据：'+e.message;}
  finally{busy=false;}
 }
 document.getElementById('reload-details').addEventListener('click',()=>location.reload());
 refresh();setInterval(refresh,15000);
})();
