"""Versioned runtime-only conditions and an independent kinematic consumer.

The producer is launched by the orchestrator. The generator only receives a
socket address; it has no future condition schedule, text or embedding cache.
All participants use the same host monotonic_ns clock.
"""
import collections,json,multiprocessing as mp,queue,socket,threading,time
from pathlib import Path
import numpy as np

def stamp():return time.monotonic_ns()

class EventReceiver:
    def __init__(self,address,log):
        self.address=str(address);self.log=Path(log);self.pending=queue.Queue();self.closed=threading.Event()
        self.server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.server.bind(self.address);self.server.listen(1);self.server.settimeout(.2)
        self.thread=threading.Thread(target=self._receive,daemon=True);self.thread.start()
    def _record(self,value):
        with self.log.open('a') as f:f.write(json.dumps(value)+'\n')
    def _receive(self):
        while not self.closed.is_set():
            try:conn,_=self.server.accept()
            except socket.timeout:continue
            except OSError:return
            with conn:
                data=b''
                while b'\n' not in data:
                    part=conn.recv(65536)
                    if not part:break
                    data+=part
                try:
                    event=json.loads(data);event['arrival_ns']=stamp()
                    self._record(dict(event,event='arrival'));self.pending.put(event)
                except Exception as exc:self._record({'event':'invalid_payload','reason':str(exc),'time_ns':stamp()})
    def poll(self):
        events=[]
        while True:
            try:event=self.pending.get_nowait()
            except queue.Empty:break
            event['read_ns']=stamp();self._record(dict(event,event='read'));events.append(event)
        return events
    def close(self):
        self.closed.set();self.server.close();self.thread.join(1);Path(self.address).unlink(missing_ok=True)

def _consume(incoming,out,fps,duration,realtime,initial_pose,ready):
    buffer=collections.deque();last=np.asarray(initial_pose);version=0;rows=[];poses=[]
    start=stamp();ready.put(start);done=False;frame=0
    while frame<round(duration*fps):
        if realtime:
            remaining=(start+round(frame/fps*1e9)-stamp())/1e9
            if remaining>0:time.sleep(remaining)
        elif not buffer and not done:
            item=incoming.get()
            if item is None:done=True
            else:
                for pose in item['poses']:buffer.append((pose,item['version'],item['commit_ns'],item['chunk']))
        while True:
            try:item=incoming.get_nowait()
            except queue.Empty:break
            if item is None:done=True;break
            for pose in item['poses']:buffer.append((pose,item['version'],item['commit_ns'],item['chunk']))
        consume=stamp();held=not bool(buffer)
        if buffer:last,version,commit,chunk=buffer.popleft()
        else:commit=None;chunk=None
        rows.append({'frame':frame,'consume_ns':consume,'condition_version':version,'held':held,
                     'commit_ns':commit,'chunk':chunk,'buffer_frames':len(buffer),
                     'deadline_lateness_seconds':max(0.,(consume-start)/1e9-frame/fps) if realtime else None})
        poses.append(np.asarray(last).copy());frame+=1
    path=Path(out);np.savez_compressed(path/'consumed_motion.npz',joints=poses,fps=fps)
    (path/'consumption.json').write_text(json.dumps({'clock':'kinematic_playback' if realtime else 'paused_diagnostic',
        'start_ns':start,'rows':rows,'scope':'Independent consumer; missing supply holds last committed pose, with explicit held frames. No interpolation or physical execution claim.'}))

class Playback:
    def __init__(self,out,fps,duration,realtime,initial_pose):
        ctx=mp.get_context('spawn');self.queue=ctx.Queue();ready=ctx.Queue()
        self.process=ctx.Process(target=_consume,args=(self.queue,str(out),fps,duration,realtime,initial_pose,ready))
        self.process.start();self.start_ns=ready.get(timeout=15)
    def submit(self,poses,version,chunk):
        t=stamp();self.queue.put({'poses':np.asarray(poses),'version':version,'chunk':chunk,'commit_ns':t});return t
    def finish(self,timeout=30):
        self.queue.put(None);self.process.join(timeout)
        if self.process.is_alive():self.process.terminate();self.process.join();raise TimeoutError('Playback did not finish')
        if self.process.exitcode:raise RuntimeError(f'Playback exit {self.process.exitcode}')
