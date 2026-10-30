"""Independent event producer. Never imported by the generating worker."""
import argparse,json,socket,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--ready',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
deadline=time.monotonic()+1200
while not a.ready.exists():
    if time.monotonic()>deadline:raise TimeoutError('Generator never became ready')
    time.sleep(.05)
start=json.loads(a.ready.read_text())['start_ns'];rows=[]
for event in json.loads(a.events.read_text())['events']:
    due=start+round(event['after_seconds']*1e9)
    while time.monotonic_ns()<due:time.sleep(min(.05,(due-time.monotonic_ns())/1e9))
    payload={'version':event['version'],'payload':event['payload'],'due_ns':due,'send_ns':time.monotonic_ns()}
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        s.connect(a.socket);s.sendall((json.dumps(payload)+'\n').encode())
    payload['send_complete_ns']=time.monotonic_ns();rows.append(payload)
    a.output.write_text(json.dumps({'events':rows,'clock':'host monotonic_ns','producer_pid':__import__('os').getpid()}))
