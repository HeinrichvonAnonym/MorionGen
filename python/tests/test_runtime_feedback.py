import json,socket,tempfile,time,unittest
from pathlib import Path
import numpy as np
from motiogen.runtime_feedback import EventReceiver,Playback,stamp
from motiogen.posture_geometry import flexion,episodes

class RuntimeFeedbackTests(unittest.TestCase):
    def test_receiver_only_obtains_revealed_payload(self):
        with tempfile.TemporaryDirectory(prefix='mg-') as directory:
            path=Path(directory);receiver=EventReceiver(path/'e.sock',path/'events.jsonl')
            self.assertEqual(receiver.poll(),[])
            sent=stamp()
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
                client.connect(str(path/'e.sock'));client.sendall(json.dumps({'version':1,'payload':{'prompt':'new text'},'send_ns':sent}).encode()+b'\n')
            deadline=time.monotonic()+2;events=[]
            while not events and time.monotonic()<deadline:events=receiver.poll();time.sleep(.005)
            receiver.close();self.assertEqual(events[0]['payload']['prompt'],'new text')
            self.assertLessEqual(sent,events[0]['arrival_ns']);self.assertLessEqual(events[0]['arrival_ns'],events[0]['read_ns'])

    def test_paused_consumer_preserves_committed_pose_and_version(self):
        with tempfile.TemporaryDirectory(prefix='mg-') as directory:
            playback=Playback(directory,10,.4,False,np.zeros((2,3)))
            poses=np.arange(24).reshape(4,2,3);playback.submit(poses[:2],0,0);playback.submit(poses[2:],1,1);playback.finish()
            np.testing.assert_array_equal(np.load(Path(directory)/'consumed_motion.npz')['joints'],poses)
            rows=json.loads((Path(directory)/'consumption.json').read_text())['rows']
            self.assertEqual([r['condition_version'] for r in rows],[0,0,1,1]);self.assertFalse(any(r['held'] for r in rows))

    def test_geometry_rigid_transform_and_mirror(self):
        a=np.array([[0.,1,0]]);b=np.array([[0.,0,0]]);c=np.array([[1.,0,0]])
        self.assertAlmostEqual(flexion(a,b,c)[0],90)
        for matrix in [np.diag([-1,1,1]),np.array([[0,-1,0],[1,0,0],[0,0,1]])]:
            np.testing.assert_allclose(flexion(a@matrix+3,b@matrix+3,c@matrix+3),flexion(a,b,c))
        self.assertEqual(episodes([True,True,False,True],2),[{'start_seconds':0.,'end_seconds':1.,'duration_seconds':1.},{'start_seconds':1.5,'end_seconds':2.,'duration_seconds':.5}])

if __name__=='__main__':unittest.main()
