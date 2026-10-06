"""Offline constraint-overlay video; skeleton proxies are explicitly labelled."""
import numpy as np


def render(motion,scene,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,FFMpegWriter
    xyz=motion['joints']; names=list(motion['joint_names']); fps=float(motion['fps'])
    index={name:i for i,name in enumerate(names)}
    chains=[['pelvis','neck','head'],['pelvis','left_hip','left_knee','left_ankle','left_foot'],
            ['pelvis','right_hip','right_knee','right_ankle','right_foot'],
            ['neck','left_shoulder','left_elbow','left_wrist'],['neck','right_shoulder','right_elbow','right_wrist']]
    fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111,projection='3d')
    targets=[c['position_m'] for c in scene.get('constraints',[]) if 'position_m' in c]
    all_points=np.concatenate([xyz.reshape(-1,3),np.asarray(targets).reshape(-1,3)],axis=0) if targets else xyz.reshape(-1,3)
    lower,upper=all_points.min(0)-0.2,all_points.max(0)+0.2
    def frame(i):
        ax.clear(); ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),zlim=(min(0,lower[2]),upper[2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
        ax.scatter(*xyz[i].T,s=7)
        for chain in chains:
            ids=[index[x] for x in chain if x in index]
            if len(ids)>1:
                ax.plot(*xyz[i,ids].T)
        for c in scene.get('constraints',[]):
            if 'position_m' in c:
                ax.scatter(*c['position_m'],color='red',marker='x',s=80)
                ax.text(*c['position_m'],f" {c['joint']} @ {c['time_seconds']:.2f}s",fontsize=7)
            if 'rotation_matrix' in c and c['joint'] in index:
                origin=np.asarray(c.get('position_m',xyz[min(round(c['time_seconds']*fps),len(xyz)-1),index[c['joint']]]))
                rotation=np.asarray(c['rotation_matrix'])
                for axis,color in enumerate(('red','green','blue')):
                    tip=origin+0.15*rotation[:,axis]
                    ax.plot(*np.stack([origin,tip]).T,color=color,linestyle='--')
        ax.set_title(f"{scene['name']} | {i/fps:.2f}s\nSkeleton motion; posture proxies",fontsize=9)
    animation=FuncAnimation(fig,frame,frames=len(xyz),interval=1000/fps)
    animation.save(str(path),writer=FFMpegWriter(fps=fps,codec='libx264'))
    frame(len(xyz)//2); fig.savefig(str(path.with_suffix('.png')),dpi=130)
    plt.close(fig)
