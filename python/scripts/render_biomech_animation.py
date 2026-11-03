"""Render side-by-side OpenSim musculoskeletal animations of evaluated runs.

Bones are the model meshes posed by the evaluated coordinates, muscles are their
geometry paths coloured by static-optimization activation, green arrows are the
foot ground reaction forces and the magenta arrow is the residual force at the
centre of mass. Black dots are the joint-centre markers the IK tracked.
"""
import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import opensim as osim

GEOMETRY = Path.home()/'.local/share/opensimcreator/resources/OpenSimCreator/geometry'
GRAVITY = 9.81
# Display conventions (约定，未经推演).
BW_LENGTH = .6
ARROW_MAX = 2.2
MUSCLE_WIDTH = .009
CLUSTER = .006
AZIMUTH, ELEVATION = 35., 12.
HALF_WIDTH, FLOOR, TOP = 1.3, -.25, 2.1
BONE = np.array([.93, .90, .82])
IDLE, ACTIVE = np.array([.33, .37, .62]), np.array([.86, .08, .08])
LIGHT = np.array([.4, .8, .45])/np.linalg.norm([.4, .8, .45])

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def table(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == 'endheader')+1
    names = lines[start].split('\t')
    rows = [[float(v) for v in line.split('\t')[:len(names)]] for line in lines[start+1:] if line.strip()]
    return names, np.array(rows)


def trc(path):
    lines = Path(path).read_text().splitlines()
    names = [n for n in lines[3].split('\t')[2:] if n]
    rows = np.array([[float(v) for v in line.split('\t')[:2+3*len(names)]] for line in lines[5:] if line.strip()])
    return rows[:, 1], rows[:, 2:].reshape(len(rows), len(names), 3)


def vtp(name, scale):
    piece = ET.parse(GEOMETRY/name).getroot().find('PolyData/Piece')
    points = np.array(piece.find('Points/DataArray').text.split(), float).reshape(-1, 3)*scale
    polys = piece.find('Polys')
    connectivity = np.array(polys.find("DataArray[@Name='connectivity']").text.split(), int)
    offsets = np.array(polys.find("DataArray[@Name='offsets']").text.split(), int)
    triangles, start = [], 0
    for end in offsets:
        face, start = connectivity[start:end], end
        triangles += [(face[0], face[k], face[k+1]) for k in range(1, len(face)-1)]
    return decimate(points, np.array(triangles))


def decimate(points, triangles):
    """Vertex clustering on a CLUSTER grid; drops degenerate and duplicate faces."""
    _, cell, inverse = np.unique(np.floor(points/CLUSTER).astype(int), axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    merged = np.zeros((len(cell), 3))
    np.add.at(merged, inverse, points)
    merged /= np.bincount(inverse)[:, None]
    faces = inverse[triangles]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    _, keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return merged, faces[np.sort(keep)]


def matrix(rotation):
    return np.array([[rotation.get(a, b) for b in range(3)] for a in range(3)])


def vector(vec):
    return np.array([vec.get(i) for i in range(3)])


class Panel:
    def __init__(self, label, work, metrics):
        work = Path(work)
        self.label, self.metrics = label, json.loads(Path(metrics).read_text())
        osim.Logger.setLevelString('error')
        self.model = osim.Model(str(work/'model.osim'))
        self.state = self.model.initSystem()
        self.weight = self.model.getTotalMass(self.state)*GRAVITY
        names, data = table(work/'coordinates.mot')
        degrees = 'inDegrees=yes' in (work/'coordinates.mot').read_text()[:400]
        coordinates = self.model.getCoordinateSet()
        self.time, self.coordinates = data[:, 0], []
        for i, name in enumerate(names[1:], 1):
            if coordinates.contains(name):
                coordinate = coordinates.get(name)
                values = np.deg2rad(data[:, i]) if degrees and coordinate.getMotionType() == 1 else data[:, i]
                self.coordinates.append((coordinate, values))
        tx, tz = data[:, names.index('pelvis_tx')], data[:, names.index('pelvis_tz')]
        kernel = np.ones(101)/101
        self.centre = np.stack([np.convolve(np.pad(v, 50, mode='edge'), kernel, 'valid') for v in (tx, tz)], 1)
        self.path = np.stack([tx, np.zeros_like(tx), tz], 1)
        self.meshes = []
        for component in self.model.getComponentsList():
            if component.getConcreteClassName() == 'Mesh':
                mesh = osim.Mesh.safeDownCast(component)
                points, triangles = vtp(mesh.get_mesh_file(), vector(mesh.get_scale_factors()))
                self.meshes.append((mesh.getFrame(), points, triangles))
        self.muscles = [self.model.getMuscles().get(i) for i in range(self.model.getMuscles().getSize())]
        names, data = table(work/'so/so_so_activation.sto')
        self.activation_time = data[:, 0]
        self.activation = np.stack([data[:, names.index(m.getName())] for m in self.muscles], 1)
        _, grf = table(work/'grf.mot')
        self.grf = grf
        dynamics = np.load(work/'dynamics.npz')
        self.dynamics_time, self.com = dynamics['times'], dynamics['com']
        self.residual = dynamics['residual_force']
        self.marker_time, self.markers = trc(work/'markers.trc')

    def pose(self, t):
        i = int(np.clip(np.searchsorted(self.time, t), 0, len(self.time)-1))
        for coordinate, values in self.coordinates:
            coordinate.setValue(self.state, values[i], False)
        self.model.realizePosition(self.state)
        triangles = []
        for frame, points, faces in self.meshes:
            transform = frame.getTransformInGround(self.state)
            world = points@matrix(transform.R()).T+vector(transform.p())
            triangles.append(world[faces])
        activation = np.array([np.interp(t, self.activation_time, a) for a in self.activation.T])
        paths = []
        for muscle in self.muscles:
            current = muscle.getGeometryPath().getCurrentPath(self.state)
            paths.append(np.array([vector(current.get(j).getLocationInGround(self.state)) for j in range(current.getSize())]))
        g = int(np.clip(np.searchsorted(self.grf[:, 0], t), 0, len(self.grf)-1))
        forces = [(self.grf[g, 4:7], self.grf[g, 1:4]), (self.grf[g, 13:16], self.grf[g, 10:13])]
        d = int(np.clip(np.searchsorted(self.dynamics_time, t), 0, len(self.dynamics_time)-1))
        markers = np.stack([np.interp(t, self.marker_time, self.markers[:, m, k])
                            for m in range(self.markers.shape[1]) for k in range(3)]).reshape(-1, 3)
        return dict(index=i, triangles=np.concatenate(triangles), activation=activation, paths=paths,
                    forces=forces, com=self.com[d], residual=self.residual[d], markers=markers)


def camera():
    a, e = np.deg2rad(AZIMUTH), np.deg2rad(ELEVATION)
    view = np.array([np.cos(e)*np.cos(a), np.sin(e), np.cos(e)*np.sin(a)])
    right = np.cross([0., 1., 0.], view)
    right /= np.linalg.norm(right)
    return right, np.cross(view, right), view


RIGHT, UP, VIEW = camera()


def project(points, centre):
    shifted = points-np.array([centre[0], 0., centre[1]])
    return np.stack([shifted@RIGHT, shifted@UP], -1), shifted@VIEW


def arrow(ax, start, vector, centre, colour, width):
    length = np.linalg.norm(vector)
    if length < .01:
        return
    vector = vector*min(1., ARROW_MAX/length)
    (a, b), _ = project(np.stack([start, start+vector]), centre)
    ax.annotate('', xy=b, xytext=a, zorder=5,
                arrowprops=dict(arrowstyle='-|>', color=colour, lw=width, mutation_scale=16, shrinkA=0, shrinkB=0))


def draw(ax, panel, pose):
    centre = panel.centre[pose['index']]
    ax.cla()
    ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
    ax.set_ylim(FLOOR, TOP)
    ax.set_aspect('equal')
    ax.axis('off')
    grid = np.arange(-4., 4.01, .5)
    cx, cz = np.round(centre/.5)*.5
    lines = [np.array([[cx+x, 0, cz+grid[0]], [cx+x, 0, cz+grid[-1]]]) for x in grid]
    lines += [np.array([[cx+grid[0], 0, cz+z], [cx+grid[-1], 0, cz+z]]) for z in grid]
    lines = [project(line, centre)[0] for line in lines]
    ax.add_collection(LineCollection(lines, colors='#d4d4d4', linewidths=.6, zorder=0))
    trail, _ = project(panel.path, centre)
    ax.plot(trail[:, 0], trail[:, 1], color='#9db7d6', lw=1., zorder=1)
    triangles = pose['triangles']
    flat, depth = project(triangles.reshape(-1, 3), centre)
    flat, depth = flat.reshape(-1, 3, 2), depth.reshape(-1, 3).mean(1)
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)+1e-12
    shade = .35+.65*np.abs(normal@LIGHT)
    colours = [np.clip(BONE*shade[:, None], 0, 1)]
    polygons = [flat]
    depths = [depth]
    for path, a in zip(pose['paths'], pose['activation']):
        if len(path) < 2:
            continue
        ends, end_depth = project(path, centre)
        direction = np.diff(ends, axis=0)
        normal2 = np.stack([-direction[:, 1], direction[:, 0]], 1)
        normal2 *= MUSCLE_WIDTH/2/(np.linalg.norm(normal2, axis=1, keepdims=True)+1e-12)
        quads = np.stack([ends[:-1]+normal2, ends[1:]+normal2, ends[1:]-normal2, ends[:-1]-normal2], 1)
        a = float(np.clip(a, 0, 1))
        polygons.append(quads)
        depths.append((end_depth[:-1]+end_depth[1:])/2+.02)
        colours.append(np.tile(IDLE+(ACTIVE-IDLE)*a, (len(quads), 1)))
    order = np.argsort(np.concatenate(depths))
    flat_polygons = [p for group in polygons for p in group]
    colours = np.concatenate(colours)
    ax.add_collection(PolyCollection([flat_polygons[i] for i in order], facecolors=colours[order],
                                     edgecolors='none', antialiaseds=False, zorder=2))
    markers, _ = project(pose['markers'], centre)
    ax.scatter(markers[:, 0], markers[:, 1], s=9, c='k', edgecolors='white', linewidths=.4, zorder=4)
    for point, force in pose['forces']:
        arrow(ax, point, force/panel.weight*BW_LENGTH, centre, '#1f9e3a', 2.)
    residual = pose['residual']/panel.weight
    arrow(ax, pose['com'], residual*BW_LENGTH, centre, '#d81b8c', 3.)
    m = panel.metrics
    ax.set_title(f"{panel.label}\n残余力 RMS {m['metrics']['residual_force_rms_bw']:.3f} BW · IK 均方根 "
                 f"{m['ik']['rms_mean_m']*100:.1f} cm / 最大 {m['ik']['max_m']*100:.1f} cm · "
                 f"{'有效' if m['valid'] else '无效'}", fontsize=12)
    ax.text(.02, .02, f"当前残余力 {np.linalg.norm(residual):.2f} BW", transform=ax.transAxes, fontsize=11,
            color='#d81b8c', zorder=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', nargs=3, action='append', metavar=('LABEL', 'WORK', 'METRICS'), required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--frames', type=int)
    args = parser.parse_args()
    panels = [Panel(*p) for p in args.panel]
    start = max(p.activation_time[0] for p in panels)
    end = min(p.activation_time[-1] for p in panels)
    times = np.arange(start, end, 1/args.fps)[:args.frames]
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.suptitle(args.title, fontsize=14, y=.985)
    axes = [fig.add_axes([.01+i*.495, .27, .48, .66]) for i in range(len(panels))]
    strip = fig.add_axes([.06, .06, .9, .16])
    colours = ['#d81b8c', '#3a6fb0']
    series = [(p.dynamics_time, np.linalg.norm(p.residual, axis=1)/p.weight) for p in panels]
    window = (series[0][0] >= start) & (series[0][0] <= end)
    still = series[0][0][window][np.argmax(series[0][1][window])]
    fig.text(.5, .235, '骨骼：模型网格；肌肉：静态优化激活（蓝 0 → 红 1）；绿箭头：足底地面反力；'
             '品红箭头：残余力（地面反力解释不了的部分，作用于质心）；黑点：IK 跟踪的关节中心；1 BW = 0.6 m 箭头',
             ha='center', fontsize=9.5, color='#444')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    width, height = fig.canvas.get_width_height()
    encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                '-s', f'{width}x{height}', '-r', str(args.fps), '-i', '-', '-c:v', 'libx264',
                                '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
                                '-movflags', '+faststart', str(args.out)], stdin=subprocess.PIPE)
    for k, t in enumerate(times):
        for ax, panel in zip(axes, panels):
            draw(ax, panel, panel.pose(t))
        strip.cla()
        for (time, value), panel, colour in zip(series, panels, colours):
            keep = (time >= start) & (time <= end)
            strip.plot(time[keep], value[keep], color=colour, lw=1., label=panel.label)
        strip.axvline(t, color='k', lw=.8)
        strip.set_xlim(start, end)
        strip.set_ylim(0, None)
        strip.set_xlabel('时间 (s)')
        strip.set_ylabel('残余力 (BW)')
        strip.legend(loc='upper right', fontsize=9, frameon=False)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
        if abs(t-still) < .5/args.fps or (k == len(times)-1 and not args.out.with_suffix('.png').exists()):
            fig.savefig(args.out.with_suffix('.png'), dpi=100)
        if k % 50 == 0:
            print(f'{args.out.name} frame {k}/{len(times)} t={t:.2f}', flush=True)
    encoder.stdin.close()
    if encoder.wait():
        raise SystemExit(f'ffmpeg failed for {args.out}')
    print(f'done {args.out}', flush=True)


if __name__ == '__main__':
    main()
