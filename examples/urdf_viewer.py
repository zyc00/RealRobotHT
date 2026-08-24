"""Build a self-contained HTML viewer for the PiPER-X URDF with joint sliders.

Reads the URDF the gravity model uses (piper_ht/piper_description.urdf) plus
the vendor STL meshes, and writes ONE html file with an embedded WebGL renderer,
forward kinematics, and a slider per joint (limits taken from the URDF).  No
server, no CDN: open the file in any browser, or share it as-is.

    python examples/urdf_viewer.py                       # -> data/piperx_viewer.html
    python examples/urdf_viewer.py -o /tmp/v.html --open

Meshes: the URDF references package://agx_arm_description/agx_arm_urdf/piper_x/
meshes/dae/*.dae; we load the binary STL twins from --mesh-dir (default is the
agx_arm_urdf checkout in /tmp).  Only numpy + stdlib are needed.

The FK in the page mirrors piper_ht/model.py: fixed-axis rpy (Rz Ry Rx), joint
value applied after the origin, q[i] in radians == SDK joint i.

The page also carries the hand-guidance analysis (floating panel on the stage):
the J456 reach (probe-point sweep of the wrist about J4's origin, J123 frozen),
the J123 reach ((q2,q3) sheets at several q1, wrist frozen), and at a user-set
probe point: the wrist SENSING DISC (column space of J456 -- pushes in this
plane deflect the wrist), the BLIND LINE perpendicular to it (no wrist
deflection; only J1-3 torques can sense it), the J1/J2/J3 hand-motion arrows
(columns of J123) with, per base joint, the share of its motion lying in the
wrist plane and the torque signature J123^T n of a push along the blind line.
Plus an "intent probe": pick a wrist torque / deflection sign pattern, get the
hand-force estimate under a push-not-twist prior, f = (J456_pos^T)^+ tau456,
and the torque that push applies to each base joint, tau123 = J123_pos^T f --
its sign is the direction the base friction compensation should push.  (Use
this torque mapping, not the kinematic J123^+ v: the two disagree in sign at
most poses, and stiction breakaway follows the applied joint torque.)  The wrist positional Jacobian at a hand-held point
is rank 2 (sigma3 ~ 0) at essentially every pose, even for an off-axis grasp,
which is why the sensing set is a plane and not a volume.
"""

import argparse
import base64
import json
import os
import sys
import webbrowser
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_URDF = os.path.join(REPO, "piper_ht", "piper_description.urdf")
DEFAULT_MESH_DIR = "/tmp/agx_arm_urdf/piper_x/meshes"
DEFAULT_OUT = os.path.join(REPO, "data", "piperx_viewer.html")


# --------------------------------------------------------------------------- URDF

def _floats(s, n, default):
    if s is None:
        return list(default)
    v = [float(x) for x in s.split()]
    assert len(v) == n, s
    return v


def parse_urdf(path):
    root = ET.parse(path).getroot()
    links, joints = {}, []
    for el in root.findall("link"):
        vis = el.find("visual")
        mesh = None
        if vis is not None and vis.find("geometry/mesh") is not None:
            o = vis.find("origin")
            mesh = {
                "file": vis.find("geometry/mesh").get("filename"),
                "xyz": _floats(o.get("xyz") if o is not None else None, 3, (0, 0, 0)),
                "rpy": _floats(o.get("rpy") if o is not None else None, 3, (0, 0, 0)),
            }
        links[el.get("name")] = {"name": el.get("name"), "mesh": mesh}
    for el in root.findall("joint"):
        o, a, lim, mim = el.find("origin"), el.find("axis"), el.find("limit"), el.find("mimic")
        joints.append({
            "name": el.get("name"),
            "type": el.get("type"),
            "parent": el.find("parent").get("link"),
            "child": el.find("child").get("link"),
            "xyz": _floats(o.get("xyz") if o is not None else None, 3, (0, 0, 0)),
            "rpy": _floats(o.get("rpy") if o is not None else None, 3, (0, 0, 0)),
            "axis": _floats(a.get("xyz") if a is not None else None, 3, (0, 0, 1)),
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            "mimic": None if mim is None else {
                "joint": mim.get("joint"),
                "multiplier": float(mim.get("multiplier", 1.0)),
                "offset": float(mim.get("offset", 0.0)),
            },
        })
    # Parent-before-child order so the page can walk the tree in one pass.
    children = {j["child"] for j in joints}
    roots = [n for n in links if n not in children]
    assert len(roots) == 1, roots
    ordered, frontier = [], [roots[0]]
    while frontier:
        p = frontier.pop(0)
        for j in joints:
            if j["parent"] == p:
                ordered.append(j)
                frontier.append(j["child"])
    assert len(ordered) == len(joints)
    return root.get("name"), links, ordered


# --------------------------------------------------------------------------- meshes

def load_binary_stl(path):
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        data = np.frombuffer(f.read(n * rec.itemsize), dtype=rec)
    assert len(data) == n, f"{path}: truncated STL"
    return np.ascontiguousarray(data["v"]).astype(np.float64)  # (n, 3, 3)


def weld_and_shade(tri, crease_deg=35.0):
    """Index the triangle soup and compute vertex normals with a crease angle.

    Corners at the same position share a vertex (and get a smooth, area-
    weighted normal) unless their face normal deviates from the position's
    mean normal by more than crease_deg, in which case they keep a hard edge.
    """
    n = len(tri)
    corners = tri.reshape(-1, 3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
    fn_unit = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    corner_fn = np.repeat(fn, 3, axis=0)
    corner_fn_unit = np.repeat(fn_unit, 3, axis=0)

    key = np.round(corners, 6)
    _, pid, = np.unique(key, axis=0, return_inverse=True)[:2]
    pid = pid.ravel()
    mean = np.zeros((pid.max() + 1, 3))
    np.add.at(mean, pid, corner_fn)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)

    hard = (corner_fn_unit * mean[pid]).sum(1) < np.cos(np.radians(crease_deg))
    # Hard corners are split by a coarse quantisation of their face normal.
    sub = np.zeros(len(pid), dtype=np.int64)
    sub[hard] = (np.round(corner_fn_unit[hard] * 4).astype(np.int64) + 5) @ np.array([1, 11, 121]) + 1
    combo = np.stack([pid, sub], axis=1)
    _, vid = np.unique(combo, axis=0, return_inverse=True)[:2]
    vid = vid.ravel()
    nv = vid.max() + 1
    pos = np.zeros((nv, 3))
    pos[vid] = corners
    nrm = np.zeros((nv, 3))
    np.add.at(nrm, vid, corner_fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pos, nrm, vid.reshape(n, 3)


def pack_mesh(path):
    tri = load_binary_stl(path)
    pos, nrm, idx = weld_and_shade(tri)
    lo, hi = pos.min(0), pos.max(0)
    scale = np.maximum(hi - lo, 1e-9) / 65535.0
    q = np.clip(np.round((pos - lo) / scale), 0, 65535).astype("<u2")
    n8 = np.clip(np.round(nrm * 127), -127, 127).astype("i1")
    if len(pos) < 65536:
        ib, itype = idx.astype("<u2"), "u16"
    else:
        ib, itype = idx.astype("<u4"), "u32"
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")
    return {
        "min": lo.tolist(), "scale": scale.tolist(),
        "nverts": int(len(pos)), "ntris": int(len(idx)), "itype": itype,
        "pos": b64(q), "nrm": b64(n8), "idx": b64(ib),
    }


def resolve_mesh(filename, mesh_dir):
    base = os.path.splitext(os.path.basename(filename))[0] + ".stl"
    p = os.path.join(mesh_dir, base)
    if not os.path.exists(p):
        raise FileNotFoundError(f"mesh {base} not found in {mesh_dir} (from {filename})")
    return p


# --------------------------------------------------------------------------- page

def build_html(robot_name, links, joints, meshes, urdf_path, template):
    ntris = sum(m["ntris"] for m in meshes.values())
    robot = {
        "name": robot_name,
        "urdf": os.path.relpath(urdf_path, REPO),
        "ntris": ntris,
        "links": [links[n] for n in links],
        "joints": joints,
        "meshes": meshes,
    }
    return template.replace("__ROBOT_JSON__", json.dumps(robot, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--mesh-dir", default=DEFAULT_MESH_DIR)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    ap.add_argument("--open", action="store_true", help="open the result in a browser")
    args = ap.parse_args()

    name, links, joints = parse_urdf(args.urdf)
    meshes = {}
    for lname, link in links.items():
        if link["mesh"] is None:
            continue
        p = resolve_mesh(link["mesh"]["file"], args.mesh_dir)
        meshes[lname] = pack_mesh(p)
        m = meshes[lname]
        print(f"{lname:14s} {m['ntris']:6d} tris {m['nverts']:6d} verts  ({os.path.basename(p)})")

    template_path = os.path.join(HERE, "urdf_viewer_template.html")
    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    html = build_html(name, links, joints, meshes, args.urdf, template)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html)/1e6:.1f} MB, {sum(m['ntris'] for m in meshes.values())} triangles)")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    sys.exit(main())
