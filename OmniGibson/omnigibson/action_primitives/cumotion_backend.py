"""
cuMotion 1.1.0 torch-free backend for the cap-x BEHAVIOR/OmniGibson motion path.

This module is the *core* of the cuMotion swap-in for the StanfordVL cuRobo fork.
It is deliberately PyTorch-free (numpy in / numpy out) so it can run natively on
aarch64 GB10 / sm_121 under CUDA 13 without the cuRobo-custom-op / Isaac-bundled-
torch-cu128 illegal-memory-access crash class.  The torch-facing,
CuRoboMotionGenerator-compatible shim lives in ``cumotion_motion_generator.py``
and delegates to this backend.

Key design facts (verified on this box, cuMotion 1.1.0 cu13 aarch64 cp311):
  * cuMotion's Python world model exposes ONLY primitive obstacle types
    ``Obstacle.Type.{SPHERE, CUBOID, CAPSULE}``.  There is **no mesh type and no
    SDF type** in the 1.1.0 standalone wheel (the docs mention SDF but the wheel
    does not expose ``Obstacle.Type.SDF``).  cap-x's BEHAVIOR world is mesh-heavy
    (``curobo ... WorldConfig(mesh=...)``), so meshes must be *bridged* to
    primitives.  See ``mesh_to_cuboid`` / ``MeshWorldBridge`` below.
  * Robot description is URDF + XRDF, not cuRobo YAML.  ``xrdf_from_curobo_yaml``
    converts a cap-x cuRobo robot config (collision spheres + cspace) into an XRDF.
  * The API is decomposed: kinematics / collision-free IK / motion planner are
    separate objects rather than a single cuRobo ``MotionGen``.
"""
from __future__ import annotations

import math
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import cumotion as cm


# --------------------------------------------------------------------------- #
# Robot-description bridge: cap-x cuRobo YAML -> cuMotion XRDF                  #
# --------------------------------------------------------------------------- #
def xrdf_from_curobo_yaml(
    curobo_yaml_path: str,
    out_xrdf_path: str,
    acceleration_limit: float = 12.0,
    jerk_limit: float = 500.0,
    joint_allowlist: Optional[Sequence[str]] = None,
    link_allowlist: Optional[Sequence[str]] = None,
    locked_joint_positions: Optional[Dict[str, float]] = None,
    tool_frames: Optional[Sequence[str]] = None,
    ee_link: Optional[str] = None,
) -> Dict:
    """Convert a cap-x cuRobo robot config into a cuMotion XRDF (format 2.0).

    Emits the schema the shipped cuMotion 1.1.0 wheel actually validates against
    (verified against the bundled ``franka.xrdf``):
      * ``cspace.joint_names`` (NOT ``joint_map``) = the *active* (optimized)
        joints, with ``acceleration_limits`` / ``jerk_limits`` as **lists**
        aligned to ``joint_names``;
      * a **top-level** ``default_joint_positions`` dict that also pins the
        *locked* joints (e.g. gripper fingers) that exist in the URDF but are
        not optimized;
      * ``world_collision`` / ``self_collision`` referencing a named
        ``geometry`` block of robot collision spheres.

    ``joint_allowlist`` is the set of joints to OPTIMIZE (the active c-space).
    This is REQUIRED for R1Pro: its cuRobo ``cspace.joint_names`` lists all 28
    joints, but ``lock_joints`` pins the 6 *synthetic* holonomic-base joints
    (``base_footprint_*_joint``, present only in the live USD and in NO URDF)
    and the 4 gripper-finger joints, leaving an 18-DOF actuated arm/torso chain.
    Pass ``joint_allowlist = cspace.joint_names - lock_joints`` so the URDF+XRDF
    cuMotion load restricts to that chain.  ``locked_joint_positions`` supplies
    the fixed values for the locked joints that DO exist in the URDF (gripper
    fingers); base joints are absent from the URDF and handled by base-frame
    planning instead.  ``link_allowlist`` keeps only collision spheres whose
    link is present in the chosen URDF.
    """
    kin = yaml.safe_load(open(curobo_yaml_path))["robot_cfg"]["kinematics"]
    jn_all = list(kin["cspace"]["joint_names"])
    retract = kin["cspace"].get("retract_config") or [0.0] * len(jn_all)
    retract = list(retract)[: len(jn_all)] + [0.0] * max(0, len(jn_all) - len(retract))
    retract_by_name = {j: float(v) for j, v in zip(jn_all, retract)}

    if joint_allowlist is not None:
        allow = set(joint_allowlist)
        active = [j for j in jn_all if j in allow]
    else:
        active = jn_all
    link_allow = set(link_allowlist) if link_allowlist is not None else None
    ee = ee_link or kin.get("ee_link")
    tool = list(tool_frames) if tool_frames is not None else ([ee] if ee else [])

    sphere_buf = kin.get("collision_sphere_buffer", 0.0)
    if isinstance(sphere_buf, dict):
        sphere_buf = float(next(iter(sphere_buf.values()))) if sphere_buf else 0.0

    spheres: Dict[str, List[Dict]] = {}
    for link, lst in (kin.get("collision_spheres", {}) or {}).items():
        if link_allow is not None and link not in link_allow:
            continue
        out = []
        for s in lst:
            c = s["center"]
            r = float(s["radius"])
            if r <= 0:  # cuRobo disables spheres with negative radius
                continue
            out.append({"center": [float(c[0]), float(c[1]), float(c[2])],
                        "radius": r + float(sphere_buf)})
        if out:
            spheres[link] = out

    # Top-level default positions: active joints at their retract value, plus the
    # locked joints (gripper fingers) pinned at the caller-supplied values.
    default_positions = {j: retract_by_name.get(j, 0.0) for j in active}
    for j, v in (locked_joint_positions or {}).items():
        default_positions[j] = float(v)

    self_ignore = kin.get("self_collision_ignore", {}) or {}
    geom_name = "robot_collision_spheres"
    xrdf = {
        "format": "xrdf",
        "format_version": 2.0,
        "default_joint_positions": default_positions,
        "cspace": {
            "joint_names": active,
            "acceleration_limits": [float(acceleration_limit)] * len(active),
            "jerk_limits": [float(jerk_limit)] * len(active),
        },
        "tool_frames": tool,
        "world_collision": {"geometry": geom_name},
        "self_collision": {
            "geometry": geom_name,
            "ignore": {
                k: [x for x in (v if isinstance(v, list) else [v]) if x in spheres]
                for k, v in self_ignore.items() if k in spheres
            },
        },
        "geometry": {geom_name: {"spheres": spheres}},
    }
    with open(out_xrdf_path, "w") as f:
        yaml.safe_dump(xrdf, f, sort_keys=False)
    return xrdf


# --------------------------------------------------------------------------- #
# URDF resolution: append synthetic tool frames absent from the shipped URDF   #
# --------------------------------------------------------------------------- #
def eef_frames_from_source_cfg(source_cfg_path: str) -> List[Dict]:
    """Read the OmniGibson robot ``*_source_cfg.yaml`` ``eef_vis_links`` block and
    return tool-frame specs ``[{name, parent_link, xyz, quat_xyzw}, ...]``.

    R1Pro's planning ee_link (``left_eef_link`` / ``right_eef_link``) is a
    synthetic frame defined here as a fixed offset from the gripper link; it is
    NOT in any shipped URDF, so it must be appended before a URDF+XRDF cuMotion
    load can resolve it as a tool frame.
    """
    cfg = yaml.safe_load(open(source_cfg_path))
    out = []
    for e in cfg.get("eef_vis_links", []) or []:
        off = e.get("offset", {})
        out.append({
            "name": e["link"],
            "parent_link": e["parent_link"],
            "xyz": list(off.get("position", [0.0, 0.0, 0.0])),
            "quat_xyzw": list(off.get("orientation", [0.0, 0.0, 0.0, 1.0])),
        })
    return out


def derive_urdf_with_tool_frames(urdf_in: str, frames: Sequence[Dict], out_path: str) -> str:
    """Write a copy of ``urdf_in`` with extra fixed ``frames`` appended.

    Each frame dict is ``{name, parent_link, xyz, quat_xyzw}``.  Frames whose
    name already exists (or whose parent link is missing) are skipped.  Returns
    ``out_path``.  Pure-XML, torch-free.
    """
    tree = ET.parse(urdf_in)
    root = tree.getroot()
    links = {l.get("name") for l in root.findall("link")}
    for fr in frames:
        name = fr["name"]
        if name in links or fr["parent_link"] not in links:
            continue
        R = _quat_xyzw_to_R(fr.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]))
        rpy = _R_to_rpy(R)
        ET.SubElement(root, "link").set("name", name)
        j = ET.SubElement(root, "joint")
        j.set("name", name + "_fixed_joint")
        j.set("type", "fixed")
        ET.SubElement(j, "parent").set("link", fr["parent_link"])
        ET.SubElement(j, "child").set("link", name)
        org = ET.SubElement(j, "origin")
        org.set("xyz", " ".join(str(float(x)) for x in fr.get("xyz", [0, 0, 0])))
        org.set("rpy", " ".join(str(float(x)) for x in rpy))
        links.add(name)
    tree.write(out_path)
    return out_path


def urdf_joint_link_names(urdf_path: str) -> Tuple[set, set]:
    """Return (joint_names, link_names) present in a URDF."""
    root = ET.parse(urdf_path).getroot()
    return ({j.get("name") for j in root.findall("joint")},
            {l.get("name") for l in root.findall("link")})


def _quat_xyzw_to_R(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    sy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    if sy > 1e-6:
        return (math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy),
                math.atan2(R[1, 0], R[0, 0]))
    return math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0


# --------------------------------------------------------------------------- #
# Mesh-world bridge: arbitrary triangle mesh -> cuMotion primitive(s)          #
# --------------------------------------------------------------------------- #
# cuMotion 1.1.0 has NO mesh / NO SDF obstacle type, so OmniGibson collision
# meshes (passed by cap-x as curobo.geom.types.Mesh / WorldConfig(mesh=...)) are
# bridged to *conservative* primitives.  Two strategies are provided:
#   * AABB / OBB cuboid  (default, 1 cuboid per mesh, over-approximating => safe)
#   * sphere fill        (optional, finer for round geometry)
# A convex-decomposition path (mesh -> N cuboids/capsules) is documented in
# cumotion_swap.md as the higher-fidelity follow-up; it slots in here unchanged.
def _quat_wxyz_to_R(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mesh_to_cuboid(
    name: str,
    vertices: np.ndarray,
    faces: Optional[np.ndarray] = None,
    pose_pos: Optional[Sequence[float]] = None,
    pose_quat_wxyz: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
    oriented: bool = False,
) -> Tuple["cm.Obstacle", "cm.Pose3", np.ndarray, np.ndarray]:
    """Bridge a triangle mesh to a single conservative cuMotion CUBOID obstacle.

    ``vertices`` are in the mesh's local frame; ``pose_pos`` / ``pose_quat_wxyz``
    place it in the planning (robot) frame, matching the (pos, wxyz) convention
    cap-x already uses when building curobo ``Mesh`` obstacles.
    Returns (obstacle, pose, side_lengths, center_in_world).
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if scale is not None:
        v = v * np.asarray(scale, dtype=np.float64).reshape(1, 3)
    R = _quat_wxyz_to_R(pose_quat_wxyz) if pose_quat_wxyz is not None else np.eye(3)
    t = np.asarray(pose_pos, dtype=np.float64) if pose_pos is not None else np.zeros(3)

    if oriented:
        # OBB in the mesh local frame, then rotate the box frame into world.
        lo = v.min(axis=0); hi = v.max(axis=0)
        local_center = (lo + hi) / 2.0
        side = np.maximum(hi - lo, 1e-4)
        center = R @ local_center + t
        box_R = R
    else:
        # AABB of the world-transformed vertices (axis-aligned, never rotated).
        vw = (R @ v.T).T + t
        lo = vw.min(axis=0); hi = vw.max(axis=0)
        center = (lo + hi) / 2.0
        side = np.maximum(hi - lo, 1e-4)
        box_R = np.eye(3)

    ob = cm.create_obstacle(cm.Obstacle.Type.CUBOID)
    ob.set_attribute(cm.Obstacle.Attribute.SIDE_LENGTHS, side.astype(np.float64))
    _maybe_name(ob, name)
    pose = _pose_from_R_t(box_R, center)
    return ob, pose, side, center


def _maybe_name(ob, name):
    attr = getattr(cm.Obstacle.Attribute, "NAME", None)
    if attr is not None:
        try:
            ob.set_attribute(attr, str(name))
        except Exception:
            pass


def _pose_from_R_t(R: np.ndarray, t: np.ndarray) -> "cm.Pose3":
    """Build a cuMotion Pose3 from rotation matrix + translation.

    The 1.1.0 wheel exposes ``Rotation3.from_matrix`` (verified by introspection);
    there is no quaternion constructor.  Identity rotations take the cheaper
    translation-only path."""
    t = np.asarray(t, dtype=np.float64)
    if np.allclose(R, np.eye(3)):
        return cm.Pose3.from_translation(t)
    return cm.Pose3(cm.Rotation3.from_matrix(np.asarray(R, dtype=np.float64)), t)


# --------------------------------------------------------------------------- #
# Result containers (numpy)                                                     #
# --------------------------------------------------------------------------- #
class IkResult:
    def __init__(self, success: bool, q: Optional[np.ndarray], n_solutions: int,
                 time_ms: float, status: str):
        self.success = success
        self.q = q                      # (ndof,) best solution or None
        self.n_solutions = n_solutions
        self.time_ms = time_ms
        self.status = status


class PlanResult:
    def __init__(self, success: bool, knots: Optional[np.ndarray],
                 interpolated: Optional[np.ndarray], time_ms: float):
        self.success = success
        self.path = knots               # (K, ndof) sparse waypoints or None
        self.interpolated_path = interpolated  # (T, ndof) dense trajectory or None
        self.time_ms = time_ms


# --------------------------------------------------------------------------- #
# Arm backend                                                                  #
# --------------------------------------------------------------------------- #
class CuMotionArmBackend:
    """Torch-free cuMotion motion backend for a single (arm) kinematic chain.

    Wraps: RobotDescription + Kinematics + World(+world_view) +
    CollisionFreeIkSolver + MotionPlanner.  All public methods take/return numpy.
    """

    def __init__(
        self,
        urdf_path: str,
        xrdf_path: str,
        planner_yaml_path: Optional[str] = None,
        ee_link: Optional[str] = None,
        num_ik_seeds: int = 24,
    ):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(urdf_path)
        if not os.path.isfile(xrdf_path):
            raise FileNotFoundError(xrdf_path)
        self.urdf_path = urdf_path
        self.xrdf_path = xrdf_path
        # Joints that actually exist in the URDF.  A locked-joint setter must
        # never write a joint absent from the URDF into the XRDF, or the
        # cm.load_robot_from_file reload throws (e.g. synthetic base_footprint_*
        # joints).  Used as a defensive guard in set_locked_joint_positions.
        self._urdf_joints = set(urdf_joint_link_names(urdf_path)[0])
        # planner_yaml_path is optional: cuMotion's create_default_motion_planner_config
        # builds a working planner from the robot description alone (no robot-specific
        # tuning file is shipped for R1Pro).
        self.planner_yaml_path = planner_yaml_path if (
            planner_yaml_path and os.path.isfile(planner_yaml_path)) else None

        self.robot = cm.load_robot_from_file(xrdf_path, urdf_path)
        self.kin = self.robot.kinematics()
        self.ndof = self.kin.num_cspace_coords()
        # The authoritative c-space order is cuMotion's internal order, which need
        # not equal the XRDF joint_names order.  Name-based mapping in the shim
        # MUST gather the start config in THIS order.
        self.cspace_names = [self.kin.cspace_coord_name(i) for i in range(self.ndof)]
        # tool/ee frame: explicit arg, else first tool frame from the XRDF
        self.ee_link = ee_link or yaml.safe_load(open(xrdf_path))["tool_frames"][0]

        self.world = cm.create_world()
        self.world_view = self.world.add_world_view()
        self.world_view.update()
        self._obstacles: List = []

        self._num_ik_seeds = num_ik_seeds
        self._ik_solver = None
        self._planner = None
        # Real robot-vs-world(+self) collision query (see check_collisions / in_collision).
        self._inspector = cm.create_robot_world_inspector(self.robot, self.world_view)
        self._build_solvers()

    # -- solver (re)build: needed after the world_view is first published ------
    def _build_solvers(self):
        ik_cfg = cm.create_default_collision_free_ik_solver_config(
            self.robot, self.ee_link, self.world_view
        )
        try:
            ik_cfg.set_param("num_seeds", int(self._num_ik_seeds))
        except Exception:
            pass
        self._ik_solver = cm.create_collision_free_ik_solver(ik_cfg)

        if self.planner_yaml_path is not None:
            planner_cfg = cm.create_motion_planner_config_from_file(
                self.planner_yaml_path, self.robot, self.ee_link, self.world_view
            )
        else:
            planner_cfg = cm.create_default_motion_planner_config(
                self.robot, self.ee_link, self.world_view
            )
        self._planner = cm.create_motion_planner(planner_cfg)
        # Re-point the collision inspector at the current world snapshot.
        self._inspector.set_world_view(self.world_view)

    # -- world management -----------------------------------------------------
    def clear_obstacles(self):
        """Drop all obstacles and republish an empty world snapshot."""
        self.world = cm.create_world()
        self.world_view = self.world.add_world_view()
        self.world_view.update()
        self._obstacles = []
        self._build_solvers()

    def set_obstacles(self, obstacles: List[Dict]):
        """Rebuild the world from a list of obstacle dicts, then publish.

        Each obstacle dict is one of:
          {"type": "cuboid",  "name", "dims":[lx,ly,lz], "pos":[x,y,z], "quat_wxyz"?}
          {"type": "sphere",  "name", "radius", "pos":[x,y,z]}
          {"type": "capsule", "name", "radius", "length", "pos", "quat_wxyz"?}
          {"type": "mesh",    "name", "vertices", "faces"?, "pos"?, "quat_wxyz"?, "scale"?}
        Mesh obstacles are bridged to a conservative AABB cuboid (see module docs).
        """
        self.world = cm.create_world()
        self._obstacles = []
        for ob in obstacles:
            t = ob.get("type", "cuboid")
            if t == "mesh":
                obj, pose, _, _ = mesh_to_cuboid(
                    ob.get("name", "mesh"), ob["vertices"], ob.get("faces"),
                    ob.get("pos"), ob.get("quat_wxyz"), ob.get("scale"),
                    oriented=ob.get("oriented", False),
                )
            elif t == "sphere":
                obj = cm.create_obstacle(cm.Obstacle.Type.SPHERE)
                obj.set_attribute(cm.Obstacle.Attribute.RADIUS, float(ob["radius"]))
                _maybe_name(obj, ob.get("name", "sphere"))
                pose = cm.Pose3.from_translation(np.asarray(ob["pos"], dtype=np.float64))
            elif t == "capsule":
                obj = cm.create_obstacle(cm.Obstacle.Type.CAPSULE)
                obj.set_attribute(cm.Obstacle.Attribute.RADIUS, float(ob["radius"]))
                if hasattr(cm.Obstacle.Attribute, "LENGTH"):
                    obj.set_attribute(cm.Obstacle.Attribute.LENGTH, float(ob["length"]))
                _maybe_name(obj, ob.get("name", "capsule"))
                pose = _pose_from_R_t(
                    _quat_wxyz_to_R(ob["quat_wxyz"]) if ob.get("quat_wxyz") else np.eye(3),
                    np.asarray(ob["pos"], dtype=np.float64),
                )
            else:  # cuboid
                obj = cm.create_obstacle(cm.Obstacle.Type.CUBOID)
                obj.set_attribute(cm.Obstacle.Attribute.SIDE_LENGTHS,
                                  np.asarray(ob["dims"], dtype=np.float64))
                _maybe_name(obj, ob.get("name", "cuboid"))
                pose = _pose_from_R_t(
                    _quat_wxyz_to_R(ob["quat_wxyz"]) if ob.get("quat_wxyz") else np.eye(3),
                    np.asarray(ob["pos"], dtype=np.float64),
                )
            self.world.add_obstacle(obj, pose)
            self._obstacles.append(obj)
        self.world_view = self.world.add_world_view()
        self.world_view.update()
        self._build_solvers()
        return len(self._obstacles)

    # -- kinematics -----------------------------------------------------------
    def fk(self, q: np.ndarray, frame: Optional[str] = None):
        return self.kin.pose(np.asarray(q, dtype=np.float64), frame or self.ee_link)

    # -- real collision query -------------------------------------------------
    def in_collision(self, q: np.ndarray, self_collision: bool = True,
                     world: bool = True) -> bool:
        """True if config ``q`` (in this backend's c-space order) collides with a
        world obstacle and/or itself.

        Uses cuMotion's ``RobotWorldInspector`` (the same sphere model the IK /
        planner use), giving true robot-vs-world (+self) collision -- matching
        cuRobo's ``check_collisions`` semantics -- rather than an FK/IK proxy.
        """
        q = np.asarray(q, dtype=np.float64)
        hit = False
        if world:
            hit = hit or bool(self._inspector.in_collision_with_obstacle(q))
        if self_collision:
            hit = hit or bool(self._inspector.in_self_collision(q))
        return hit

    def min_obstacle_distance(self, q: np.ndarray) -> float:
        """Signed min distance between robot spheres and obstacles (>0 == clear)."""
        return float(self._inspector.min_distance_to_obstacle(np.asarray(q, dtype=np.float64)))

    # -- locked-joint handling ------------------------------------------------
    def set_locked_joint_positions(self, values: Dict[str, float], tol: float = 5e-3) -> bool:
        """Re-pin the URDF-present locked joints (e.g. gripper fingers) at
        ``values`` and reload the robot if any value changed by more than ``tol``.

        cuMotion bakes non-c-space (locked) joints at the XRDF
        ``default_joint_positions``; there is no in-place setter, so changing a
        locked value requires a robot reload + solver rebuild.  We therefore only
        reload when a value actually moved (gripper open/close), keeping the
        common case (unchanged grippers) free.  Base joints are absent from the
        URDF and handled by base-frame planning, so they are ignored here.
        Returns True iff a reload happened.
        """
        if not values:
            return False
        xrdf = yaml.safe_load(open(self.xrdf_path))
        djp = xrdf.get("default_joint_positions", {}) or {}
        active = set(xrdf.get("cspace", {}).get("joint_names", []))
        changed = False
        for j, v in values.items():
            if j in active:
                continue  # optimized joints are not "locked"
            if j not in self._urdf_joints:
                # Absent from the URDF (e.g. synthetic base_footprint_* joints):
                # baking it into default_joint_positions would crash the reload.
                # Base placement is handled by base-frame planning instead.
                continue
            if abs(float(djp.get(j, 0.0)) - float(v)) > tol:
                djp[j] = float(v)
                changed = True
        if not changed:
            return False
        xrdf["default_joint_positions"] = djp
        with open(self.xrdf_path, "w") as f:
            yaml.safe_dump(xrdf, f, sort_keys=False)
        # Reload robot + rebuild solvers/inspector against the existing world.
        self.robot = cm.load_robot_from_file(self.xrdf_path, self.urdf_path)
        self.kin = self.robot.kinematics()
        self.cspace_names = [self.kin.cspace_coord_name(i) for i in range(self.ndof)]
        self._inspector = cm.create_robot_world_inspector(self.robot, self.world_view)
        self._build_solvers()
        return True

    # -- collision-free IK ----------------------------------------------------
    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_quat_wxyz: Optional[Sequence[float]] = None,
        seed: Optional[np.ndarray] = None,
    ) -> IkResult:
        tpos = np.asarray(target_pos, dtype=np.float64)
        Tr = cm.CollisionFreeIkSolver.TranslationConstraint.target(tpos)
        if target_quat_wxyz is not None:
            # cuMotion has no quaternion ctor; go wxyz quat -> R -> from_matrix.
            Orot = cm.Rotation3.from_matrix(_quat_wxyz_to_R(target_quat_wxyz))
            Oc = cm.CollisionFreeIkSolver.OrientationConstraint.target(Orot)
            tgt = cm.CollisionFreeIkSolver.TaskSpaceTarget(Tr, Oc)
        else:
            tgt = cm.CollisionFreeIkSolver.TaskSpaceTarget(Tr)
        t0 = time.time()
        res = self._ik_solver.solve(tgt)
        dt = (time.time() - t0) * 1e3
        status = str(res.status())
        try:
            qs = np.asarray(res.cspace_positions())
        except Exception:
            qs = None
        if qs is not None and qs.ndim == 1:
            qs = qs.reshape(1, -1)
        ok = ("SUCCESS" in status.upper()) and qs is not None and len(qs) > 0
        best = qs[0] if ok else None
        return IkResult(ok, best, (len(qs) if qs is not None else 0), dt, status)

    # -- motion planning ------------------------------------------------------
    def plan_to_cspace(self, q_start: np.ndarray, q_goal: np.ndarray,
                       gen_interpolation: bool = True) -> PlanResult:
        t0 = time.time()
        res = self._planner.plan_to_cspace_target(
            np.asarray(q_start, dtype=np.float64),
            np.asarray(q_goal, dtype=np.float64),
            gen_interpolation,
        )
        return self._wrap_plan(res, time.time() - t0)

    def plan_to_pose(self, q_start: np.ndarray, target_pos: Sequence[float],
                     target_quat_wxyz: Optional[Sequence[float]] = None,
                     gen_interpolation: bool = True) -> PlanResult:
        """Plan from start config ``q_start`` to a goal ee pose.

        Uses cuMotion's native pose/translation planner target so the planner
        chooses a *reachable* goal configuration itself (the planner solves goal
        IK internally and optimizes a collision-free trajectory).  This is more
        robust than committing to a single IK solution up front, which can pick a
        collision-free-but-unreachable goal config.
        """
        q_start = np.asarray(q_start, dtype=np.float64)
        tpos = np.asarray(target_pos, dtype=np.float64)
        t0 = time.time()
        if target_quat_wxyz is not None:
            pose = cm.Pose3(cm.Rotation3.from_matrix(_quat_wxyz_to_R(target_quat_wxyz)), tpos)
            res = self._planner.plan_to_pose_target(q_start, pose, gen_interpolation)
        else:
            res = self._planner.plan_to_translation_target(q_start, tpos, gen_interpolation)
        return self._wrap_plan(res, time.time() - t0)

    def _wrap_plan(self, res, dt_s: float) -> PlanResult:
        found = bool(getattr(res, "path_found", False))
        knots = None
        interp = None
        if found:
            try:
                knots = np.asarray(res.path)
            except Exception:
                knots = None
            try:
                interp = np.asarray(res.interpolated_path)
            except Exception:
                interp = None
        return PlanResult(found, knots, interp, dt_s * 1e3)
