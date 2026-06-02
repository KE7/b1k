"""
cuMotion-backed, drop-in replacement for ``CuRoboMotionGenerator``.

==============================================================================
LIMITATION — ARM-ONLY. DO NOT USE AS A BASE-NAV MOTION-GENERATOR BACKEND.
==============================================================================
This shim is an ARM motion generator only. It can plan for the R1Pro (and
similar) ARM chain, but it CANNOT be used as the backend for BASE / holonomic
base navigation (``CuRoboEmbodimentSelection.BASE`` / ``_navigate_to_pose``).

Why the BASE backend cannot even be constructed (verified — see ``cumotion_swap.md``
/ the cumotion-planner-probe runs):
  * R1Pro's holonomic base is realized by SYNTHETIC virtual joints
    ``base_footprint_x`` / ``base_footprint_y`` / ``base_footprint_rz`` (plus the
    locked ``base_footprint_z/rx/ry``). These joints exist ONLY in the robot USD
    and in ``robot.get_joint_positions()`` — they are present in NO URDF and in NO
    XRDF.
  * cuMotion builds its robot model from URDF + XRDF via
    ``cm.load_robot_from_file(xrdf, urdf)``. There is no holonomic / planar-joint
    primitive to represent the base DOFs, and writing those synthetic joints into
    the XRDF (e.g. as cspace or as locked ``default_joint_positions``) makes the
    load fail hard:
        RuntimeError: ... URDF does not include a joint with name
        'base_footprint_z_joint'
  * This shim therefore FILTERS the ``base_footprint_*`` joints out of the active
    chain and out of locked-joint handling (see ``update_locked_joints`` and
    ``_to_base_frame``), so it can construct an ARM-only cuMotion robot. The base
    pose is consumed only as a planning FRAME (targets expressed in the base
    frame), never as plannable DOFs.
  * Net: there is no cuMotion robot that exposes the base as movable joints, so a
    BASE motion generator simply cannot be built from this path. Base navigation
    must keep using the existing cuRobo BASE backend; only ARM planning may be
    swapped to this shim.
==============================================================================

This shim presents the *same public surface* that cap-x / OmniGibson consume from
``omnigibson.action_primitives.curobo.CuRoboMotionGenerator`` (constructor +
``batch_size``, ``ee_link``/``base_link``, ``update_obstacles``,
``remove_obstacles``, ``compute_trajectories``, ``path_to_joint_trajectory``,
``add_linearly_interpolated_waypoints``, ``check_collisions``,
``update_locked_joints``, ``tensor_args``), but its internals call the torch-free
``cumotion_backend.CuMotionArmBackend`` (cuMotion 1.1.0 IkSolver /
CollisionFreeIkSolver / MotionPlanner) instead of the cuRobo ``MotionGen`` stack.

To SWAP IN, change one import in the caller(s)::

    # from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
    from omnigibson.action_primitives.cumotion_motion_generator import CuRoboMotionGenerator

PyTorch is used only for the I/O boundary (OmniGibson hands torch tensors and the
controllers expect torch back); all GPU motion math runs in cuMotion's static
CUDA-13 / sm_121 runtime, dodging the cuRobo-custom-op / cu128-torch crash class.

Mesh-world bridge: OmniGibson collision meshes are converted to conservative
cuMotion CUBOID obstacles (cuMotion 1.1.0 exposes no mesh / no SDF type) -- see
``cumotion_backend`` and ``cumotion_swap.md`` for the fidelity trade-off and the
documented convex-decomposition follow-up.
"""
from __future__ import annotations

import math
import os
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch as th  # MUST come before importing omni!!!

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros

from omnigibson.action_primitives import cumotion_backend as cmb


m = create_module_macros(module_path=__file__)
m.DEFAULT_COLLISION_ACTIVATION_DISTANCE = 0.005
m.DEFAULT_INTERPOLATION_DT = 0.01


class CuRoboEmbodimentSelection(str, Enum):
    BASE = "base"
    ARM = "arm"
    DEFAULT = "default"


# --------------------------------------------------------------------------- #
# Lightweight stand-in for cuRobo JointState path objects                       #
# --------------------------------------------------------------------------- #
class JointStatePath:
    """Minimal cuRobo-JointState-compatible trajectory holder.

    Holds a dense (T, ndof_arm) trajectory over the backend's arm joints plus the
    embedding info needed to expand into the robot's full joint vector.
    """

    def __init__(self, arm_traj: np.ndarray, arm_joint_names: List[str],
                 full_joint_names: List[str], base_full_position: th.Tensor):
        self._arm_traj = np.asarray(arm_traj, dtype=np.float64)
        self.joint_names = list(arm_joint_names)
        self._full_joint_names = list(full_joint_names)
        self._base_full = base_full_position  # (D_full,) current robot joint pos

    @property
    def position(self) -> th.Tensor:
        return th.tensor(self._arm_traj, dtype=th.float32)

    def to_full_trajectory(self) -> th.Tensor:
        """Expand the arm trajectory into a (T, D_full) trajectory, holding all
        non-arm (locked / base / gripper) joints at their current positions."""
        T_steps = self._arm_traj.shape[0]
        full = self._base_full.detach().clone().to(th.float32).unsqueeze(0).repeat(T_steps, 1)
        name_to_idx = {n: i for i, n in enumerate(self._full_joint_names)}
        for j, jn in enumerate(self.joint_names):
            if jn in name_to_idx:
                full[:, name_to_idx[jn]] = th.tensor(self._arm_traj[:, j], dtype=th.float32)
        return full


class CuMotionMotionGenerator:
    """cuMotion-backed motion generator with a cuRobo-compatible public surface."""

    def __init__(
        self,
        robot,
        robot_cfg_path=None,
        robot_usd_path=None,
        device="cuda:0",
        motion_cfg_kwargs=None,
        batch_size=2,
        use_cuda_graph=True,
        debug=False,
        use_default_embodiment_only=False,
        collision_activation_distance=m.DEFAULT_COLLISION_ACTIVATION_DISTANCE,
        urdf_path=None,
        planner_yaml_path=None,
    ):
        assert len(og.sim.scenes) == 1
        self.robot = robot
        self.debug = debug
        self.batch_size = batch_size
        self.device = device
        self.collision_activation_distance = collision_activation_distance
        self.robot_joint_names = list(robot.joints.keys())

        # Resolve the per-embodiment cuRobo robot-config paths (same source the
        # cuRobo generator uses: robot.curobo_path, a dict keyed by embodiment).
        robot_cfg_path_dict = robot.curobo_path if robot_cfg_path is None else robot_cfg_path
        if not isinstance(robot_cfg_path_dict, dict):
            robot_cfg_path_dict = {CuRoboEmbodimentSelection.DEFAULT: robot_cfg_path_dict}
        if use_default_embodiment_only:
            robot_cfg_path_dict = {
                CuRoboEmbodimentSelection.DEFAULT: robot_cfg_path_dict[CuRoboEmbodimentSelection.DEFAULT]
            }

        import yaml

        # Per-embodiment metadata + the active/locked joint split (cuRobo lists
        # ALL joints in cspace.joint_names and pins a subset via lock_joints; the
        # ACTIVE chain cuMotion optimizes is cspace - lock_joints).
        self.backends: Dict[CuRoboEmbodimentSelection, cmb.CuMotionArmBackend] = {}
        self.ee_link: Dict[CuRoboEmbodimentSelection, str] = {}
        self.base_link: Dict[CuRoboEmbodimentSelection, str] = {}
        self.additional_links: Dict[CuRoboEmbodimentSelection, list] = {}
        self.active_joints: Dict[CuRoboEmbodimentSelection, list] = {}
        self.locked_joints: Dict[CuRoboEmbodimentSelection, list] = {}

        kin_by_emb = {}
        all_ee_links = []
        for emb_sel, cfg_path in robot_cfg_path_dict.items():
            kin = yaml.safe_load(open(cfg_path))["robot_cfg"]["kinematics"]
            kin_by_emb[emb_sel] = kin
            self.ee_link[emb_sel] = kin["ee_link"]
            self.base_link[emb_sel] = kin["base_link"]
            self.additional_links[emb_sel] = list(kin.get("link_names", []) or [])
            for l in [kin["ee_link"], *self.additional_links[emb_sel]]:
                if l not in all_ee_links:
                    all_ee_links.append(l)

        # Resolve a real R1Pro URDF (arm chain) and append the synthetic ee/eef
        # frames that live only in the USD, so cuMotion can resolve the tool frame.
        self._urdf_path = urdf_path or self._guess_urdf_path(
            robot, all_ee_links, cfg_paths=list(robot_cfg_path_dict.values()))
        self._planner_yaml_path = planner_yaml_path  # None -> cuMotion default config
        urdf_joints, urdf_links = cmb.urdf_joint_link_names(self._urdf_path)
        # Authoritative set of joints that actually exist in the planning URDF.
        # Used to filter locked joints so synthetic holonomic-base joints
        # (base_footprint_*_joint, present in robot.get_joint_positions() but in
        # NO URDF) are never forwarded to the backend XRDF (see
        # update_locked_joints / set_locked_joint_positions).
        self._urdf_joints = set(urdf_joints)

        # Current joint positions (robot.joints order) for pinning locked joints.
        try:
            full_q0 = self.robot.get_joint_positions().detach().cpu().numpy()
        except Exception:
            full_q0 = None
        name_to_q0 = ({n: float(full_q0[i]) for i, n in enumerate(self.robot_joint_names)}
                      if full_q0 is not None else {})

        for emb_sel, cfg_path in robot_cfg_path_dict.items():
            kin = kin_by_emb[emb_sel]
            cspace = list(kin["cspace"]["joint_names"])
            locked = set((kin.get("lock_joints") or {}).keys())
            active = [j for j in cspace if j not in locked]
            self.active_joints[emb_sel] = active
            self.locked_joints[emb_sel] = [j for j in cspace if j in locked]

            # Locked joints that actually exist in the URDF (gripper fingers): pin
            # them at the robot's current value.  Base joints are absent from the
            # URDF and handled by base-frame planning, so they are skipped.
            locked_pos = {j: name_to_q0.get(j, 0.0)
                          for j in locked if j in urdf_joints}
            tool_frames = [l for l in [kin["ee_link"], *self.additional_links[emb_sel]]
                           if l in urdf_links]

            xrdf_path = os.path.join("/tmp", f"r1_swap_{str(emb_sel)}.xrdf")
            cmb.xrdf_from_curobo_yaml(
                cfg_path, xrdf_path,
                joint_allowlist=active,
                link_allowlist=urdf_links,
                locked_joint_positions=locked_pos,
                tool_frames=tool_frames or [kin["ee_link"]],
                ee_link=kin["ee_link"],
            )
            self.backends[emb_sel] = cmb.CuMotionArmBackend(
                urdf_path=self._urdf_path,
                xrdf_path=xrdf_path,
                planner_yaml_path=self._planner_yaml_path,
                ee_link=kin["ee_link"],
            )

    # ---- path helpers -------------------------------------------------------
    @staticmethod
    def _guess_urdf_path(robot, ee_links=None, cfg_paths=None):
        """Resolve an R1Pro URDF containing the actuated arm chain, appending the
        synthetic ee/eef tool frames (defined in the OmniGibson source cfg, not in
        any shipped URDF) so cuMotion can resolve the planning tool frame.

        Returns a path under /tmp to the PATCHED URDF (one with the tool frames
        appended).  The asset root and robot model are derived AUTHORITATIVELY
        from ``cfg_paths`` (== ``robot.curobo_path`` values, the very paths
        ``__init__`` loads the cuRobo cfg from) rather than from a hard-coded
        ``datasets/`` guess, so no explicit ``urdf_path`` is ever needed.  The
        returned URDF is re-parsed and the call FAILS LOUDLY if the tool frames
        were not actually appended, instead of silently returning an unpatched
        URDF that the backend would later reject with a missing-tool-frame error.
        """
        import glob

        ee_links = list(ee_links or [])

        # 1. Asset roots + model name derived from the cuRobo cfg paths, which are
        #    laid out as <assets>/models/<model>/curobo/<cfg>.yaml.  This is the
        #    same source __init__ already trusts, so it always matches the live
        #    dataset location (the old datasets/ guess pointed at a path that does
        #    not exist on aarch64 / DGX Spark and yielded an UNPATCHED URDF).
        asset_roots = []
        model_from_cfg = None
        for cfg_path in (cfg_paths or []):
            if not cfg_path:
                continue
            model_dir = os.path.dirname(os.path.dirname(cfg_path))  # models/<model>
            assets = os.path.dirname(os.path.dirname(model_dir))    # <assets>
            if os.path.isdir(os.path.join(assets, "models")) and assets not in asset_roots:
                asset_roots.append(assets)
                if model_from_cfg is None:
                    model_from_cfg = os.path.basename(model_dir)

        # 2. Candidate base URDFs: model-specific (from the authoritative asset
        #    roots) first, then any-model in those roots, then explicit robot
        #    attrs, then the legacy datasets/ glob as a last-resort fallback.
        candidates = []
        for assets in asset_roots:
            if model_from_cfg:
                candidates.extend(sorted(glob.glob(
                    os.path.join(assets, "models", model_from_cfg, "urdf", "*.urdf"))))
            candidates.extend(sorted(glob.glob(
                os.path.join(assets, "models", "*", "urdf", "*.urdf"))))
        for attr in ("urdf_path", "model_urdf_path"):
            p = getattr(robot, attr, None)
            if p and os.path.isfile(p):
                candidates.append(p)
        b1k_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", ".."))
        candidates.extend(sorted(glob.glob(os.path.join(
            b1k_root, "datasets", "omnigibson-robot-assets", "models", "*", "urdf", "*.urdf"))))

        # 3. pick the first URDF that contains the arm joints (revolute arm chain).
        need = ["left_arm_joint1", "right_arm_joint1", "torso_joint1"]
        base_urdf = None
        for p in candidates:
            try:
                joints, _ = cmb.urdf_joint_link_names(p)
            except Exception:
                continue
            if all(j in joints for j in need):
                base_urdf = p
                break
        if base_urdf is None:
            raise FileNotFoundError(
                "Could not locate an R1Pro URDF containing the arm joints; "
                "pass urdf_path=... explicitly."
            )

        # 4. append the synthetic ee/eef tool frames from the source cfg.
        _, links = cmb.urdf_joint_link_names(base_urdf)
        missing = [l for l in ee_links if l not in links]
        if not missing:
            return base_urdf
        # Prefer the model-specific source cfg under the authoritative asset root
        # (so r1pro's frames are used, not e.g. r1's); fall back to the directory
        # walk only if that is unavailable.
        src_cfg = None
        for assets in asset_roots:
            if not model_from_cfg:
                break
            for h in sorted(glob.glob(os.path.join(
                    assets, "source", model_from_cfg, "*_source_cfg.yaml"))):
                try:
                    if cmb.eef_frames_from_source_cfg(h):
                        src_cfg = h
                        break
                except Exception:
                    continue
            if src_cfg:
                break
        if src_cfg is None:
            src_cfg = CuMotionMotionGenerator._find_source_cfg(base_urdf)
        frames = cmb.eef_frames_from_source_cfg(src_cfg) if src_cfg else []
        frames = [f for f in frames if f["name"] in missing]
        out = os.path.join("/tmp", "r1pro_cumotion_patched.urdf")
        patched = cmb.derive_urdf_with_tool_frames(base_urdf, frames, out)

        # 5. VERIFY the patch actually took (parents present, frames appended).
        #    Returning an unpatched URDF here is exactly the bug being fixed, so
        #    surface it loudly instead of deferring to a cryptic backend failure.
        _, patched_links = cmb.urdf_joint_link_names(patched)
        still_missing = [l for l in ee_links if l not in patched_links]
        if still_missing:
            raise FileNotFoundError(
                f"Auto-resolved URDF '{base_urdf}' is missing tool frame(s) "
                f"{still_missing} and they could not be appended from source cfg "
                f"'{src_cfg}'. Pass urdf_path=... with the frames pre-appended."
            )
        return patched

    @staticmethod
    def _find_source_cfg(urdf_path):
        """Locate the OmniGibson *_source_cfg.yaml that defines the eef frames."""
        # .../models/<m>/urdf/<m>.urdf  ->  .../source/<m>/<m>_source_cfg.yaml
        models_dir = os.path.dirname(os.path.dirname(os.path.dirname(urdf_path)))
        root = os.path.dirname(models_dir)
        import glob
        hits = glob.glob(os.path.join(root, "source", "*", "*_source_cfg.yaml"))
        # prefer one whose eef_vis_links are present
        for h in sorted(hits):
            try:
                if cmb.eef_frames_from_source_cfg(h):
                    return h
            except Exception:
                continue
        return hits[0] if hits else None

    @property
    def tensor_args(self):
        return th.device(self.device)

    # ---- world / obstacles --------------------------------------------------
    def _scene_obstacles(self, ignore_objects=None) -> List[Dict]:
        """Extract OmniGibson collision meshes as mesh-obstacle dicts in the robot
        (base) frame, matching the conversion the cuRobo path performs."""
        obstacles: List[Dict] = []
        robot_transform = T.pose_inv(T.pose2mat(self.robot.root_link.get_position_orientation()))
        ignore_objects = ignore_objects or []
        for obj in self.robot.scene.objects:
            if obj == self.robot or getattr(obj, "visual_only", False) or obj in ignore_objects:
                continue
            for link in obj.links.values():
                for collision_mesh in (link.collision_meshes or {}).values():
                    if collision_mesh is None or collision_mesh.geom_type != "Mesh":
                        continue
                    try:
                        obj_pose = T.pose2mat(collision_mesh.get_position_orientation())
                        pos, orn = T.mat2pose(robot_transform @ obj_pose)
                        orn_wxyz = orn[[3, 0, 1, 2]]
                        obstacles.append({
                            "type": "mesh",
                            "name": collision_mesh.prim_path,
                            "vertices": collision_mesh.points.numpy(),
                            "faces": collision_mesh.faces.numpy(),
                            "pos": pos.numpy(),
                            "quat_wxyz": orn_wxyz.numpy(),
                            "scale": collision_mesh.get_world_scale().numpy(),
                        })
                    except Exception as e:
                        print(f"[cumotion] skip obstacle {getattr(collision_mesh,'prim_path','?')}: {e}")
        return obstacles

    def update_obstacles(self, ignore_objects=None):
        obstacles = self._scene_obstacles(ignore_objects=ignore_objects)
        for be in self.backends.values():
            be.set_obstacles(obstacles)
        return len(obstacles)

    def remove_obstacles(self):
        for be in self.backends.values():
            be.clear_obstacles()

    def update_locked_joints(self, cu_joint_state, emb_sel=CuRoboEmbodimentSelection.DEFAULT):
        """Hold the locked (non-active) joints fixed at the caller's values.

        Two classes of locked joint exist for R1Pro:
          * synthetic holonomic-base joints (``base_footprint_*``) -- absent from
            the URDF; their effect (where the base sits) is handled by planning in
            the base frame (see ``_to_base_frame``), so nothing to bake here;
          * gripper-finger joints -- present in the URDF as fixed joints.  We
            re-pin them in the backend's XRDF ``default_joint_positions`` (a robot
            reload only happens if a value actually moved, e.g. after a grasp).

        ``cu_joint_state`` may be a {name: value} dict, a full joint vector in
        ``robot_joint_names`` order, or an object exposing ``joint_names`` +
        ``position``.
        """
        if emb_sel not in self.backends:
            emb_sel = next(iter(self.backends))
        be = self.backends[emb_sel]
        name_to_val = self._joint_state_to_dict(cu_joint_state)
        # Restrict to locked joints that actually exist in the planning URDF's
        # active chain.  cuRobo's cspace lists synthetic holonomic-base joints
        # (base_footprint_*_joint) as locked, but they are absent from every URDF
        # and are realized by base-frame planning (see _to_base_frame).  Passing
        # them to the backend would write a non-existent joint into the XRDF and
        # crash the cm.load_robot_from_file reload.  Filtering them here leaves
        # only URDF-present locked joints (the gripper fingers).
        urdf_joints = getattr(self, "_urdf_joints", None)
        locked = [j for j in self.locked_joints.get(emb_sel, [])
                  if urdf_joints is None or j in urdf_joints]
        values = {j: name_to_val[j] for j in locked if j in name_to_val}
        if not values:
            return None
        return be.set_locked_joint_positions(values)

    def _joint_state_to_dict(self, cu_joint_state) -> Dict[str, float]:
        """Normalize various joint-state representations to {name: value}."""
        if cu_joint_state is None:
            return {}
        if isinstance(cu_joint_state, dict):
            return {k: float(v) for k, v in cu_joint_state.items()}
        jn = getattr(cu_joint_state, "joint_names", None)
        pos = getattr(cu_joint_state, "position", None)
        if jn is not None and pos is not None:
            arr = np.asarray(pos.detach().cpu() if isinstance(pos, th.Tensor) else pos,
                             dtype=np.float64).reshape(-1)
            return {n: float(arr[i]) for i, n in enumerate(jn) if i < arr.shape[0]}
        # bare full vector in robot_joint_names order
        arr = np.asarray(cu_joint_state.detach().cpu() if isinstance(cu_joint_state, th.Tensor)
                         else cu_joint_state, dtype=np.float64).reshape(-1)
        return {n: float(arr[i]) for i, n in enumerate(self.robot_joint_names)
                if i < arr.shape[0]}

    # ---- pose conversion ----------------------------------------------------
    def _to_base_frame(self, emb_sel, pos_xyz: th.Tensor, quat_xyzw: th.Tensor, is_local: bool):
        """Return (pos[np], quat_wxyz[np]) of a single target in the cuMotion base
        frame (the XRDF root)."""
        if not is_local:
            base_link_name = self.base_link[emb_sel]
            robot_pos, robot_quat = self.robot.links[base_link_name].get_position_orientation()
            pose = th.eye(4)
            pose[:3, :3] = T.quat2mat(quat_xyzw)
            pose[:3, 3] = pos_xyz
            pose = T.pose_inv(T.pose2mat((robot_pos, robot_quat))) @ pose
            pos_xyz = pose[:3, 3]
            quat_xyzw = T.mat2quat(pose[:3, :3])
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return pos_xyz.cpu().numpy().astype(np.float64), quat_wxyz.cpu().numpy().astype(np.float64)

    # ---- main planning entry point -----------------------------------------
    def compute_trajectories(
        self,
        target_pos,
        target_quat,
        initial_joint_pos=None,
        is_local=False,
        max_attempts=5,
        timeout=2.0,
        ik_fail_return=5,
        enable_finetune_trajopt=True,
        finetune_attempts=1,
        return_full_result=False,
        success_ratio=None,
        attached_obj=None,
        attached_obj_scale=None,
        motion_constraint=None,
        skip_obstacle_update=False,
        ik_only=False,
        ik_world_collision_check=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    ):
        """Compute trajectories to reach target ee pose(s).

        Mirrors the cuRobo generator's contract: returns
        ``(successes: BoolTensor[N], paths: List[JointStatePath|None])`` when
        ``return_full_result`` is False, else a list of raw cuMotion PlanResults.
        """
        if emb_sel not in self.backends:
            emb_sel = next(iter(self.backends))
        be = self.backends[emb_sel]

        if not skip_obstacle_update:
            self.update_obstacles()

        # Accept either a single ee tensor or a per-link dict (default ee only here).
        if isinstance(target_pos, th.Tensor):
            target_pos = {self.ee_link[emb_sel]: target_pos}
        if isinstance(target_quat, th.Tensor):
            target_quat = {self.ee_link[emb_sel]: target_quat}
        assert target_pos.keys() == target_quat.keys()
        ee = self.ee_link[emb_sel]
        tp = target_pos[ee]
        tq = target_quat[ee]
        if tp.ndim == 1:
            tp = tp.unsqueeze(0)
            tq = tq.unsqueeze(0)
        n = tp.shape[0]

        # Start config (full robot joint vector); arm subset extracted by NAME.
        full_q = (self.robot.get_joint_positions() if initial_joint_pos is None
                  else initial_joint_pos)
        # Pin the locked (gripper) joints to their current values before planning.
        self.update_locked_joints(full_q, emb_sel)
        q_start_arm = self._arm_subvector(be, full_q)

        successes = []
        paths: List[Optional[JointStatePath]] = []
        results = []
        for i in range(n):
            pos_np, quat_np = self._to_base_frame(emb_sel, tp[i], tq[i], is_local)
            if ik_only:
                ik = be.solve_ik(pos_np, quat_np)
                ok = ik.success
                res = ik
                traj = ik.q.reshape(1, -1) if ik.success else None
            else:
                res = be.plan_to_pose(q_start_arm, pos_np, quat_np, gen_interpolation=True)
                ok = res.success
                traj = res.interpolated_path if (ok and res.interpolated_path is not None) else (
                    res.path if ok else None)
            successes.append(bool(ok))
            results.append(res)
            if ok and traj is not None:
                paths.append(JointStatePath(traj, be_joint_names(be), self.robot_joint_names, full_q))
            else:
                paths.append(None)

        if return_full_result:
            return results
        return th.tensor(successes, dtype=th.bool), paths

    def _arm_subvector(self, be, full_q) -> np.ndarray:
        """Gather the backend's active c-space config from the full robot joint
        vector BY NAME, in cuMotion's internal c-space order (``be.cspace_names``).

        ``full_q`` is ordered by ``self.robot_joint_names`` (== robot.joints
        order, the order ``robot.get_joint_positions()`` returns).  R1Pro
        interleaves base/torso/dual-arm/gripper joints, so positional slicing is
        WRONG; we map each cuMotion c-space coord name to its index in
        ``robot_joint_names`` and gather.
        """
        names = be.cspace_names
        if full_q is None:
            return np.zeros(len(names), dtype=np.float64)
        arr = (full_q.detach().cpu().numpy().astype(np.float64)
               if isinstance(full_q, th.Tensor) else np.asarray(full_q, dtype=np.float64))
        idx = {n: i for i, n in enumerate(self.robot_joint_names)}
        out = np.zeros(len(names), dtype=np.float64)
        for k, jn in enumerate(names):
            if jn in idx and idx[jn] < arr.shape[0]:
                out[k] = arr[idx[jn]]
        return out

    # ---- trajectory post-processing ----------------------------------------
    def path_to_joint_trajectory(self, path, get_full_js=True,
                                 emb_sel=CuRoboEmbodimentSelection.DEFAULT):
        """Convert a JointStatePath into a (T, D) torch trajectory.

        With ``get_full_js`` the arm trajectory is expanded into the robot's full
        joint vector (non-arm joints held at current positions)."""
        if path is None:
            return None
        if get_full_js:
            return path.to_full_trajectory()
        return path.position

    def add_linearly_interpolated_waypoints(self, traj: th.Tensor, max_inter_dist=0.01):
        assert len(traj) > 1, "Plan must have at least 2 waypoints to interpolate"
        out = []
        for i in range(len(traj) - 1):
            max_diff = (traj[i + 1] - traj[i]).abs().max()
            n_int = max(1, math.ceil(max_diff.item() / max_inter_dist))
            for a in range(n_int):
                out.append(traj[i] + (traj[i + 1] - traj[i]) * (a / n_int))
        out.append(traj[-1])
        return th.stack(out)

    # ---- collision checking -------------------------------------------------
    def check_collisions(self, q, initial_joint_pos=None, self_collision_check=True,
                         skip_obstacle_update=False, attached_obj=None,
                         attached_obj_scale=None,
                         emb_sel=CuRoboEmbodimentSelection.DEFAULT):
        """True robot-vs-world (+optionally self) collision for each config.

        Uses cuMotion's ``RobotWorldInspector`` collision query on the same sphere
        model the planner uses (matching cuRobo's ``check_collisions`` semantics),
        rather than an FK/IK feasibility proxy.  Each row of ``q`` is the full
        robot joint vector (robot.joints order); we gather the active arm config
        by name.  Returns a (N,) bool tensor (True == in collision)."""
        if emb_sel not in self.backends:
            emb_sel = next(iter(self.backends))
        be = self.backends[emb_sel]
        if not skip_obstacle_update:
            self.update_obstacles()
        q = q if q.ndim == 2 else q.unsqueeze(0)
        out = []
        for i in range(q.shape[0]):
            arm_q = self._arm_subvector(be, q[i])
            out.append(be.in_collision(arm_q, self_collision=self_collision_check, world=True))
        return th.tensor(out, dtype=th.bool)


# Backward-compatibility alias: the class was historically named
# ``CuRoboMotionGenerator`` (a drop-in for the cuRobo generator of the same
# name). It was renamed ``CuMotionMotionGenerator`` to reflect that the backend
# is cuMotion, not cuRobo. Keep the old name bound to the new class so existing
# imports (``from ...cumotion_motion_generator import CuRoboMotionGenerator``)
# keep working unchanged.
CuRoboMotionGenerator = CuMotionMotionGenerator


# -- module-level helpers ----------------------------------------------------
def be_joint_names(be) -> List[str]:
    """The backend's active c-space coord names, in cuMotion's INTERNAL order
    (which need not equal the XRDF yaml order)."""
    return list(be.cspace_names)
