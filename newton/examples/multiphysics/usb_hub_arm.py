# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Franka setup and a housing-grasp trajectory for the USB hub example."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.ik as ik
import newton.utils
from newton.solvers import SolverMuJoCo

_INITIAL_Q = (-0.00368, 0.02390, 0.00368, -2.36832, -0.00013, 2.39222, 0.78549, 0.04, 0.04)
_TCP_OFFSET = 0.1034
_GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)


@dataclass
class FrankaArm:
    """Indices of the arm in the coupled model."""

    bodies: list[int]
    joints: list[int]
    shapes: list[int]
    finger_bodies: list[int]
    finger_shapes: list[int]
    gripper_bodies: list[int]
    gripper_joints: list[int]
    hand_body: int
    coord_start: int
    coord_count: int
    base_z: float


def add_franka(builder: newton.ModelBuilder, *, base_z: float = 0.25) -> FrankaArm:
    """Add the fixed-base arm with analytic finger colliders and finite PD effort."""
    if "mujoco:gravcomp" not in builder.custom_attributes:
        SolverMuJoCo.register_custom_attributes(builder)
    body_start = builder.body_count
    joint_start = builder.joint_count
    shape_start = builder.shape_count
    coord_start = len(builder.joint_q)
    dof_start = len(builder.joint_target_ke)

    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        force_show_colliders=False,
        joint_damping_mode="passive",
    )
    builder.joint_q[coord_start : coord_start + 9] = _INITIAL_Q
    builder.joint_target_q[coord_start : coord_start + 9] = _INITIAL_Q
    builder.joint_target_ke[dof_start : dof_start + 9] = [1500.0] * 7 + [3000.0] * 2
    builder.joint_target_kd[dof_start : dof_start + 9] = [100.0] * 7 + [40.0] * 2
    builder.joint_effort_limit[dof_start : dof_start + 9] = [87.0] * 4 + [12.0] * 3 + [70.0] * 2
    builder.joint_armature[dof_start : dof_start + 9] = [0.01] * 7 + [0.0] * 2

    bodies = list(range(body_start, builder.body_count))
    joints = list(range(joint_start, builder.joint_count))
    shapes = list(range(shape_start, builder.shape_count))
    hand_body = next(body for body in bodies if builder.body_label[body].endswith("fr3_hand"))
    finger_bodies = [
        body for body in bodies if builder.body_label[body].endswith(("fr3_leftfinger", "fr3_rightfinger"))
    ]
    finger_shapes = [shape for shape in shapes if builder.shape_body[shape] in finger_bodies]
    gripper_bodies = [
        body
        for body in bodies
        if builder.body_label[body].endswith(("fr3_hand", "fr3_hand_tcp", "fr3_leftfinger", "fr3_rightfinger"))
    ]
    gripper_joints = [
        joint
        for joint in joints
        if builder.joint_parent[joint] in gripper_bodies and builder.joint_child[joint] in gripper_bodies
    ]
    for shape in finger_shapes:
        builder.shape_material_ke[shape] = 1.0e5
        builder.shape_material_kd[shape] = 1.0e-3
        builder.shape_material_kf[shape] = 1.0e3
        builder.shape_material_mu[shape] = 1.0

    # Gravity compensation cancels only robot weight; the grasp still transmits
    # payload forces through contact and the finite joint drives.
    gravity_compensation = builder.custom_attributes["mujoco:gravcomp"]
    if gravity_compensation.values is None:
        gravity_compensation.values = {}
    for body in bodies:
        gravity_compensation.values[body] = 1.0

    return FrankaArm(
        bodies=bodies,
        joints=joints,
        shapes=shapes,
        finger_bodies=finger_bodies,
        finger_shapes=finger_shapes,
        gripper_bodies=gripper_bodies,
        gripper_joints=gripper_joints,
        hand_body=hand_body,
        coord_start=coord_start,
        coord_count=len(builder.joint_q) - coord_start,
        base_z=base_z,
    )


@wp.kernel
def _set_targets(
    target: wp.array[wp.vec3],
    fingers: wp.array[float],
    position: wp.vec3,
    finger_position: float,
):
    target[0] = position
    fingers[0] = finger_position


@wp.kernel
def _set_fingers(joint_q: wp.array2d[float], finger_position: wp.array[float]):
    joint_q[0, 7] = finger_position[0]
    joint_q[0, 8] = finger_position[0]


class PickupController:
    """Track a top-down approach, frictional grasp, vertical lift, and hold.

    Construct before initializing simulation states: the initial approach pose
    is written to ``model.joint_q`` and the control position targets. No payload
    state or attachment is changed by this controller.
    """

    def __init__(
        self,
        model: newton.Model,
        control: newton.Control,
        arm: FrankaArm,
        *,
        case_center: tuple[float, float, float] = (0.5, 0.0, 0.265),
        lift_height: float = 0.17,
    ):
        self.model = model
        self.control = control
        self.arm = arm
        self.device = model.device
        self.grasp_position = np.asarray(case_center, dtype=np.float32)
        self.lift_height = lift_height
        self.approach_height = 0.10
        self.lift_start_time = 3.0
        self.lift_end_time = 7.5
        self.finish_time = 12.0
        self.finger_open = 0.04
        # The 40 mm housing stops each physical finger near 20 mm. A smaller
        # target supplies pinch force without an artificial grasp constraint.
        self.finger_closed = 0.008
        self.ik_iterations = 16

        ik_builder = newton.ModelBuilder()
        ik_arm = add_franka(ik_builder, base_z=arm.base_z)
        self.ik_model = ik_builder.finalize(device=self.device)
        self.joint_q = wp.array([_INITIAL_Q], dtype=float, device=self.device)
        position = self.grasp_position + np.array((0.0, 0.0, self.approach_height), dtype=np.float32)
        self.target_position = wp.array([position], dtype=wp.vec3, device=self.device)
        self.target_rotation = wp.array([_GRIPPER_DOWN], dtype=wp.vec4, device=self.device)
        self.finger_position = wp.array([self.finger_open], dtype=float, device=self.device)
        self.position_objective = ik.IKObjectivePosition(
            link_index=ik_arm.hand_body,
            link_offset=wp.vec3(0.0, 0.0, _TCP_OFFSET),
            target_positions=self.target_position,
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            link_index=ik_arm.hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.target_rotation,
        )
        limits_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=10.0,
        )
        self.solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.position_objective, self.rotation_objective, limits_objective],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.solver.step(self.joint_q, self.joint_q, iterations=64)
        wp.launch(_set_fingers, dim=1, inputs=[self.joint_q, self.finger_position], device=self.device)
        wp.copy(
            model.joint_q,
            self.joint_q.flatten(),
            dest_offset=arm.coord_start,
            count=arm.coord_count,
        )
        self._copy_targets()

    @staticmethod
    def _smoothstep(time: float, start: float, end: float) -> float:
        alpha = float(np.clip((time - start) / (end - start), 0.0, 1.0))
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def target(self, time: float) -> tuple[np.ndarray, float]:
        """Return the desired TCP position [m] and each finger displacement [m]."""
        position = self.grasp_position.copy()
        position[2] += self.approach_height * (1.0 - self._smoothstep(time, 0.5, 1.5))
        position[2] += self.lift_height * self._smoothstep(time, self.lift_start_time, self.lift_end_time)
        closed = self._smoothstep(time, 1.5, 2.5)
        finger_position = self.finger_open + closed * (self.finger_closed - self.finger_open)
        return position, finger_position

    def _copy_targets(self) -> None:
        wp.copy(
            self.control.joint_target_q,
            self.joint_q.flatten(),
            dest_offset=self.arm.coord_start,
            count=self.arm.coord_count,
        )

    def set_targets(self, time: float) -> None:
        """Write the task-space targets before executing a captured simulation graph."""
        position, finger_position = self.target(time)
        wp.launch(
            _set_targets,
            dim=1,
            inputs=[self.target_position, self.finger_position, wp.vec3(*position), finger_position],
            device=self.device,
        )

    def solve(self) -> None:
        """Solve IK and write PD targets; this operation supports CUDA graph capture."""
        self.solver.step(self.joint_q, self.joint_q, iterations=self.ik_iterations)
        wp.launch(_set_fingers, dim=1, inputs=[self.joint_q, self.finger_position], device=self.device)
        self._copy_targets()

    def update(self, time: float) -> None:
        """Update arm and finger PD targets while leaving simulated states untouched."""
        self.set_targets(time)
        self.solve()
