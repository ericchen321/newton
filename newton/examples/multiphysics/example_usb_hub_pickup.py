# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lift a procedural rigid and tetrahedral USB hub with a Franka arm.

The housing, plug, and cable share one VBD solve. Compliant particle-body
attachments join two cross-sections at each cable end to the rigid hardware.
MuJoCo integrates the arm; experimental proxy coupling returns contact impulses
from the VBD gripper proxies to the arm. The grasp uses friction, with no
attachment between the gripper and the hub.

Run with ``python -m newton.examples usb_hub_pickup --test --viewer null``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.examples.multiphysics.usb_hub import CABLE_DENSITY, CABLE_YOUNGS_MODULUS, add_usb_hub
from newton.examples.multiphysics.usb_hub_arm import PickupController, add_franka
from newton.examples.multiphysics.usb_hub_contacts import create_hub_collision_pipeline
from newton.solvers import SolverMuJoCo, SolverVBD


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        if args.substeps < 1:
            raise ValueError("substeps must be positive")
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.surface_z = 0.25
        self.case_center = np.array((0.5, 0.0, 0.265), dtype=np.float32)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder)
        builder.rigid_gap = 0.002
        self.arm = add_franka(builder, base_z=self.surface_z)
        self.hub = add_usb_hub(
            builder,
            case_position=tuple(self.case_center),
            cable_length=0.14,
            cable_radius=0.003,
            cable_segments=28,
            poisson_ratio=args.cable_poisson_ratio,
        )
        builder.add_ground_plane(
            height=self.surface_z,
            cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=10.0, mu=0.9, gap=0.002),
            label="work_surface",
        )
        builder.color()
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 0.01
        self.model.soft_contact_mu = 0.9
        self.control = self.model.control()
        self.controller = PickupController(
            self.model, self.control, self.arm, case_center=tuple(self.case_center), lift_height=0.17
        )
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit", soft_contact_gap=0.001)
        self.contacts = self.collision_pipeline.contacts()
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="arm",
                    solver=lambda model: SolverMuJoCo(
                        model=model,
                        use_mujoco_contacts=False,
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        iterations=20,
                        njmax=512,
                        nconmax=128,
                    ),
                    bodies=self.arm.bodies,
                    joints=self.arm.joints,
                ),
                SolverCoupledProxy.Entry(
                    name="hub",
                    solver=lambda model: SolverVBD(
                        model,
                        iterations=args.vbd_iterations,
                        rigid_contact_history=True,
                        friction_epsilon=0.001,
                        particle_enable_self_contact=False,
                        rigid_soft_enable_dat=False,
                        rigid_body_particle_contact_buffer_size=2048,
                        rigid_joint_linear_ke=2.0e7,
                        rigid_joint_angular_ke=2.0e6,
                    ),
                    bodies=self.hub.body_indices,
                    joints=self.hub.joint_indices,
                    particles=self.hub.particle_indices,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="arm",
                        destination="hub",
                        bodies=self.arm.gripper_bodies,
                        joints=self.arm.gripper_joints,
                        mass_scale=1.0,
                        mode="lagged",
                        proxy_relaxation=0.5,
                        collision_pipeline=lambda model: create_hub_collision_pipeline(
                            model, self.hub.seam_body, contact_matching="latest"
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=args.proxy_iterations,
            ),
        )
        self.solver.prepare_contacts(self.contacts)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.25, -1.25, 0.95), pitch=-22.0, yaw=125.0)
        self.graph = None
        if args.graph_capture and self.model.device.is_cuda:
            if self.sim_substeps % 2:
                raise ValueError("CUDA graph capture requires an even number of substeps")
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        self.frames = []
        self.diagnostics = []
        self._rest_tets = self.model.tet_indices.numpy()
        self._rest_volumes = self._tet_determinants(self.hub.rest_particle_q)
        self._rest_centerline = self.hub.rest_particle_q.reshape(-1, 9, 3).mean(axis=1)
        self._rest_segment_lengths = np.linalg.norm(np.diff(self._rest_centerline, axis=0), axis=1)
        self._rigid_corners = {self.hub.case_body: [], self.hub.plug_body: []}
        scales = self.model.shape_scale.numpy()
        transforms = self.model.shape_transform.numpy()
        for shape in self.hub.shape_indices:
            body = self.hub.case_body if shape == self.hub.case_shape else self.hub.plug_body
            local_pose = wp.transform(wp.vec3(*transforms[shape, :3]), wp.quat(*transforms[shape, 3:]))
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        corner = scales[shape] * (x, y, z)
                        self._rigid_corners[body].append(wp.transform_point(local_pose, wp.vec3(*corner)))
        self._record_frame()

    def simulate(self):
        self.controller.solve()
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.controller.set_targets(self.sim_time)
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self._record_frame()
        if self.args.record_path and len(self.frames) == self.args.num_frames + 1:
            self.save_recording(self.args.record_path)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _tet_determinants(self, particles):
        vertices = particles[self._rest_tets]
        edges = vertices[:, 1:] - vertices[:, :1]
        return np.linalg.det(edges)

    def _record_frame(self):
        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()
        errors = []
        for particle, body, point in zip(
            self.hub.attachment_particle_indices,
            self.hub.attachment_body_indices,
            self.hub.attachment_body_points,
            strict=True,
        ):
            pose = wp.transform(wp.vec3(*body_q[body, :3]), wp.quat(*body_q[body, 3:]))
            anchor = np.array(wp.transform_point(pose, wp.vec3(*point)))
            errors.append(np.linalg.norm(particle_q[particle] - anchor))
        volume_ratio = self._tet_determinants(particle_q) / self._rest_volumes
        centerline = particle_q[self.hub.particle_indices].reshape(-1, 9, 3).mean(axis=1)
        segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        bottom = {}
        for body, corners in self._rigid_corners.items():
            pose = wp.transform(wp.vec3(*body_q[body, :3]), wp.quat(*body_q[body, 3:]))
            bottom[body] = min(wp.transform_point(pose, corner)[2] for corner in corners)
        self.diagnostics.append(
            {
                "time": self.sim_time,
                "case_z": float(body_q[self.hub.case_body, 2]),
                "case_bottom_z": float(bottom[self.hub.case_body]),
                "plug_z": float(body_q[self.hub.plug_body, 2]),
                "plug_bottom_z": float(bottom[self.hub.plug_body]),
                "cable_bottom_z": float(particle_q[:, 2].min() - 0.0005),
                "max_attachment_error": float(max(errors)),
                "minimum_volume_ratio": float(volume_ratio.min()),
                "maximum_volume_ratio": float(volume_ratio.max()),
                "cable_arc_length": float(segment_lengths.sum()),
                "maximum_axial_stretch": float((segment_lengths / self._rest_segment_lengths).max()),
            }
        )
        if self.args.record_path:
            # CPU .numpy() exposes the reusable simulation buffers.
            self.frames.append((body_q.copy(), particle_q.copy()))

    def test_post_step(self):
        """Check that the assembly stays joined and the tetrahedra remain valid."""
        latest = self.diagnostics[-1]
        assert np.isfinite(list(latest.values())).all(), f"Non-finite state: {latest}"
        assert latest["minimum_volume_ratio"] > 0.0, f"Inverted cable tetrahedron: {latest}"
        assert latest["max_attachment_error"] < 0.0005, f"Cable endpoint detached: {latest}"
        assert latest["maximum_axial_stretch"] < 1.02, f"Excessive cable centerline stretch: {latest}"
        assert latest["cable_arc_length"] < 1.01 * self._rest_segment_lengths.sum(), (
            f"Excessive cable extension: {latest}"
        )
        assert latest["case_bottom_z"] > self.surface_z - 0.002, f"Housing penetrated the table: {latest}"
        assert latest["plug_bottom_z"] > self.surface_z - 0.002, f"Plug penetrated the table: {latest}"
        assert latest["cable_bottom_z"] > self.surface_z - 0.002, f"Cable penetrated the table: {latest}"

    def test_final(self):
        """Verify a sustained frictional pickup over the complete episode."""
        self.test_post_step()
        assert self.sim_time >= 9.0, "Run at least 540 frames to verify lifting and holding"
        hold = [row for row in self.diagnostics if row["time"] >= 8.0]
        lift = min(row["case_z"] for row in hold) - float(self.case_center[2])
        assert lift > 0.145, f"Housing did not remain lifted: minimum lift {lift:.4f} m"
        assert min(row["plug_bottom_z"] for row in hold) > self.surface_z + 0.003, "Plug did not clear the table"
        assert min(row["cable_bottom_z"] for row in hold) > self.surface_z + 0.003, "Cable did not clear the table"
        assert max(row["max_attachment_error"] for row in self.diagnostics) < 0.0005
        assert max(row["maximum_axial_stretch"] for row in self.diagnostics) < 1.02
        assert max(row["cable_arc_length"] for row in self.diagnostics) < 1.01 * self._rest_segment_lengths.sum()
        assert min(row["minimum_volume_ratio"] for row in self.diagnostics) > 0.0
        for key in ("case_bottom_z", "plug_bottom_z", "cable_bottom_z"):
            assert min(row[key] for row in self.diagnostics) > self.surface_z - 0.002
        assert np.isfinite(self.state_0.body_qd.numpy()).all()
        assert np.isfinite(self.state_0.particle_qd.numpy()).all()

    def save_recording(self, path):
        """Save simulated poses and deforming surface vertices for offline rendering."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "body_q": np.stack([frame[0] for frame in self.frames]),
            "particle_q": np.stack([frame[1] for frame in self.frames]),
            "time": np.asarray([row["time"] for row in self.diagnostics], dtype=np.float64),
            "tri_indices": self.model.tri_indices.numpy(),
            "shape_body": self.model.shape_body.numpy(),
            "shape_type": self.model.shape_type.numpy(),
            "shape_type_name": np.asarray([newton.GeoType(value).name for value in self.model.shape_type.numpy()]),
            "shape_scale": self.model.shape_scale.numpy(),
            "shape_transform": self.model.shape_transform.numpy(),
            "shape_flags": self.model.shape_flags.numpy(),
            "shape_color": self.model.shape_color.numpy(),
            "shape_label": np.asarray(self.model.shape_label),
            "body_label": np.asarray(self.model.body_label),
            "fps": self.fps,
            "surface_z": self.surface_z,
            "case_body": self.hub.case_body,
            "plug_body": self.hub.plug_body,
            "case_shape": self.hub.case_shape,
            "plug_shapes": np.asarray(self.hub.plug_shapes, dtype=np.int32),
            "cable_particles": self.hub.particle_indices,
            "cable_youngs_modulus": CABLE_YOUNGS_MODULUS,
            "cable_poisson_ratio": self.args.cable_poisson_ratio,
            "cable_density": CABLE_DENSITY,
            "substeps": self.sim_substeps,
            "vbd_iterations": self.args.vbd_iterations,
            "proxy_iterations": self.args.proxy_iterations,
        }
        for index, source in enumerate(self.model.shape_source):
            if isinstance(source, newton.Mesh):
                data[f"mesh_{index}_vertices"] = np.asarray(source.vertices)
                data[f"mesh_{index}_indices"] = np.asarray(source.indices)
        np.savez_compressed(path, **data)
        path.with_suffix(".json").write_text(json.dumps(self.diagnostics, indent=2) + "\n")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=720)
        parser.add_argument("--substeps", type=int, default=24)
        parser.add_argument("--vbd-iterations", type=int, default=80)
        parser.add_argument("--proxy-iterations", type=int, default=2)
        parser.add_argument(
            "--cable-poisson-ratio",
            type=float,
            default=0.35,
            help="Effective composite-cable Poisson ratio; the HAG4R reference uses 0.49.",
        )
        parser.add_argument("--graph-capture", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--record-path", type=str, default=None, help="Save an NPZ simulation recording.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
