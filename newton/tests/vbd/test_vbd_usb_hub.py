# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the procedural USB hub's bonded interfaces and contact exclusions."""

import unittest
from collections import Counter

import numpy as np
import warp as wp

import newton
from newton.examples.multiphysics.usb_hub import CABLE_YOUNGS_MODULUS, add_usb_hub
from newton.examples.multiphysics.usb_hub_contacts import create_hub_collision_pipeline
from newton.solvers import SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices


def _contact_records(contacts):
    """Return comparable contact geometry together with mesh feature identity."""
    count = int(contacts.soft_contact_count.numpy()[0])
    arrays = [
        contacts.soft_contact_indices.numpy()[:count],
        contacts.soft_contact_shape.numpy()[:count, None],
        np.round(contacts.soft_contact_barycentric.numpy()[:count], decimals=5),
        np.round(contacts.soft_contact_body_pos.numpy()[:count], decimals=5),
        np.round(contacts.soft_contact_normal.numpy()[:count], decimals=5),
    ]
    if contacts._soft_contact_mesh_features is not None:
        arrays.append(contacts._soft_contact_mesh_features.numpy()[:count])
    return [tuple(row) for row in np.concatenate(arrays, axis=1)]


def test_usb_hub_seam_filter(test, device):
    """Remove only bonded contacts while preserving full-surface mesh records."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    # Skip one global body so the coupled view must remap attachment body IDs.
    builder.add_body(xform=wp.transform((10.0, 0.0, 0.0), wp.quat_identity()))
    case = builder.add_body()
    builder.add_shape_box(case, hx=0.05, hy=0.05, hz=0.05)
    other = builder.add_body()
    builder.add_shape_mesh(other, mesh=newton.Mesh.create_box(0.05, 0.05, 0.05))
    builder.add_cloth_grid(
        pos=(-0.045, -0.045, 0.049),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=3,
        dim_y=3,
        cell_x=0.03,
        cell_y=0.03,
        mass=0.001,
        particle_radius=0.002,
    )
    builder.add_attachment_particle_body(0, case, body_point=builder.particle_q[0])
    builder.color()
    model = builder.finalize(device=device)
    coupled = SolverCoupled(
        model,
        [
            SolverCoupled.Entry(
                "payload",
                lambda view: SolverVBD(view, iterations=1),
                bodies=[case, other],
                particles=range(model.particle_count),
            )
        ],
    )
    view = coupled._entries["payload"].view
    test.assertLess(view.body_count, model.body_count)
    local_case = int(view.attachment_particle_body_body.numpy()[0])
    seam = np.full(model.particle_count, -1, dtype=np.int32)
    seam[0] = case
    pipeline = create_hub_collision_pipeline(view, seam, soft_contact_gap=0.005)
    raw_pipeline = newton.CollisionPipeline(
        view,
        broad_phase="explicit",
        soft_contact_gap=0.005,
        enable_rigid_soft_full_surface_contact=True,
    )
    contacts = pipeline.contacts()
    raw_contacts = raw_pipeline.contacts()
    state = view.state()
    raw_pipeline.collide(state, raw_contacts)
    raw_records = _contact_records(raw_contacts)
    shape_body = view.shape_body.numpy()
    keep = [not (shape_body[int(row[3])] == local_case and 0 in row[:3]) for row in raw_records]
    removed = [row for row, retained in zip(raw_records, keep, strict=True) if not retained]
    test.assertTrue(any(row[1] < 0 for row in removed), "The fixture must include bonded particle contacts")
    test.assertTrue(any(row[1] >= 0 for row in removed), "The fixture must include bonded edge/face contacts")
    expected = Counter(row for row, retained in zip(raw_records, keep, strict=True) if retained)
    test.assertTrue(any(shape_body[int(row[3])] == local_case for row in expected))
    test.assertTrue(any(shape_body[int(row[3])] != local_case and 0 in row[:3] for row in expected))

    pipeline.collide(state, contacts)
    test.assertEqual(Counter(_contact_records(contacts)), expected)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as captured:
            pipeline.collide(state, contacts)
        wp.capture_launch(captured.graph)
        test.assertEqual(Counter(_contact_records(contacts)), expected)


def test_usb_hub_native_attachment_reaction(test, device):
    """Transmit a plug displacement through massive tet collars into both rigids."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    poisson_ratio = 0.35
    hub = add_usb_hub(
        builder, case_position=(0.0, 0.0, 0.0), cable_length=0.04, cable_segments=8, poisson_ratio=poisson_ratio
    )
    builder.color()
    model = builder.finalize(device=device)
    test.assertTrue(np.all(model.particle_mass.numpy() > 0.0))
    test.assertEqual(model.attachment_particle_body_count, 36)
    materials = model.tet_materials.numpy()
    mu, lam = materials[:, 0], materials[:, 1]
    np.testing.assert_allclose(2.0 * mu * (1.0 + poisson_ratio), CABLE_YOUNGS_MODULUS, rtol=1e-6)
    np.testing.assert_allclose(lam / (2.0 * (lam + mu)), poisson_ratio, rtol=1e-6)

    state = model.state()
    next_state = model.state()
    body_q = state.body_q.numpy().copy()
    body_q[hub.plug_body, 2] += 0.002
    state.body_q.assign(body_q)
    solver = SolverVBD(model, iterations=15)
    control = model.control()
    for _ in range(10):
        state.clear_forces()
        solver.step(state, next_state, control, None, 0.001)
        state, next_state = next_state, state
    particle_q = state.particle_q.numpy()
    final_body_q = state.body_q.numpy()
    test.assertTrue(np.isfinite(particle_q).all())
    test.assertTrue(np.isfinite(final_body_q).all())
    test.assertLess(final_body_q[hub.plug_body, 2], body_q[hub.plug_body, 2] - 1.0e-5)
    test.assertGreater(float(np.linalg.norm(final_body_q[hub.case_body, :3])), 1.0e-6)
    test.assertGreater(float(np.linalg.norm(np.mean(particle_q[-18:] - hub.rest_particle_q[-18:], axis=0))), 1.0e-5)
    anchors = np.array(
        [
            tuple(
                wp.transform_point(
                    wp.transform(wp.vec3(final_body_q[body, :3]), wp.quat(final_body_q[body, 3:])),
                    wp.vec3(point),
                )
            )
            for body, point in zip(hub.attachment_body_indices, hub.attachment_body_points, strict=True)
        ]
    )
    errors = np.linalg.norm(particle_q[hub.attachment_particle_indices] - anchors, axis=1)
    test.assertLess(float(np.max(errors)), 0.001)


class TestVBDUSBHub(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


for function in (test_usb_hub_seam_filter, test_usb_hub_native_attachment_reaction):
    add_function_test(TestVBDUSBHub, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
