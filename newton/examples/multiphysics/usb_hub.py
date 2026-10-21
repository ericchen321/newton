# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Procedural rigid housing and plug joined by a tetrahedral USB cable."""

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

# HAG4R-Control/assets/objects/usb_hub_revised/heterogeneous_params.npz:
# part_indices == 1 (black_flexible_usb_cable), also documented in that
# asset's inferred_params.json. Only these material values are reused.
CABLE_YOUNGS_MODULUS = 5.0e6
CABLE_POISSON_RATIO = 0.49
CABLE_DENSITY = 1100.0


@dataclass
class USBHub:
    """Indices and rest geometry of one procedurally assembled USB hub."""

    case_body: int
    plug_body: int
    case_shape: int
    plug_shapes: tuple[int, int]
    body_indices: list[int]
    joint_indices: list[int]
    shape_indices: list[int]
    particle_indices: np.ndarray
    tet_indices: np.ndarray
    rest_particle_q: np.ndarray
    attachment_indices: np.ndarray
    attachment_particle_indices: np.ndarray
    attachment_body_indices: np.ndarray
    attachment_body_points: np.ndarray
    seam_body: np.ndarray


def _cable_mesh(length: float, radius: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Extrude an octagonal disk with conforming triangular-prism splits."""
    angles = np.arange(8) * (2.0 * np.pi / 8.0)
    disk = np.concatenate((np.zeros((1, 2)), radius * np.column_stack((np.cos(angles), np.sin(angles)))))
    vertices = np.zeros((segments + 1, 9, 3), dtype=np.float64)
    vertices[:, :, 0] = np.linspace(0.0, length, segments + 1)[:, None]
    vertices[:, :, 1:] = disk
    vertices = vertices.reshape(-1, 3)

    tetrahedra = []
    for segment in range(segments):
        for side in range(8):
            # Sorting the disk vertices selects the same diagonal on every
            # shared rectangular face, including the wraparound eighth wedge.
            a, b, c = sorted((0, side + 1, (side + 1) % 8 + 1))
            a, b, c = (vertex + 9 * segment for vertex in (a, b, c))
            tetrahedra.extend(((a, b, c, c + 9), (a, b, b + 9, c + 9), (a, a + 9, b + 9, c + 9)))
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    tetrahedron_vertices = vertices[tetrahedra]
    determinants = np.linalg.det(tetrahedron_vertices[:, 1:] - tetrahedron_vertices[:, :1])
    inverted = determinants < 0.0
    tetrahedra[inverted, :2] = tetrahedra[inverted, 1::-1]
    return vertices, tetrahedra


def add_usb_hub(
    builder: newton.ModelBuilder,
    *,
    case_position: tuple[float, float, float] = (0.5, 0.0, 0.265),
    cable_length: float = 0.22,
    cable_radius: float = 0.004,
    cable_segments: int = 44,
    poisson_ratio: float = CABLE_POISSON_RATIO,
    attachment_stiffness: float = 2.0e4,
    attachment_damping: float = 0.2,
) -> USBHub:
    """Build two free rigid bodies and a cable with compliant end welds.

    The case is a 100 by 40 by 30 mm box. The cable exits its positive X
    face; the plug consists of a plastic box sleeve and a metal box tip.
    Two complete cable cross-sections are embedded in each rigid part and
    attached at their individual body-local rest positions. Their spatial
    separation transmits bending and twisting moments as well as tension.
    Every cable particle retains its physical mass.

    The case's effective density includes its hollow interior. The cable
    uses the revised HAG4R-Control cable's 5 MPa Young's modulus and
    1100 kg/m³ density. Its Poisson ratio defaults to the reference's 0.49
    but can be overridden; all geometry is generated here.
    ``seam_body`` is indexed locally within this cable and identifies the
    attached rigid body, or -1. Contact filtering can use it to avoid
    repulsion between each embedded collar and its own attachment body.
    """
    if cable_segments < 4:
        raise ValueError("The cable needs at least four segments for two disjoint attachment collars.")
    if cable_length <= 0.0 or cable_radius <= 0.0:
        raise ValueError("Cable length and radius must be positive.")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("Poisson ratio must lie strictly between -1 and 0.5.")
    spacing = cable_length / cable_segments
    if spacing > 0.01:
        raise ValueError("Cable segments must be at most 10 mm long to fit inside the plug sleeve.")

    case_position = np.asarray(case_position, dtype=np.float64)
    cfg_case = newton.ModelBuilder.ShapeConfig(density=550.0, mu=0.8, ke=1.0e5, kd=100.0, gap=0.001)
    cfg_sleeve = newton.ModelBuilder.ShapeConfig(density=1100.0, mu=0.6, ke=1.0e5, kd=100.0, gap=0.001)
    cfg_tip = newton.ModelBuilder.ShapeConfig(density=7850.0, mu=0.4, ke=1.0e5, kd=100.0, gap=0.001)

    case_body = builder.add_body(xform=wp.transform(case_position, wp.quat_identity()), label="usb_hub_case")
    case_joint = builder.joint_count - 1
    case_shape = builder.add_shape_box(
        case_body,
        hx=0.05,
        hy=0.02,
        hz=0.015,
        cfg=cfg_case,
        color=(0.30, 0.36, 0.40),
        label="usb_hub_case",
    )

    cable_start = case_position + np.array((0.05 - spacing, 0.0, 0.0))
    # The last two rings lie at the sleeve's rear face and one cell inside.
    plug_position = cable_start + np.array((cable_length - spacing + 0.012, 0.0, 0.0))
    plug_body = builder.add_body(xform=wp.transform(plug_position, wp.quat_identity()), label="usb_hub_plug")
    plug_joint = builder.joint_count - 1
    sleeve_shape = builder.add_shape_box(
        plug_body,
        hx=0.012,
        hy=0.009,
        hz=0.006,
        cfg=cfg_sleeve,
        color=(0.055, 0.06, 0.065),
        label="usb_hub_plug_sleeve",
    )
    tip_shape = builder.add_shape_box(
        plug_body,
        xform=wp.transform((0.018, 0.0, 0.0), wp.quat_identity()),
        hx=0.006,
        hy=0.006,
        hz=0.0022,
        cfg=cfg_tip,
        color=(0.65, 0.68, 0.72),
        label="usb_hub_male_tip",
    )

    vertices, tetrahedra = _cable_mesh(cable_length, cable_radius, cable_segments)
    particle_start = builder.particle_count
    mu = CABLE_YOUNGS_MODULUS / (2.0 * (1.0 + poisson_ratio))
    lam = CABLE_YOUNGS_MODULUS * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    builder.add_soft_mesh(
        pos=cable_start,
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=vertices,
        indices=tetrahedra.flatten(),
        density=CABLE_DENSITY,
        k_mu=mu,
        k_lambda=lam,
        k_damp=500.0,
        particle_radius=0.0005,
        color=(0.035, 0.04, 0.045),
        validate_mesh=True,
        label="usb_hub_cable",
    )

    seam_body = np.full(len(vertices), -1, dtype=np.int32)
    seam_body[:18] = case_body
    seam_body[-18:] = plug_body
    attachment_particles = np.flatnonzero(seam_body >= 0).astype(np.int32) + particle_start
    attachment_bodies = seam_body[seam_body >= 0]
    attachment_points = []
    attachment_indices = []
    for particle, body in zip(attachment_particles, attachment_bodies, strict=True):
        body_point = wp.transform_point(wp.transform_inverse(builder.body_q[body]), builder.particle_q[particle])
        attachment_points.append(tuple(body_point))
        attachment_indices.append(
            builder.add_attachment_particle_body(
                int(particle),
                int(body),
                body_point=body_point,
                stiffness=attachment_stiffness,
                damping=attachment_damping,
            )
        )

    return USBHub(
        case_body=case_body,
        plug_body=plug_body,
        case_shape=case_shape,
        plug_shapes=(sleeve_shape, tip_shape),
        body_indices=[case_body, plug_body],
        joint_indices=[case_joint, plug_joint],
        shape_indices=[case_shape, sleeve_shape, tip_shape],
        particle_indices=np.arange(particle_start, builder.particle_count, dtype=np.int32),
        tet_indices=tetrahedra + particle_start,
        rest_particle_q=np.asarray(builder.particle_q[particle_start:], dtype=np.float32),
        attachment_indices=np.asarray(attachment_indices, dtype=np.int32),
        attachment_particle_indices=attachment_particles,
        attachment_body_indices=attachment_bodies,
        attachment_body_points=np.asarray(attachment_points, dtype=np.float32),
        seam_body=seam_body,
    )
