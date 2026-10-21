# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exclude bonded USB hub collars from rigid-soft collision response."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton


@wp.kernel
def _compact_unbonded_contacts(
    seam_body: wp.array[int],
    shape_body: wp.array[int],
    count: wp.array[int],
    particle: wp.array[int],
    indices: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    shape: wp.array[int],
    body_pos: wp.array[wp.vec3],
    body_vel: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    mesh_features: wp.array[wp.vec3i],
    mesh_params: wp.array[float],
):
    # Serial, in-place compaction preserves ordering without scratch or host reads.
    # The small cable has only a few hundred active contacts.
    write = int(0)
    limit = wp.min(count[0], indices.shape[0])
    for read in range(limit):
        corners = indices[read]
        contact_body = shape_body[shape[read]]
        bonded = False
        for corner in range(3):
            p = corners[corner]
            if p >= 0 and contact_body >= 0:
                if seam_body[p] == contact_body:
                    bonded = True
        if not bonded:
            if write != read:
                particle[write] = particle[read]
                indices[write] = corners
                barycentric[write] = barycentric[read]
                shape[write] = shape[read]
                body_pos[write] = body_pos[read]
                body_vel[write] = body_vel[read]
                normal[write] = normal[read]
                if mesh_features:
                    mesh_features[write] = mesh_features[read]
                    mesh_params[write] = mesh_params[read]
            write += 1
    count[0] = write


class USBHubCollisionPipeline(newton.CollisionPipeline):
    """Generate full-surface contacts while excluding each bonded collar.

    ``seam_body`` contains a rigid body index for attached particles and ``-1``
    elsewhere, or is omitted to infer the collars from active attachments.
    Body indices are resolved from the model's attachment rows, so the pipeline
    also supports coupled views that compact rigid body indices.
    A particle, edge, or face touching an attached particle is
    excluded only against that particle's own rigid body. The rest of the cable
    can still contact the case and plug, and every particle can contact the
    table and gripper. This example-specific pipeline is not differentiable.
    """

    def __init__(
        self,
        model: newton.Model,
        seam_body: np.ndarray | None = None,
        *,
        soft_contact_gap: float = 0.002,
        soft_contact_max: int | None = None,
        broad_phase: str = "explicit",
        contact_matching: str = "disabled",
    ):
        if model.requires_grad:
            raise ValueError("USB hub seam filtering does not support differentiable contacts")
        if seam_body is not None:
            seam_body = np.asarray(seam_body, dtype=np.int32)
            if seam_body.shape != (model.particle_count,):
                raise ValueError("seam_body must contain one body index per model particle")
        resolved = np.full(model.particle_count, -1, dtype=np.int32)
        for particle, body, enabled in zip(
            model.attachment_particle_body_particle.numpy(),
            model.attachment_particle_body_body.numpy(),
            model.attachment_particle_body_enabled.numpy(),
            strict=True,
        ):
            if enabled and (seam_body is None or seam_body[particle] >= 0):
                if resolved[particle] >= 0 and resolved[particle] != body:
                    raise ValueError("USB hub seam particles must attach to exactly one rigid body")
                resolved[particle] = body
        super().__init__(
            model,
            broad_phase=broad_phase,
            soft_contact_gap=soft_contact_gap,
            soft_contact_max=soft_contact_max,
            enable_rigid_soft_full_surface_contact=True,
            contact_matching=contact_matching,
        )
        self.seam_body = wp.array(resolved, dtype=int, device=model.device)

    def collide(
        self,
        state: newton.State,
        contacts: newton.Contacts,
        *,
        soft_contact_margin: float | None = None,
        soft_self_contact: bool = False,
        dt: float | None = None,
    ):
        super().collide(
            state,
            contacts,
            soft_contact_margin=soft_contact_margin,
            soft_self_contact=soft_self_contact,
            dt=dt,
        )
        if contacts.soft_contact_max:
            wp.launch(
                _compact_unbonded_contacts,
                dim=1,
                inputs=[
                    self.seam_body,
                    self.model.shape_body,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    # Mesh eligibility is indexed by contact row, so retain its
                    # metadata when rows move alongside public contact geometry.
                    contacts._soft_contact_mesh_features,
                    contacts._soft_contact_mesh_params,
                ],
                device=self.model.device,
            )


def create_hub_collision_pipeline(
    model: newton.Model,
    seam_body: np.ndarray | None = None,
    *,
    soft_contact_gap: float = 0.002,
    soft_contact_max: int | None = None,
    broad_phase: str = "explicit",
    contact_matching: str = "disabled",
) -> USBHubCollisionPipeline:
    """Construct the VBD proxy pipeline for a procedurally assembled USB hub."""
    return USBHubCollisionPipeline(
        model,
        seam_body,
        soft_contact_gap=soft_contact_gap,
        soft_contact_max=soft_contact_max,
        broad_phase=broad_phase,
        contact_matching=contact_matching,
    )
