.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

USB hub pickup with rigid-soft attachments
=========================================

The ``usb_hub_pickup`` example uses one Franka FR3 arm to approach a USB
hub, close its parallel fingers around the housing, lift by 17 cm, and hold.
The housing and male connector are dynamic rigid bodies. The cable is a
volumetric tetrahedral FEM mesh, rather than a chain of rigid capsules.
The gripper has no attachment to the payload: friction must support its weight.

Run the full episode and its physical checks:

.. code-block:: console

   uv run --extra examples -m newton.examples usb_hub_pickup --viewer null --test

Omit ``--viewer null`` to open the interactive viewer. The default episode
contains 720 frames at 60 Hz. A shorter test must still include at least 540
frames to exercise the lift and a sustained hold.

Geometry and material
---------------------

All USB hub geometry is constructed procedurally in
``newton/examples/multiphysics/usb_hub.py``. The 100 × 40 × 30 mm housing is a
single box. The male connector is a separate rigid body with a plastic sleeve
box and a metal tip box. The cable has an octagonal cross-section, a center
vertex in each slice, and conforming tetrahedra between slices. The example's
140 mm long, 6 mm diameter cable contains 261 vertices and 672 tetrahedra.

Only material constants come from HAG4R-Control's revised USB hub:
``assets/objects/usb_hub_revised/inferred_params.json`` and
``heterogeneous_params.npz``, cable part index 1. No mesh, tet connectivity,
asset file, or simulated motion from that repository is loaded or copied.

* Young's modulus: 5,000,000 Pa.
* Density: 1100 kg/m³.

The reference uses Poisson's ratio 0.49. This example instead uses 0.35 as
an effective composite-cable assumption, retaining the required Young's
modulus exactly. This is not a calibrated reproduction of the reference
asset. Short cable release tests showed smaller attachment errors and
extension with this choice and a smaller timestep. Those tests did not
isolate volumetric locking from iterative error or establish mesh convergence.
Use ``--cable-poisson-ratio 0.49`` to select the reference ratio, but expect
to revisit solver resolution for that nearly incompressible material.

The builder receives Lamé parameters derived from the selected constants:
``mu = E / (2 * (1 + nu))`` and
``lambda = E * nu / ((1 + nu) * (1 - 2 * nu))``.
The tetrahedra supply the cable's elasticity; its exterior triangles are
used for visualization and contact without adding membrane stiffness.

Attachment and contact choices
------------------------------

Two complete cross-sections at each end are attached to the corresponding
rigid body with :meth:`newton.ModelBuilder.add_attachment_particle_body`.
Every vertex has its own body-local rest anchor. This distributed connection
transfers tension, bending moments, and torsion. The attachment is compliant,
with stiffness 20,000 N/m per vertex and damping 0.2 N·s/m; every vertex
retains its physical mass.

The housing, plug, and cable are owned by one :class:`newton.solvers.SolverVBD`.
It solves attachment forces on both particles and rigid bodies. A MuJoCo
solver owns the robot. The experimental
``newton.solvers.experimental.coupled.SolverCoupledProxy`` interface exposes
the hand and finger bodies and their internal joints to VBD, and feeds
contact reactions back to MuJoCo. These experimental coupling APIs may change.

Newton's :class:`newton.CollisionPipeline` handles rigid box contact and
full-surface cable contact, including triangle interiors and edges.
The example compacts the contact buffer to remove only contacts between an
embedded attachment collar and its own rigid body. Contact with the table,
gripper, opposite connector, and the rest of the cable remains available.
The filter derives body IDs from the destination solver's attachment rows
because proxy model views compact their body indices.

Rigid contact uses VBD's compliant augmented Lagrangian formulation.
Contact history and contact matching preserve static friction across steps.
Rigid-soft contact uses normal penalty forces and friction. Detection is
refreshed each simulation substep. The defaults use 24 substeps per 60 Hz
frame, 80 VBD iterations and two proxy coupling iterations.
DAT and the optional log barrier are
disabled: DAT requires a solver-owned collision pipeline, while the current
proxy contact-feedback path uses an external destination pipeline.
This configuration does not provide a continuous collision detection or
penetration-free guarantee. Cable self-contact is disabled for this short,
initially straight, single-cable pickup.

Validation and rendering
------------------------

The episode checks positive tet volume, finite state, attachment error below
0.5 mm, cable centerline segment stretch below 2%, total centerline extension
below 1%, table support, and a sustained housing lift of at least 14.5 cm.
The plug and the entire cable must also remain clear of the table during
the hold. These are task-level checks, not a numerical convergence guarantee.
Its recording contains measured body transforms, cable vertices, surface
triangles, and visual geometry; Blender replays this simulation.

With the default settings and a warm kernel cache, the complete 720-frame
episode took 622 seconds on an NVIDIA RTX 4000 SFF Ada GPU. The example test
uses 540 frames, retaining the complete approach and lift plus a hold;
its estimated cost is 467 seconds on that device. It has its own test class
and a 1200-second timeout to accommodate initialization and compilation.
Shorter smoke tests would miss failures during the lift and sustained grasp.
The focused attachment/contact tests cover CPU and CUDA in about 1.2 seconds
with a warm cache.

.. code-block:: console

   uv run --extra examples -m newton.examples usb_hub_pickup \
       --viewer null --test --record-path outputs/usb_hub/episode.npz

The accompanying JSON file stores per-frame lift, attachment error, and
tet volume diagnostics. The NPZ also records material and solver settings.
Rendering requires a separate Blender installation with Cycles OptiX support,
an NVIDIA GPU, and FFmpeg on ``PATH``. Blender and FFmpeg are not needed to run
the Newton example:

.. code-block:: console

   uv run --no-sync blender --background \
       --python scripts/render_usb_hub_pickup.py -- \
       --recording outputs/usb_hub/episode.npz \
       --output outputs/usb_hub/render --camera cinematic --samples 24 --fps 24

The renderer exports a 1280 × 720 H.264 video. It interpolates the recorded
60 Hz poses and cable vertices onto the 24 Hz video timeline without
retiming the episode. Cosmetic bevels, ports, materials, and studio lighting
do not alter the simulated contact geometry.
