# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render recorded Newton USB pickup states with Blender Cycles.

Run using Blender's bundled Python, for example::

    uv run --no-sync blender --background --python scripts/render_usb_hub_pickup.py -- \
        --recording outputs/usb_hub_pickup/trajectory.npz \
        --output outputs/usb_hub_pickup/render --camera closeup

The robot and cable geometry and all motion come from the recording. Ports,
lettering, edge bevels, the bench and lighting are rendering-only details.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


def material(name, color, *, metal=0.0, roughness=0.4, texture=0.0):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metal
    shader.inputs["Roughness"].default_value = roughness
    if texture:
        nodes = result.node_tree.nodes
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 170.0
        noise.inputs["Detail"].default_value = 2.0
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.10
        bump.inputs["Distance"].default_value = texture
        result.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        result.node_tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return result


def set_pose(obj, pose):
    obj.location = pose[:3]
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Quaternion((pose[6], pose[3], pose[4], pose[5]))


def box(name, half_size, location, surface, *, parent=None, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    obj.name = name
    obj.scale = half_size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.parent = parent
    obj.location = location
    obj.data.materials.append(surface)
    if bevel:
        modifier = obj.modifiers.new("Machined edge radius", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
        modifier = obj.modifiers.new("Corner normals", "WEIGHTED_NORMAL")
        modifier.keep_sharp = True
    return obj


def look_at(obj, position):
    obj.rotation_euler = (Vector(position) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_light(name, position, target, energy, size, color):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    look_at(obj, target)


def add_hub_details(case, plug, surfaces):
    """Small cosmetic features follow the recorded rigid-body poses."""
    for port, x in enumerate((-0.032, 0.0, 0.032)):
        box(f"Port {port + 1} socket", (0.009, 0.00035, 0.0037), (x, -0.02015, 0.0), surfaces["black"], parent=case)
        for z in (-0.0041, 0.0041):
            box(
                f"Port {port + 1} rim",
                (0.0094, 0.0004, 0.0004),
                (x, -0.0203, z),
                surfaces["metal"],
                parent=case,
                bevel=0.0002,
            )
        for side in (-1, 1):
            box(
                f"Port {port + 1} side",
                (0.0004, 0.0004, 0.0041),
                (x + side * 0.0094, -0.0203, 0.0),
                surfaces["metal"],
                parent=case,
                bevel=0.0002,
            )
        box(
            f"Port {port + 1} tongue",
            (0.0072, 0.00042, 0.0008),
            (x, -0.0204, -0.0012),
            surfaces["blue"],
            parent=case,
        )

    text = bpy.data.curves.new("Housing lettering", "FONT")
    text.body = "N E W T O N"
    text.size = 0.0055
    text.extrude = 0.0
    obj = bpy.data.objects.new("Housing lettering", text)
    bpy.context.collection.objects.link(obj)
    obj.parent = case
    obj.location = (-0.035, 0.003, 0.01512)
    obj.data.materials.append(surfaces["label"])

    for y in (-0.0027, 0.0027):
        box("USB-A retention opening", (0.0017, 0.001, 0.00004), (0.018, y, 0.00225), surfaces["black"], parent=plug)
    box("USB-A terminal insert", (0.00004, 0.0047, 0.0013), (0.02405, 0.0, 0.0), surfaces["black"], parent=plug)


def create_shape(data, index, bodies, surfaces):
    kind = str(data["shape_type_name"][index])
    body = int(data["shape_body"][index])
    name = str(data["shape_label"][index])
    scale = data["shape_scale"][index]
    if kind in ("PLANE", "NONE"):
        return
    case_shape = int(data["case_shape"])
    plug_shapes = data["plug_shapes"]
    if index == case_shape:
        surface = surfaces["housing"]
    elif index == plug_shapes[0]:
        surface = surfaces["rubber"]
    elif index == plug_shapes[1]:
        surface = surfaces["metal"]
    else:
        color = np.clip(data["shape_color"][index], 0.0, 1.0)
        # Newton's viewer colors are linear RGB, as are Blender shader colors.
        body_label = str(data["body_label"][body]) if body >= 0 else ""
        if body_label.endswith(("fr3_hand", "fr3_leftfinger", "fr3_rightfinger")):
            # Use the black finish of the same Franka hand geometry.
            color = (0.025, 0.029, 0.034) if np.mean(color) > 0.7 else (0.085, 0.095, 0.11)
            surface = material(name, color, metal=0.15, roughness=0.36)
        else:
            surface = material(name, color * 0.75, metal=0.08, roughness=0.34)
    if kind == "BOX":
        bevel = 0.0009 if index == case_shape else min(float(np.min(scale)) * 0.12, 0.0006)
        obj = box(name, scale, (0, 0, 0), surface, bevel=bevel)
    elif kind in ("MESH", "CONVEX_MESH"):
        vertices = data[f"mesh_{index}_vertices"] * scale
        indices = data[f"mesh_{index}_indices"].reshape(-1, 3)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices.tolist(), [], indices.tolist())
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(surface)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    elif kind in ("SPHERE", "ELLIPSOID"):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
        obj = bpy.context.object
        obj.scale = scale if kind == "ELLIPSOID" else (scale[0],) * 3
        obj.data.materials.append(surface)
    elif kind in ("CYLINDER", "CAPSULE"):
        if kind == "CAPSULE":
            raise ValueError(f"Visible capsule {name!r} needs mesh serialization before rendering.")
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=float(scale[0]), depth=float(2.0 * scale[1]))
        obj = bpy.context.object
        obj.data.materials.append(surface)
    else:
        raise ValueError(f"Unsupported visible recorded geometry {kind}: {name}")
    obj.name = name
    if body >= 0:
        obj.parent = bodies[body]
    set_pose(obj, data["shape_transform"][index])


def build_scene(data, args):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    devices = [device for device in preferences.devices if device.type == "OPTIX"]
    if not devices:
        raise RuntimeError("No OptiX GPU is available for the requested Cycles render.")
    for device in preferences.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = True
    scene.cycles.adaptive_threshold = 0.025
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.fps = args.fps
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.7
    scene.world.color = (0.16, 0.17, 0.19)

    surfaces = {
        "housing": material(
            "Anodized graphite aluminum", (0.19, 0.235, 0.26), metal=0.82, roughness=0.29, texture=0.000035
        ),
        "metal": material("Brushed nickel USB shield", (0.52, 0.56, 0.61), metal=0.94, roughness=0.23),
        "rubber": material("Cable and overmold rubber", (0.017, 0.021, 0.025), roughness=0.42, texture=0.000025),
        "black": material("Socket interior", (0.004, 0.005, 0.007), roughness=0.58),
        "blue": material("USB 3 polymer tongue", (0.025, 0.13, 0.38), roughness=0.30),
        "label": material("Laser etched lettering", (0.67, 0.72, 0.74), metal=0.2, roughness=0.5),
    }
    bodies = []
    for index, label in enumerate(data["body_label"]):
        obj = bpy.data.objects.new(str(label), None)
        bpy.context.collection.objects.link(obj)
        set_pose(obj, data["body_q"][0, index])
        bodies.append(obj)
    for index, flags in enumerate(data["shape_flags"]):
        if int(flags) & 1:
            create_shape(data, index, bodies, surfaces)
    add_hub_details(bodies[int(data["case_body"])], bodies[int(data["plug_body"])], surfaces)

    mesh = bpy.data.meshes.new("Recorded tetrahedral cable surface")
    mesh.from_pydata(data["particle_q"][0].tolist(), [], data["tri_indices"].tolist())
    mesh.update()
    cable = bpy.data.objects.new("Deforming FEM cable", mesh)
    bpy.context.collection.objects.link(cable)
    cable.data.materials.append(surfaces["rubber"])
    for polygon in cable.data.polygons:
        polygon.use_smooth = True

    surface_z = float(data["surface_z"])
    bench = material("Warm matte workbench", (0.48, 0.46, 0.415), roughness=0.55, texture=0.00012)
    box("Studio workbench", (0.9, 0.65, 0.02), (0.35, 0, surface_z - 0.02), bench, bevel=0.007)
    floor = material("Studio floor", (0.13, 0.15, 0.17), roughness=0.7)
    box("Studio floor", (200, 200, 0.01), (0, 0, -0.02), floor)
    dark = material("Bench frame", (0.045, 0.055, 0.063), metal=0.65, roughness=0.35)
    for x in (-0.35, 1.05):
        box("Bench pedestal", (0.025, 0.45, surface_z / 2 - 0.02), (x, 0, surface_z / 2 - 0.02), dark, bevel=0.005)

    target = (0.43, 0.0, surface_z + 0.22)
    add_light("Large softbox", (0.25, -0.7, 1.7), target, 50, 1.0, (1.0, 0.90, 0.78))
    add_light("Cool fill", (1.1, 0.45, 1.2), target, 30, 0.8, (0.76, 0.86, 1.0))
    add_light("Edge strip", (-0.6, 0.35, 1.1), target, 35, 0.65, (1.0, 0.98, 0.94))
    camera_data = bpy.data.cameras.new("Pickup camera")
    camera = bpy.data.objects.new("Pickup camera", camera_data)
    bpy.context.collection.objects.link(camera)
    if args.camera in ("closeup", "cinematic"):
        camera.location = (1.08, -0.65, 0.66)
        look_at(camera, (0.55, 0.0, surface_z + 0.14))
        camera_data.lens = 52
    else:
        camera.location = (1.65, -1.5, 1.2)
        look_at(camera, (0.35, 0.0, surface_z + 0.38))
        camera_data.lens = 40
    camera_data.clip_start = 0.01
    camera_data.clip_end = 500
    scene.camera = camera
    return bodies, cable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=("closeup", "overview", "cinematic"), default="cinematic")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--frames", type=int, nargs="+", help="Recorded source frame indices for proof stills.")
    parser.add_argument(
        "--start-frame", type=int, default=0, help="Resume a matching full render at this output frame."
    )
    parser.add_argument("--save-blend", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.recording, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    bodies, cable = build_scene(data, args)
    scene = bpy.context.scene
    source_fps = float(data["fps"])
    if not 0 < args.fps <= source_fps:
        raise ValueError("Output FPS must be positive and no greater than the recorded simulation FPS.")
    # Initialization is sample zero. Interpolate between recorded states when
    # the output cadence does not divide the simulation cadence (e.g. 24/60).
    # Omitting the endpoint gives exactly 288 frames for 12 seconds at 24 Hz.
    duration = (len(data["body_q"]) - 1) / source_fps
    output_frames = max(1, round(duration * args.fps))
    source_frames = args.frames or (np.arange(output_frames) * (source_fps / args.fps)).tolist()
    receipt = {
        "recording": str(args.recording.resolve()),
        "recording_sha256": hashlib.sha256(args.recording.read_bytes()).hexdigest(),
        "source_frames": source_frames,
        "source_fps": source_fps,
        "output_fps": args.fps,
        "rendered_frame_count": len(source_frames),
        "video_duration_seconds": len(source_frames) / args.fps if args.frames is None else None,
        "camera": args.camera,
        "resolution": [args.width, args.height],
        "engine": "Cycles OptiX",
        "denoiser": "OpenImageDenoise GPU (high quality)",
        "samples": args.samples,
        "cable_material": {
            "youngs_modulus_pa": float(data["cable_youngs_modulus"]),
            "poisson_ratio": float(data["cable_poisson_ratio"]),
            "density_kg_m3": float(data["cable_density"]),
        },
        "geometry": "Recorded visible rigid shapes and tetrahedral surface; cosmetic ports and bevels.",
        "motion": (
            "Recorded Newton states at their original time. Between samples, positions and particles use "
            "linear interpolation and body rotations use quaternion SLERP. No pose correction."
        ),
    }
    receipt_path = args.output / "render.json"
    if not 0 <= args.start_frame < len(source_frames):
        raise ValueError("start-frame must be within the output timeline.")
    if args.start_frame:
        if args.frames is not None:
            raise ValueError("start-frame resumes full videos only; it cannot be combined with frames.")
        if not receipt_path.exists() or json.loads(receipt_path.read_text()) != receipt:
            raise ValueError("Resuming requires the same recording, frame timeline and rendering settings.")
        missing = [index for index in range(args.start_frame) if not (args.output / f"{index:05d}.png").is_file()]
        if missing:
            raise ValueError(f"Cannot resume: previous rendered frames are missing: {missing}")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    for output_frame, source_frame in enumerate(source_frames):
        if output_frame < args.start_frame:
            continue
        left = int(source_frame)
        right = min(left + 1, len(data["body_q"]) - 1)
        weight = source_frame - left
        for body_index, obj in enumerate(bodies):
            pose = data["body_q"][left, body_index]
            set_pose(obj, pose)
            if weight:
                following = data["body_q"][right, body_index]
                obj.location = pose[:3] * (1.0 - weight) + following[:3] * weight
                rotation = Quaternion((following[6], following[3], following[4], following[5]))
                obj.rotation_quaternion = obj.rotation_quaternion.slerp(rotation, weight)
        particles = data["particle_q"][left]
        if weight:
            particles = particles * (1.0 - weight) + data["particle_q"][right] * weight
        cable.data.vertices.foreach_set("co", particles.reshape(-1))
        cable.data.update()
        if args.camera == "cinematic":
            u = min(source_frame / source_fps / 3.0, 1.0)
            blend = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
            surface_z = float(data["surface_z"])
            scene.camera.location = Vector((1.65, -1.5, 1.2)).lerp(Vector((1.08, -0.65, 0.66)), blend)
            target = Vector((0.35, 0.0, surface_z + 0.38)).lerp(Vector((0.55, 0.0, surface_z + 0.14)), blend)
            look_at(scene.camera, target)
            scene.camera.data.lens = 40.0 + 12.0 * blend
        scene.frame_set(output_frame + 1)
        scene.render.filepath = str(args.output / f"{output_frame:05d}.png")
        if args.save_blend and output_frame == 0:
            bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "scene.blend"))
        bpy.ops.render.render(write_still=True)
        print(f"Completed recorded frame {source_frame} ({output_frame + 1}/{len(source_frames)})", flush=True)
    if args.frames is None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(args.output / "%05d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output / "usb_hub_pickup.mp4"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
