def parse_scene(scene_str: str):
    """Parses a scene string to its different components."""
    scene_objects = scene_str.split("***")
    shapes = [obj.split()[0] for obj in scene_objects]
    materials = [obj.split()[1] for obj in scene_objects]
    sizes = [obj.split()[2] for obj in scene_objects]
    colors = [obj.split()[3] for obj in scene_objects]
    lefts = [obj.split()[4] for obj in scene_objects]
    fronts = [obj.split()[5] for obj in scene_objects]

    return shapes, materials, sizes, colors, lefts, fronts