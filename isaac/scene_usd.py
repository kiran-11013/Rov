"""Build the exported scene in the open USD stage (shared by capture.py and fly_episode.py).

Import only after SimulationApp has started (it needs pxr). Objects are either plain proxies (box / cylinder
coloured by their attribute, the Stage 3a default) or realistic models from the Isaac Sim asset library
(assets.json), scaled uniformly to the class height and stood on the same spot. Every object lives under
/World/Objects/obj_<oid>, so instance segmentation maps back to object ids either way.
"""
import json
import math
import os

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
COLOURS = {"red": (0.75, 0.12, 0.10), "blue": (0.12, 0.30, 0.75), "green": (0.12, 0.55, 0.20),
           "black": (0.05, 0.05, 0.05), "white": (0.92, 0.92, 0.90), "grey": (0.50, 0.50, 0.50),
           "yellow": (0.90, 0.75, 0.10)}
FURNITURE = {"desk": (0.62, 0.50, 0.35), "table": (0.62, 0.50, 0.35), "part": (0.55, 0.62, 0.72),
             "cub": (0.55, 0.62, 0.72), "shelf": (0.35, 0.35, 0.38), "clutter": (0.30, 0.25, 0.22)}


def log(msg):
    print(f"[scene] {msg}", flush=True)


def cube(stage, path, cx, cy, cz, sx, sy, sz, colour, rot_z=0.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.GetSizeAttr().Set(1.0)
    xf = UsdGeom.XformCommonAPI(c)
    xf.SetTranslate(Gf.Vec3d(cx, cy, cz))
    if rot_z:
        xf.SetRotate(Gf.Vec3f(0.0, 0.0, rot_z), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xf.SetScale(Gf.Vec3f(sx, sy, sz))
    c.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return c


def cylinder(stage, path, cx, cy, z_base, radius, height, colour):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.GetRadiusAttr().Set(radius)
    c.GetHeightAttr().Set(height)
    c.GetAxisAttr().Set("Z")
    UsdGeom.XformCommonAPI(c).SetTranslate(Gf.Vec3d(cx, cy, z_base + height / 2))
    c.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return c


def add_label(prim, label):
    """Semantic class label; the API name changed across Isaac Sim versions, so try each."""
    for attempt in ("add_labels", "add_update_semantics"):
        try:
            mod = __import__("isaacsim.core.utils.semantics", fromlist=[attempt])
            fn = getattr(mod, attempt)
            if attempt == "add_labels":
                fn(prim, labels=[label], instance_name="class")
            else:
                fn(prim, label)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def assets_root():
    for mod, fn in (("isaacsim.storage.native", "get_assets_root_path"),
                    ("isaacsim.core.utils.nucleus", "get_assets_root_path")):
        try:
            return getattr(__import__(mod, fromlist=[fn]), fn)()
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("cannot find the Isaac Sim assets root (needs internet or a local asset pack)")


def asset_for(table, ob):
    """(model path, rx override or None) for an object: open-laptop model if the spec's laptop is tall; mugs vary
    with colour. An entry is a path, a list of paths, or {"path": ..., "rx": degrees} to force the up-rotation."""
    cls = ob["cls"]
    a = table["laptop_open"] if (cls == "laptop" and ob["dims"][-1] > 0.1 and "laptop_open" in table) else table[cls]
    if isinstance(a, list):
        a = a[sorted(COLOURS).index(ob["colour"]) % len(a) if ob["colour"] in COLOURS else 0]
    if isinstance(a, dict):
        return a["path"], a.get("rx")
    return a, None


_UP = {}


def _asset_up_axis(url):
    if url not in _UP:
        try:
            _UP[url] = UsdGeom.GetStageUpAxis(Usd.Stage.Open(url, Usd.Stage.LoadNone))
        except Exception:  # noqa: BLE001
            _UP[url] = UsdGeom.Tokens.z
    return _UP[url]


def tint(stage, prim_path, colour):
    """Override the model's materials with a flat colour (keeps the colour attribute meaningful for re-ID)."""
    mpath = f"{prim_path}/TintMaterial"
    mat = UsdShade.Material.Define(stage, mpath)
    sh = UsdShade.Shader.Define(stage, f"{mpath}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", "color3f").Set(Gf.Vec3f(*colour))
    sh.CreateInput("roughness", "float").Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


_REPORTED = set()


def place_model(stage, root, ob, rel_path, colour=None, rx_override=None):
    """Reference a model under ob['prim'], turn it Z-up, scale it uniformly to fit inside the proxy's size
    (height and both footprint sides), give it a per-object yaw, and stand it on (x, y, z_base).
    Returns the scale used."""
    url = f"{root}/{rel_path}"
    wrapper = UsdGeom.Xform.Define(stage, ob["prim"])
    inner = UsdGeom.Xform.Define(stage, f"{ob['prim']}/model")
    inner.GetPrim().GetReferences().AddReference(url)
    for _ in range(6):  # instanced meshes would segment under shared prototype paths, not under obj_<oid>
        inst = [p for p in Usd.PrimRange(inner.GetPrim()) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
    up = _asset_up_axis(url)
    rx = rx_override if rx_override is not None else (90.0 if up == UsdGeom.Tokens.y else 0.0)
    xf = UsdGeom.XformCommonAPI(wrapper)
    xf.SetRotate(Gf.Vec3f(rx, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)  # measure without yaw
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(wrapper.GetPrim()).ComputeAlignedRange()
    if box.IsEmpty():
        raise RuntimeError(f"empty bounds for {url}")
    lo, hi = box.GetMin(), box.GetMax()
    ex, ey, ez = (max(hi[i] - lo[i], 1e-6) for i in range(3))
    if ob["shape"] == "box":
        tx, ty, tz = ob["dims"]
    else:
        tx = ty = 2 * ob["dims"][0]
        tz = ob["dims"][1]
    (ma, mb), (ta, tb) = sorted((ex, ey), reverse=True), sorted((tx, ty), reverse=True)
    s = min(tz / ez, ta / ma, tb / mb)
    yaw = float((ob["oid"] * 47) % 360)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    c, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    xf.SetRotate(Gf.Vec3f(rx, 0.0, yaw), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xf.SetScale(Gf.Vec3f(s, s, s))
    xf.SetTranslate(Gf.Vec3d(ob["x"] - s * (c * cx - sn * cy), ob["y"] - s * (sn * cx + c * cy),
                             ob["z_base"] - s * lo[2]))
    if rel_path not in _REPORTED:
        _REPORTED.add(rel_path)
        limit = min(("height", tz / ez), ("footprint", ta / ma), ("footprint", tb / mb), key=lambda t: t[1])[0]
        log(f"model {ob['cls']:8s} up={up} rx={rx:g} raw size {ex:.3g} x {ey:.3g} x {ez:.3g} -> scale {s:.4g} "
            f"(limited by {limit}), final {s * ex:.2f} x {s * ey:.2f} x {s * ez:.2f} m "
            f"(proxy {tx:.2f} x {ty:.2f} x {tz:.2f}) [{rel_path.split('/')[-1]}]")
    if colour is not None:
        tint(stage, ob["prim"], colour)
    return s


def build_scene(stage, spec, assets="proxy", tint_models=False, colliders=False):
    """Room, furniture, objects and lights. assets: 'proxy' or 'real'. colliders: give the floor, walls and
    furniture static collision (needed when a physically simulated drone flies in the scene)."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    room = spec["room"]
    cube(stage, "/World/Floor", room["width"] / 2, room["height"] / 2, -0.01, room["width"], room["height"], 0.02,
         (0.80, 0.80, 0.78))
    for i, w in enumerate(spec["walls"]):
        (ax, ay), (bx, by) = w["a"], w["b"]
        length = math.hypot(bx - ax, by - ay)
        horiz = abs(by - ay) < 1e-9
        cube(stage, f"/World/Walls/wall_{i}", (ax + bx) / 2, (ay + by) / 2, room["wall_height"] / 2,
             length if horiz else 0.1, 0.1 if horiz else length, room["wall_height"], (0.88, 0.87, 0.84))
    for b in spec["boxes"]:
        colour = next((c for k, c in FURNITURE.items() if b["name"].startswith(k)), (0.6, 0.6, 0.6))
        cube(stage, f"/World/Furniture/{b['name']}", (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, b["h"] / 2,
             b["x1"] - b["x0"], b["y1"] - b["y0"], b["h"], colour)

    table, root = None, None
    if assets == "real":
        with open(os.path.join(HERE, "assets.json")) as fh:
            table = json.load(fh)
        root = assets_root()
        log(f"real models from {root}")
    n_labelled, n_real, failed = 0, 0, {}
    for ob in spec["objects"]:
        colour = COLOURS.get(ob["colour"], (0.5, 0.5, 0.5))
        prim = None
        if table is not None:
            rel, rx = asset_for(table, ob)
            try:
                place_model(stage, root, ob, rel, colour if tint_models else None, rx_override=rx)
                prim = stage.GetPrimAtPath(ob["prim"])
                n_real += 1
            except Exception as e:  # noqa: BLE001
                failed[rel] = str(e)
                stage.RemovePrim(ob["prim"])
        if prim is None:
            if ob["shape"] == "box":
                sx, sy, sz = ob["dims"]
                prim = cube(stage, ob["prim"], ob["x"], ob["y"], ob["z_base"] + sz / 2, sx, sy, sz, colour).GetPrim()
            else:
                r, hgt = ob["dims"]
                prim = cylinder(stage, ob["prim"], ob["x"], ob["y"], ob["z_base"], r, hgt, colour).GetPrim()
        n_labelled += add_label(prim, ob["cls"])
    for rel, err in failed.items():
        log(f"WARNING: model failed, proxy used instead: {rel}: {err}")
    log(f"built {len(spec['objects'])} objects ({n_real} real models, {n_labelled} with semantic labels), "
        f"{len(spec['boxes'])} furniture boxes, {len(spec['walls'])} walls")

    if colliders:
        from pxr import UsdPhysics
        for root_path in ("/World/Floor", "/World/Walls", "/World/Furniture"):
            root_prim = stage.GetPrimAtPath(root_path)
            if root_prim:
                for p in Usd.PrimRange(root_prim):
                    if p.IsA(UsdGeom.Gprim):
                        UsdPhysics.CollisionAPI.Apply(p)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(800.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-50.0, 20.0, 0.0))
    return n_real
