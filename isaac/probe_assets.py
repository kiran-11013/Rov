"""Stage 3b prep: find realistic 3D assets for our object classes, and check the ML environment.

Real detectors cannot recognise the plain box / cylinder proxies, so the scene needs real models (mugs, chairs,
laptops...). This lists what the Isaac Sim asset library offers for each class and checks torch / CUDA / model
downloads. Run with Isaac Sim's Python (needs internet for the asset library):

    ~/isaacsim/python.sh isaac/probe_assets.py

Writes isaac/out/asset_catalog.json and prints the best matches per class.
"""
import json
import os
import sys
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.client  # noqa: E402

KEYWORDS = {
    "chair": ["chair"], "stool": ["stool"], "cart": ["cart", "trolley"], "bag": ["bag", "backpack"],
    "laptop": ["laptop", "notebook"], "monitor": ["monitor", "screen", "display"],
    "box": ["cardbox", "cardboard", "box"], "bin": ["bin", "trash", "basket"], "plant": ["plant", "pot"],
    "toolbox": ["toolbox", "tool_box", "tool"], "mug": ["mug", "cup"],
}
SEARCH = ["Isaac/Props", "Isaac/Environments/Office", "Isaac/Environments/Simple_Warehouse/Props",
          "Isaac/Environments/Hospital", "NVIDIA/Assets/ArchVis", "NVIDIA/Assets/DigitalTwin",
          "NVIDIA/Assets/simready_content", "NVIDIA/Assets/Isaac"]
MAX_DEPTH, MAX_FILES, TIME_LIMIT_S = 6, 20000, 600


def log(msg):
    print(f"[probe] {msg}", flush=True)


def assets_root():
    for mod, fn in (("isaacsim.storage.native", "get_assets_root_path"),
                    ("isaacsim.core.utils.nucleus", "get_assets_root_path")):
        try:
            return getattr(__import__(mod, fromlist=[fn]), fn)()
        except Exception as e:  # noqa: BLE001
            log(f"{mod}.{fn} unavailable: {e}")
    return None


def listdir(url):
    res, entries = omni.client.list(url)
    if res != omni.client.Result.OK:
        return None
    return entries


def walk(root, rel, found, t0, depth=0):
    if depth > MAX_DEPTH or len(found) >= MAX_FILES or time.time() - t0 > TIME_LIMIT_S:
        return
    entries = listdir(f"{root}/{rel}")
    if entries is None:
        return
    for e in entries:
        name = e.relative_path.rstrip("/")
        path = f"{rel}/{name}"
        if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            if name.lower() not in (".thumbs", "materials", "textures", "texture", "mdl"):
                walk(root, path, found, t0, depth + 1)
        elif name.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
            found.append(path)


root = assets_root()
log(f"assets root: {root}")
catalog = {"root": root, "top_level": {}, "files": [], "matches": {}}
if root:
    for top in ("Isaac", "NVIDIA", "NVIDIA/Assets", "Isaac/Props", "Isaac/Environments"):
        entries = listdir(f"{root}/{top}")
        catalog["top_level"][top] = [e.relative_path for e in entries] if entries else None
        log(f"{top}: {catalog['top_level'][top]}")
    t0 = time.time()
    for rel in SEARCH:
        n0 = len(catalog["files"])
        walk(root, rel, catalog["files"], t0)
        log(f"searched {rel}: {len(catalog['files']) - n0} USD files ({time.time() - t0:.0f} s so far)")
    for cls, kws in KEYWORDS.items():
        hits = [f for f in catalog["files"] if any(k in os.path.basename(f).lower() for k in kws)]
        hits.sort(key=lambda f: (len(f), f))
        catalog["matches"][cls] = hits

# ----------------------------------------------------------------------------- ML environment
env = {"python": sys.version.split()[0]}
try:
    import torch
    env.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception as e:  # noqa: BLE001
    env["torch"] = f"missing ({e})"
for pkg in ("torchvision", "transformers", "ultralytics", "open_clip", "PIL", "cv2"):
    try:
        m = __import__(pkg)
        env[pkg] = getattr(m, "__version__", "present")
    except Exception:  # noqa: BLE001
        env[pkg] = "missing"
for name, url in (("huggingface", "https://huggingface.co"), ("pypi", "https://pypi.org/simple/ultralytics/"),
                  ("github", "https://github.com")):
    try:
        import urllib.request
        urllib.request.urlopen(url, timeout=8)
        env[f"net_{name}"] = "ok"
    except Exception as e:  # noqa: BLE001
        env[f"net_{name}"] = f"fail ({type(e).__name__})"
catalog["env"] = env

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "asset_catalog.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    json.dump(catalog, fh, indent=1)

print("\n==== ASSET MATCHES (shortest paths first, up to 8 per class) ====")
for cls, hits in catalog["matches"].items():
    print(f"{cls:8s} {len(hits):4d} | " + (" ; ".join(hits[:8]) if hits else "-"))
print("\n==== ML ENVIRONMENT ====")
for k, v in env.items():
    print(f"{k:16s} {v}")
print(f"\nwrote {out} ({len(catalog['files'])} USD files listed)")
app.close()
