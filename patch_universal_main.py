#Trains the final universal patch on all training images and tests it on the unseen valid and test tanks,
#each in 20 camera views, against grey and noise patches of the same size.

import os, json, math, csv, shutil, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT = os.environ.get("AE5510_ROOT", r"D:\Security of Safety Critical Systems")
WEIGHTS = os.path.join(ROOT, "runs", "baseline_B20", "weights", "best.pt")
TANKS_JSON = os.path.join(ROOT, "universal_all", "tanks.json")
OUT_DIR = os.path.join(ROOT, "universal_all", "run_8.6pct")
LOAD_PATCH = None

NAMES = ["Aircraft", "Building", "Truck", "Tank"]
TANK = 3

PATCH_AREA_FRAC = 0.086
PATCH_RES = 64
MIN_SIDE = 6
MIN_TANK_PX = 32

ROT_DEG = 180.0
SIZE_JIT = 0.10
SHIFT_FRAC = 0.10
BRIGHT_JIT = 0.20
CONTRAST_JIT = 0.20
NOISE_STD = 0.02

IMGSZ = 640
STEPS = 1000
IMG_BATCH = 8
LR = 0.03
LR_MIN = 0.003
TOPK = 10

N_VIEWS = 20
CAM_ZOOM = (0.7, 1.0)
CAM_BRIGHT = 0.15
CAM_CONTRAST = 0.20
CAM_BLUR = (0.5, 1.5)
CAM_HAZE = (0.1, 0.3)
CAM_NOISE = 0.01
N_BOOT = 2000

CONF = 0.25
IOU_MATCH = 0.5
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.join(OUT_DIR, "examples"), exist_ok=True)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

wrapper = YOLO(WEIGHTS)
net = wrapper.model.to(DEVICE).eval()
for prm in net.parameters():
    prm.requires_grad_(False)

data = json.load(open(TANKS_JSON))

def usable_only(items):
    out = []
    for it in items:
        big = [t for t in it["tanks"]
               if min(t["box"][2] - t["box"][0], t["box"][3] - t["box"][1]) >= MIN_TANK_PX]
        if big:
            out.append(dict(image=it["image"], tanks=big))
    return out

def patch_side(t):
    x1, y1, x2, y2 = t["box"]
    want = math.sqrt(PATCH_AREA_FRAC * (x2 - x1) * (y2 - y1))
    return max(MIN_SIDE, min(want, t["max_side"]))

train = usable_only(data["train"])
all_tanks = [t for it in train for t in it["tanks"]]
sides = np.array([patch_side(t) for t in all_tanks])
print(f"device {DEVICE}")
print(f"train : {len(train)} images, {len(all_tanks)} tanks")
print(f"patch : {100*PATCH_AREA_FRAC:.1f}% of box area   on screen "
      f"{sides.min():.0f}-{sides.max():.0f} px (median {np.median(sides):.0f})\n")

def load(path):
    bgr = cv2.resize(cv2.imread(path), (IMGSZ, IMGSZ))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

def to_input(u8):
    return u8.to(DEVICE).float().unsqueeze(0) / 255.0

def random_view(r):
    return (r.uniform(-ROT_DEG, ROT_DEG),
            1.0 + r.uniform(-SIZE_JIT, SIZE_JIT),
            r.uniform(-SHIFT_FRAC, SHIFT_FRAC),
            r.uniform(-SHIFT_FRAC, SHIFT_FRAC),
            r.uniform(-BRIGHT_JIT, BRIGHT_JIT),
            1.0 + r.uniform(-CONTRAST_JIT, CONTRAST_JIT))

def paste(img, patch, tanks, views):
    out = img
    for t, (rot, size, dx, dy, bright, contrast) in zip(tanks, views):
        side = patch_side(t) * size
        n = max(2, int(round(side)))
        small = F.interpolate(patch.unsqueeze(0), size=(n, n), mode="bilinear",
                              align_corners=False, antialias=True)
        a = math.radians(rot)
        ca, sa = math.cos(a), math.sin(a)
        k = IMGSZ / side
        A = [[k * ca, k * sa],
             [-k * sa, k * ca]]
        gx = 2.0 * (t["x"] + dx * side) / IMGSZ - 1.0
        gy = 2.0 * (t["y"] + dy * side) / IMGSZ - 1.0
        theta = torch.tensor([[[A[0][0], A[0][1], -(A[0][0] * gx + A[0][1] * gy)],
                               [A[1][0], A[1][1], -(A[1][0] * gx + A[1][1] * gy)]]],
                             dtype=torch.float32, device=DEVICE)
        grid = F.affine_grid(theta, (1, 3, IMGSZ, IMGSZ), align_corners=False)
        canvas = F.grid_sample(small, grid, align_corners=False)
        mask = F.grid_sample(torch.ones_like(small[:, :1]), grid,
                               align_corners=False)
        canvas = canvas * contrast + bright
        canvas = canvas + torch.randn_like(canvas) * NOISE_STD
        canvas = canvas.clamp(0.0, 1.0)
        out = out * (1.0 - mask) + canvas * mask
    return out

def tank_loss(x, tanks):
    out = net(x)[0][0]
    cx, cy = out[0].detach(), out[1].detach()
    score = out[4:].max(dim=0).values
    per_tank = []
    for t in tanks:
        x1, y1, x2, y2 = t["box"]
        inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        s = score[inside]
        if s.numel() > 0:
            per_tank.append(s.topk(min(TOPK, s.numel())).values.mean())
    return torch.stack(per_tank).mean() if per_tank else None

def as_bgr(x):
    a = x[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR)

def save_patch(p, name):
    arr = (p.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT_DIR, name + ".png"), arr)
    cv2.imwrite(os.path.join(OUT_DIR, name + "_print.png"),
                cv2.resize(arr, (2048, 2048), interpolation=cv2.INTER_NEAREST))

if LOAD_PATCH:
    patch = torch.load(LOAD_PATCH, map_location=DEVICE)
    print(f"loaded {LOAD_PATCH} - skipping training\n")
else:
    print("loading training images...")
    train_imgs = [load(it["image"]) for it in train]

    patch = torch.rand(3, PATCH_RES, PATCH_RES, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=LR)
    order, pos = rng.permutation(len(train)), 0
    log, t0 = [], time.time()

    print("optimising...")
    for step in range(STEPS):
        t = step / max(1, STEPS - 1)
        for g in opt.param_groups:
            g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * t))

        opt.zero_grad()
        running = 0.0
        for _ in range(IMG_BATCH):
            if pos == len(order):
                order, pos = rng.permutation(len(train)), 0
            i = order[pos]
            pos += 1
            tanks = train[i]["tanks"]
            views = [random_view(rng) for _ in tanks]
            x = paste(to_input(train_imgs[i]), patch, tanks, views)
            loss = tank_loss(x, tanks)
            if loss is None:
                continue
            (loss / IMG_BATCH).backward()
            running += loss.item() / IMG_BATCH

        opt.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)
        log.append(running)

        if step % 50 == 0 or step == STEPS - 1:
            left = (time.time() - t0) / (step + 1) * (STEPS - step - 1) / 60
            print(f"  step {step:4d}  loss {running:.4f}   (~{left:.0f} min left)")
        if step % 100 == 0:
            torch.save(patch.detach().cpu(), os.path.join(OUT_DIR, "patch.pt"))

    patch = patch.detach()
    torch.save(patch.cpu(), os.path.join(OUT_DIR, "patch.pt"))
    with open(os.path.join(OUT_DIR, "train_loss.csv"), "w") as f:
        f.write("step,loss\n" + "".join(f"{s},{l:.5f}\n" for s, l in enumerate(log)))
    del train_imgs

save_patch(patch, "patch")

def random_camera(r, view_no):
    if view_no == 0:
        return dict(k=0, flip=False, zoom=1.0, ox=0, oy=0, bright=0.0,
                    contrast=1.0, blur=0.0, haze=0.0, name="original")
    zoom = r.uniform(*CAM_ZOOM) if r.random() < 0.5 else 1.0
    n = int(round(IMGSZ * zoom))
    v = dict(k=int(r.integers(0, 4)),
             flip=bool(r.random() < 0.5),
             zoom=n / IMGSZ,
             ox=int(r.integers(0, IMGSZ - n + 1)),
             oy=int(r.integers(0, IMGSZ - n + 1)),
             bright=r.uniform(-CAM_BRIGHT, CAM_BRIGHT),
             contrast=1.0 + r.uniform(-CAM_CONTRAST, CAM_CONTRAST),
             blur=r.uniform(*CAM_BLUR) if r.random() < 0.3 else 0.0,
             haze=r.uniform(*CAM_HAZE) if r.random() < 0.3 else 0.0)
    v["name"] = (f"rot{90*v['k']}" + ("_flip" if v["flip"] else "")
                 + (f"_zoom{v['zoom']:.2f}" if v["zoom"] < 1 else "")
                 + (f"_blur{v['blur']:.1f}" if v["blur"] else "")
                 + (f"_haze{v['haze']:.2f}" if v["haze"] else ""))
    return v

def apply_camera(bgr, v, noise_seed):
    img = bgr.astype(np.float32) / 255.0

    if v["zoom"] < 1.0:
        n = int(round(IMGSZ * v["zoom"]))
        small = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
        canvas = np.empty_like(img)
        canvas[:] = img.mean(axis=(0, 1))
        canvas[v["oy"]:v["oy"] + n, v["ox"]:v["ox"] + n] = small
        img = canvas
    if v["flip"]:
        img = img[:, ::-1]
    if v["k"]:
        img = np.rot90(img, v["k"])
    img = np.ascontiguousarray(img)
    img = (img - 0.5) * v["contrast"] + 0.5 + v["bright"]
    if v["haze"]:
        img = img * (1 - v["haze"]) + 0.85 * v["haze"]
    if v["blur"]:
        img = cv2.GaussianBlur(img, (0, 0), v["blur"])
    if v["name"] != "original":
        img = img + np.random.default_rng(noise_seed).normal(0, CAM_NOISE, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def camera_box(b, v):
    pts = []
    for x, y in [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])]:
        if v["zoom"] < 1.0:
            x, y = v["ox"] + v["zoom"] * x, v["oy"] + v["zoom"] * y
        if v["flip"]:
            x = IMGSZ - x
        for _ in range(v["k"]):
            x, y = y, IMGSZ - x
        pts.append((x, y))
    xs, ys = zip(*pts)
    return [min(xs), min(ys), max(xs), max(ys)]

def camera_labels(label_file, v):
    lines = []
    if not os.path.exists(label_file):
        return lines
    for line in open(label_file):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(q) * IMGSZ for q in p[1:5])
        x1, y1, x2, y2 = camera_box([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], v)
        lines.append(f"{p[0]} {(x1+x2)/2/IMGSZ:.6f} {(y1+y2)/2/IMGSZ:.6f} "
                     f"{(x2-x1)/IMGSZ:.6f} {(y2-y1)/IMGSZ:.6f}")
    return lines

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)

def write_yaml(folder):
    path = os.path.join(folder, "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {folder.replace(os.sep, '/')}\n"
                "train: images\nval: images\nnames:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")
    return path

def label_of(img_path):
    return os.path.splitext(img_path.replace(os.sep + "images" + os.sep,
                                             os.sep + "labels" + os.sep))[0] + ".txt"

items = usable_only(data["valid"]) + usable_only(data["test"])
by_path = {it["image"]: it for it in items}
all_imgs = []
for split in ["valid", "test"]:
    d = os.path.join(ROOT, "uas", split, "images")
    all_imgs += [os.path.join(d, f) for f in sorted(os.listdir(d))]

ev = np.random.default_rng(SEED + 999)
place = {it["image"]: [random_view(ev) for _ in it["tanks"]] for it in items}
cams = {(src, vn): random_camera(ev, vn) for src in all_imgs for vn in range(N_VIEWS)}
n_tanks = sum(len(it["tanks"]) for it in items)
print(f"=== unseen tanks (valid + test): {len(all_imgs)} images, {len(items)} with "
      f"usable tanks, {n_tanks} tanks, {N_VIEWS} views each ===")

variants = [("clean", None),
            ("adversarial", patch),
            ("grey", torch.full_like(patch, 0.5)),
            ("noise", torch.rand(patch.shape, generator=torch.Generator().manual_seed(SEED)).to(DEVICE))]
save_patch(variants[3][1], "noise_control")

found = {}
ap_rows = {}
for name, p in variants:
    torch.manual_seed(SEED)
    copy = os.path.join(OUT_DIR, "tmp_dataset")
    shutil.rmtree(copy, ignore_errors=True)
    os.makedirs(os.path.join(copy, "images"))
    os.makedirs(os.path.join(copy, "labels"))
    found[name] = {}
    n_ex = 0

    for i_img, src in enumerate(all_imgs):
        stem = os.path.splitext(os.path.basename(src))[0]
        x = to_input(load(src))
        it = by_path.get(src)
        if it is not None and p is not None:
            with torch.no_grad():
                x = paste(x, p, it["tanks"], place[src])
        patched = as_bgr(x)

        for vn in range(N_VIEWS):
            v = cams[(src, vn)]
            img = apply_camera(patched, v, noise_seed=SEED + 100000 * i_img + vn)
            cv2.imwrite(os.path.join(copy, "images", f"{stem}_v{vn:02d}.png"), img)
            with open(os.path.join(copy, "labels", f"{stem}_v{vn:02d}.txt"), "w") as f:
                f.write("\n".join(camera_labels(label_of(src), v)))

            if it is None:
                continue
            r = wrapper.predict(img, conf=CONF, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for tn, t in enumerate(it["tanks"]):
                gt = camera_box(t["box"], v)
                found[name][(src, tn, vn)] = any(
                    iou(gt, b) >= IOU_MATCH and c == TANK for b, c in zip(boxes, cls))
            if n_ex < 6 and vn in (0, 1, 2):
                cv2.imwrite(os.path.join(OUT_DIR, "examples",
                                         f"{stem[:20]}_v{vn:02d}_{name}.png"), r.plot())
                n_ex += 1

    m = YOLO(WEIGHTS).val(data=write_yaml(copy), split="val", imgsz=IMGSZ,
                          batch=8, workers=0, plots=False, verbose=False,
                          project=os.path.join(OUT_DIR, "val_runs"),
                          name=name, exist_ok=True)
    idx = list(m.box.ap_class_index)
    ap_rows[name] = dict(tank_ap50=float(m.box.ap50[idx.index(TANK)]) if TANK in idx else 0.0,
                         tank_ap50_95=float(m.box.ap[idx.index(TANK)]) if TANK in idx else 0.0,
                         map50=float(m.box.map50), map50_95=float(m.box.map))
    print(f"  {name:12s} done   Tank AP50 {ap_rows[name]['tank_ap50']:.3f}")

shutil.rmtree(os.path.join(OUT_DIR, "tmp_dataset"), ignore_errors=True)

keys = list(found["clean"].keys())
tanks_ids = sorted(set((k[0], k[1]) for k in keys))
clean = found["clean"]

def hidden_frac(variant, tank, views=range(N_VIEWS)):
    ok = [vn for vn in views if clean[(tank[0], tank[1], vn)]]
    if not ok:
        return None
    return np.mean([not found[variant][(tank[0], tank[1], vn)] for vn in ok])

boot = np.random.default_rng(SEED + 7)

def mean_and_range(values):
    a = np.array(values, dtype=float)
    idx = boot.integers(0, len(a), (N_BOOT, len(a)))
    lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
    return 100 * a.mean(), 100 * lo, 100 * hi

counted = [tk for tk in tanks_ids if hidden_frac("adversarial", tk) is not None]
counted0 = [tk for tk in tanks_ids if clean[(tk[0], tk[1], 0)]]
n_view_ok = sum(clean[k] for k in keys)

print(f"\n  tanks found clean in at least one view: {len(counted)} of {len(tanks_ids)}"
      f"   (tank-views found clean: {n_view_ok} of {len(keys)})")
print(f"\n  {'variant':12s} {'all views: hidden':>18s}   {'per tank: average [95% range]':>30s}"
      f"   {'original view only':>18s}   {'Tank AP50':>9s}")
rows = []
per_tank = {}
for name, _ in variants:
    tv = [k for k in keys if clean[k]]
    view_hidden = 100 * np.mean([not found[name][k] for k in tv]) if name != "clean" else 0.0
    per_tank[name] = [hidden_frac(name, tk) for tk in counted]
    avg, lo, hi = mean_and_range(per_tank[name])
    orig_hidden = sum(not found[name][(tk[0], tk[1], 0)] for tk in counted0)
    print(f"  {name:12s} {view_hidden:17.1f}%   {avg:12.1f}% [{lo:4.1f} - {hi:4.1f}]"
          f"   {orig_hidden:8d} / {len(counted0):<8d}   {ap_rows[name]['tank_ap50']:9.3f}")
    rows.append(dict(variant=name, views_hidden_pct=f"{view_hidden:.1f}",
                     per_tank_hidden_pct=f"{avg:.1f}", range_lo=f"{lo:.1f}", range_hi=f"{hi:.1f}",
                     original_view_hidden=f"{orig_hidden}/{len(counted0)}",
                     tanks_counted=len(counted), n_views=N_VIEWS,
                     **{k: f"{v:.4f}" for k, v in ap_rows[name].items()},
                     area_frac=PATCH_AREA_FRAC, weights=WEIGHTS))

print("\n  the pattern's own effect, per tank (paired: same tank, same views):")
for ctrl in ["grey", "noise"]:
    diff = np.array(per_tank["adversarial"]) - np.array(per_tank[ctrl])
    avg, lo, hi = mean_and_range(diff)
    print(f"    adversarial - {ctrl:5s}: {avg:+5.1f} points  [95% range {lo:+.1f} to {hi:+.1f}]")
    rows.append(dict(variant=f"gap_vs_{ctrl}", per_tank_hidden_pct=f"{avg:.1f}",
                     range_lo=f"{lo:.1f}", range_hi=f"{hi:.1f}"))

print(f"\n  every tank (hidden in % of the views where it is found clean):")
print(f"  {'image':40s} {'tank':>4s} {'size px':>7s} {'views':>5s}"
      f" {'adv':>6s} {'grey':>6s} {'noise':>6s}")
tank_rows = []
for tk in counted:
    b = by_path[tk[0]]["tanks"][tk[1]]["box"]
    size = min(b[2] - b[0], b[3] - b[1])
    nv = sum(clean[(tk[0], tk[1], vn)] for vn in range(N_VIEWS))
    h = {v: 100 * hidden_frac(v, tk) for v in ["adversarial", "grey", "noise"]}
    print(f"  {os.path.basename(tk[0])[:40]:40s} {tk[1]:4d} {size:7.0f} {nv:5d}"
          f" {h['adversarial']:5.0f}% {h['grey']:5.0f}% {h['noise']:5.0f}%")
    tank_rows.append(dict(image=os.path.basename(tk[0]), tank=tk[1], tank_short_px=round(size, 1),
                          views_found_clean=nv,
                          **{f"hidden_pct_{v}": round(h[v], 1) for v in h}))

with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
    fields = sorted(set(k for r in rows for k in r), key=lambda k: list(rows[0]).index(k)
                    if k in rows[0] else 99)
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(OUT_DIR, "per_tank.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tank_rows[0].keys()))
    w.writeheader()
    w.writerows(tank_rows)
with open(os.path.join(OUT_DIR, "per_tank_view.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "tank", "view", "camera"] + [f"found_{v}" for v, _ in variants])
    for k in keys:
        w.writerow([os.path.basename(k[0]), k[1], k[2], cams[(k[0], k[2])]["name"]]
                   + [int(found[v][k]) for v, _ in variants])

print(f"\nwritten to {OUT_DIR}")
print("  patch.png / patch_print.png   the patch")
print("  results.csv                   the summary above")
print("  per_tank.csv                  every tank: % of views hidden, per variant")
print("  per_tank_view.csv             every tank in every view: found or not")
print("  examples\\                     detections in a few views")
