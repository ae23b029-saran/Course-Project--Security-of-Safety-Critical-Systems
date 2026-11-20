#Step 2 of the universal patch:trains one patch on the training tanks and tests it on the holdout, valid and test
#tanks against grey and noise patches.Used for transfer test too

import os, json, math, csv, shutil, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


ROOT = os.environ.get("AE5510_ROOT", r"D:\Security of Safety Critical Systems")
WEIGHTS = os.path.join(ROOT, "runs", "baseline_A20", "weights", "best.pt")
TANKS_JSON = os.path.join(ROOT, "universal", "tanks.json")
OUT_DIR = os.path.join(ROOT, "universal", "transfer_A20")
LOAD_PATCH = os.path.join(ROOT, "universal", "run_8.6pct", "patch.pt")

#Settings for each run in the report(give OUT_DIR a new name each time):
#main run:WEIGHTS baseline_B20, PATCH_AREA_FRAC = 0.086, BALANCE_SIZES = False, LOAD_PATCH = None.
#patch size runs: PATCH_AREA_FRAC = 0.04 or 0.12. Balanced run: BALANCE_SIZES = True.
#transfer test(as set now):WEIGHTS baseline_A20, LOAD_PATCH = the main run's patch.pt.

NAMES = ["Aircraft", "Building", "Truck", "Tank"]
TANK = 3

PATCH_AREA_FRAC = 0.086
PATCH_RES = 64
MIN_SIDE = 6

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

BALANCE_SIZES = False
SIZE_BINS = [(0, 15), (15, 25), (25, 40), (40, 10000)]

MIN_TANK_PX = 32
EVAL_SETS = ["valid", "test", "holdout"]
CONF = 0.25
IOU_MATCH = 0.5
N_EXAMPLES = 8
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

train = usable_only(data["train"])

def patch_side(t):
    x1, y1, x2, y2 = t["box"]
    want = math.sqrt(PATCH_AREA_FRAC * (x2 - x1) * (y2 - y1))
    return max(MIN_SIDE, min(want, t["max_side"]))

all_tanks = [t for it in train for t in it["tanks"]]
sides = np.array([patch_side(t) for t in all_tanks])
capped = sum(patch_side(t) < math.sqrt(PATCH_AREA_FRAC * (t["box"][2] - t["box"][0])
                                        * (t["box"][3] - t["box"][1])) - 1e-6
             for t in all_tanks)
print(f"device {DEVICE}")
print(f"train : {len(train)} images, {len(all_tanks)} tanks")
print(f"patch : {100*PATCH_AREA_FRAC:.1f}% of box area   on screen "
      f"{sides.min():.0f}-{sides.max():.0f} px (median {np.median(sides):.0f})   "
      f"{capped} tanks too small -> patch shrunk to fit\n")

def size_bin(t):
    side = patch_side(t)
    for b, (lo, hi) in enumerate(SIZE_BINS):
        if lo <= side < hi:
            return b
    return len(SIZE_BINS) - 1

counts = np.bincount([size_bin(t) for t in all_tanks], minlength=len(SIZE_BINS))
filled = counts > 0
bin_weight = np.ones(len(SIZE_BINS))
if BALANCE_SIZES:
    bin_weight[filled] = counts[filled].mean() / counts[filled]
print("size groups in training:")
for (lo, hi), c, w in zip(SIZE_BINS, counts, bin_weight):
    label = f"{lo}-{hi} px" if hi < 10000 else f">= {lo} px"
    print(f"  {label:>10s}  {c:5d} tanks   weight {w:.2f}")
print()

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
            w = float(bin_weight[size_bin(t)])
            per_tank.append(w * s.topk(min(TOPK, s.numel())).values.mean())
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

variants = [("clean", None),
            ("adversarial", patch),
            ("grey", torch.full_like(patch, 0.5)),
            ("noise", torch.rand(patch.shape, generator=torch.Generator().manual_seed(SEED)).to(DEVICE))]
save_patch(variants[3][1], "noise_control")

rows, tank_rows = [], []
for s_idx, split in enumerate(EVAL_SETS):
    items = usable_only(data[split])
    by_path = {it["image"]: it for it in items}

    if split == "holdout":
        all_imgs = [it["image"] for it in data["holdout"]]
    else:
        img_dir = os.path.join(ROOT, "uas", split, "images")
        all_imgs = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]

    ev = np.random.default_rng(SEED + 999 + s_idx)
    views = {it["image"]: [random_view(ev) for _ in it["tanks"]] for it in items}
    n_tanks = sum(len(it["tanks"]) for it in items)
    if n_tanks == 0:
        print(f"\n=== {split}: no usable tanks, skipped ===")
        continue
    print(f"\n=== {split}: {len(all_imgs)} images, {len(items)} with usable tanks, "
          f"{n_tanks} usable tanks ===")

    clean_found = None
    found_by = {}
    tank_info = []
    for name, p in variants:
        torch.manual_seed(SEED + s_idx)

        copy = os.path.join(OUT_DIR, "tmp_dataset")
        shutil.rmtree(copy, ignore_errors=True)
        os.makedirs(os.path.join(copy, "images"))
        os.makedirs(os.path.join(copy, "labels"))

        found_tank, found_any, n_ex = [], [], 0
        for src in all_imgs:
            stem = os.path.splitext(os.path.basename(src))[0]
            x = to_input(load(src))
            it = by_path.get(src)
            if it is not None and p is not None:
                with torch.no_grad():
                    x = paste(x, p, it["tanks"], views[src])
            bgr = as_bgr(x)
            cv2.imwrite(os.path.join(copy, "images", stem + ".png"), bgr)
            lab = os.path.splitext(src.replace(os.sep + "images" + os.sep,
                                   os.sep + "labels" + os.sep))[0] + ".txt"
            if os.path.exists(lab):
                shutil.copy(lab, os.path.join(copy, "labels", stem + ".txt"))

            if it is None:
                continue
            r = wrapper.predict(bgr, conf=CONF, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for t in it["tanks"]:
                if name == "clean":
                    x1, y1, x2, y2 = t["box"]
                    tank_info.append(dict(image=os.path.basename(src),
                                          tank_short_px=round(min(x2 - x1, y2 - y1), 1),
                                          patch_side_px=round(patch_side(t), 1)))
                ious = [iou(t["box"], b) for b in boxes]
                found_tank.append(any(o >= IOU_MATCH and c == TANK
                                      for o, c in zip(ious, cls)))
                found_any.append(any(o >= IOU_MATCH for o in ious))
            if n_ex < N_EXAMPLES:
                cv2.imwrite(os.path.join(OUT_DIR, "examples",
                                         f"{split}_{n_ex}_{name}.png"), r.plot())
                n_ex += 1

        found_tank = np.array(found_tank)
        found_any = np.array(found_any)
        if name == "clean":
            clean_found = found_tank
        found_by[name] = found_tank
        base = max(1, clean_found.sum())
        hidden = 100.0 * (clean_found & ~found_tank).sum() / base

        m = YOLO(WEIGHTS).val(data=write_yaml(copy), split="val", imgsz=IMGSZ,
                              batch=8, workers=0, plots=False, verbose=False,
                              project=os.path.join(OUT_DIR, "val_runs"),
                              name=f"{split}_{name}", exist_ok=True)
        idx = list(m.box.ap_class_index)
        ap50 = float(m.box.ap50[idx.index(TANK)]) if TANK in idx else 0.0
        ap = float(m.box.ap[idx.index(TANK)]) if TANK in idx else 0.0

        print(f"  {name:12s} tanks found {found_tank.sum():4d}/{len(found_tank)}"
              f"  (as any class {found_any.sum():4d})   "
              f"hidden {hidden:5.1f}%   Tank AP50 {ap50:.3f}  AP50-95 {ap:.3f}"
              f"   mAP50 {m.box.map50:.3f}")
        rows.append(dict(split=split, variant=name, tanks=len(found_tank),
                         found_as_tank=int(found_tank.sum()),
                         found_any_class=int(found_any.sum()),
                         hidden_pct=f"{hidden:.1f}",
                         tank_ap50=f"{ap50:.4f}", tank_ap50_95=f"{ap:.4f}",
                         map50=f"{m.box.map50:.4f}", map50_95=f"{m.box.map:.4f}",
                         area_frac=PATCH_AREA_FRAC, weights=WEIGHTS))

    print(f"\n  by patch size on screen ({split}):")
    print(f"  {'patch side':>12s} {'tanks':>6s} {'found clean':>12s}"
          f" {'hidden: adv':>12s} {'grey':>6s} {'noise':>6s}")
    sides_px = np.array([d["patch_side_px"] for d in tank_info])
    for lo, hi in SIZE_BINS:
        sel = (sides_px >= lo) & (sides_px < hi)
        base_sel = sel & clean_found
        if sel.sum() == 0:
            continue
        pct = lambda v: (100.0 * (base_sel & ~found_by[v]).sum() / base_sel.sum()
                         if base_sel.sum() else float("nan"))
        label = f"{lo}-{hi} px" if hi < 10000 else f">= {lo} px"
        print(f"  {label:>12s} {sel.sum():6d} {base_sel.sum():12d}"
              f" {pct('adversarial'):11.1f}% {pct('grey'):5.1f}% {pct('noise'):5.1f}%")

    for i, d in enumerate(tank_info):
        tank_rows.append(dict(split=split, **d,
                              **{f"found_{v}": int(found_by[v][i]) for v in found_by}))

shutil.rmtree(os.path.join(OUT_DIR, "tmp_dataset"), ignore_errors=True)
with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

with open(os.path.join(OUT_DIR, "per_tank.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tank_rows[0].keys()))
    w.writeheader()
    w.writerows(tank_rows)

print(f"\nwritten to {OUT_DIR}")
print("  patch.png / patch_print.png   the patch")
print("  results.csv                   the numbers above")
print("  per_tank.csv                  every tank: size, found or not, per variant")
print("  examples\\                     detections, clean vs patched")
