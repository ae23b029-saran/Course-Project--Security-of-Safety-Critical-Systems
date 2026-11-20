# Patch size sweep on one tank image: trains an adversarial patch at sizes from 1.0 m to 3.0 m,
# with and without EOT, and compares each one with grey and noise patches of the same size.

import os, glob, math, csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT = r"D:\Security of Safety Critical Systems"
WEIGHTS = os.path.join(ROOT, "runs", "baseline_B20", "weights", "best.pt")
OUT_DIR = os.path.join(ROOT, "patch_sweep_eot")
IMG_DIR = os.path.join(ROOT, "uas", "valid", "images")

NAMES = ["Aircraft", "Building", "Truck", "Tank"]
PREFER = "tank-29"

OBJECT_LENGTH_M = 7.0

RUNS = [
    ("1.0m_noEOT", 1.0, False),
    ("1.0m_EOT", 1.0, True),
    ("1.5m_EOT", 1.5, True),
    ("2.0m_EOT", 2.0, True),
    ("2.5m_EOT", 2.5, True),
    ("3.0m_EOT", 3.0, True),
]

ROT_DEG = 45.0
SCALE_JITTER = 0.25
SQUASH_JITTER = 0.15
SHIFT_FRAC = 0.06
BRIGHT_JIT = 0.20
CONTRAST_JIT = 0.20
NOISE_STD = 0.02

IMGSZ = 640
STEPS = 400
EOT_BATCH = 4
LR = 0.03
TOPK = 20
TV_WEIGHT = 0.0
SEED = 0

EVAL_TRIALS = 50
CONF = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

img_path, label_line = None, None
for p in sorted(glob.glob(os.path.join(IMG_DIR, "*"))):
    lab = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    lab = os.path.splitext(lab)[0] + ".txt"
    lines = [l for l in open(lab) if l.strip()]
    if len(lines) == 1 and PREFER in os.path.basename(p).lower():
        img_path, label_line = p, lines[0]
        break

if img_path is None:
    raise SystemExit(f"No single-object image matching '{PREFER}' in {IMG_DIR}")

cls_id, cx, cy, bw, bh = label_line.split()
cls_id = int(cls_id)
cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)

box_w, box_h = bw * IMGSZ, bh * IMGSZ
m_per_px = OBJECT_LENGTH_M / max(box_w, box_h)

px_target = 2.0 * cx - 1.0
py_target = 2.0 * cy - 1.0

print("image :", os.path.basename(img_path))
print(f"target: {NAMES[cls_id]}   box {box_w:.0f} x {box_h:.0f} px")
print(f"scale : {m_per_px*100:.2f} cm per pixel "
      f"(assuming a {OBJECT_LENGTH_M} m object)\n")

wrapper = YOLO(WEIGHTS)
net = wrapper.model.to(DEVICE).eval()
for prm in net.parameters():
    prm.requires_grad_(False)

bgr = cv2.resize(cv2.imread(img_path), (IMGSZ, IMGSZ))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
base = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1).unsqueeze(0)
base = base.to(DEVICE) / 255.0

side = 0
c0 = 0
mask_template = None

def place(patch, params):
    canvas = torch.zeros(1, 3, IMGSZ, IMGSZ, device=DEVICE)
    canvas[0, :, c0:c0 + side, c0:c0 + side] = patch
    mask = mask_template

    if params is None:
        theta = torch.tensor([[[1.0, 0.0, -px_target],
                               [0.0, 1.0, -py_target]]],
                             dtype=torch.float32, device=DEVICE)
        grid = F.affine_grid(theta, canvas.shape, align_corners=False)
        canvas = F.grid_sample(canvas, grid, align_corners=False)
        mask = F.grid_sample(mask, grid, align_corners=False)
    else:
        rot, sx, sy, dx, dy, bright, contrast = params
        a = math.radians(rot)
        cos_a, sin_a = math.cos(a), math.sin(a)

        A = [[ cos_a / sx, sin_a / sx],
             [-sin_a / sy, cos_a / sy]]

        gx, gy = px_target + dx, py_target + dy
        tx = -(A[0][0] * gx + A[0][1] * gy)
        ty = -(A[1][0] * gx + A[1][1] * gy)

        theta = torch.tensor([[[A[0][0], A[0][1], tx],
                               [A[1][0], A[1][1], ty]]],
                             dtype=torch.float32, device=DEVICE)
        grid = F.affine_grid(theta, canvas.shape, align_corners=False)
        canvas = F.grid_sample(canvas, grid, align_corners=False)
        mask = F.grid_sample(mask, grid, align_corners=False)

        canvas = canvas * contrast + bright
        canvas = canvas + torch.randn_like(canvas) * NOISE_STD
        canvas = canvas.clamp(0.0, 1.0)

    mask = mask.clamp(0.0, 1.0)
    return base * (1.0 - mask) + canvas * mask

def random_view(r):
    return (r.uniform(-ROT_DEG, ROT_DEG),
            1.0 + r.uniform(-SCALE_JITTER, SCALE_JITTER),
            1.0 + r.uniform(-SQUASH_JITTER, SQUASH_JITTER),
            r.uniform(-SHIFT_FRAC, SHIFT_FRAC),
            r.uniform(-SHIFT_FRAC, SHIFT_FRAC),
            r.uniform(-BRIGHT_JIT, BRIGHT_JIT),
            1.0 + r.uniform(-CONTRAST_JIT, CONTRAST_JIT))

def as_image(t):
    a = t[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR)

def raw_conf(x):
    with torch.no_grad():
        return net(x)[0][0, 4:, :].max().item()

clean_conf = raw_conf(base)
clean_res = wrapper.predict(as_image(base), verbose=False, conf=CONF)[0]
cv2.imwrite(os.path.join(OUT_DIR, "00_clean.png"), clean_res.plot())
print(f"CLEAN IMAGE: conf {clean_conf:.4f}, {len(clean_res.boxes)} detection(s)\n")
print("=" * 78)

all_rows = []

for label, size_m, use_eot in RUNS:
    side = max(12, int(round(size_m / m_per_px)))
    if side > IMGSZ:
        print(f"\n{label}: patch larger than the image, skipped")
        continue
    c0 = IMGSZ // 2 - side // 2
    mask_template = torch.zeros(1, 1, IMGSZ, IMGSZ, device=DEVICE)
    mask_template[0, :, c0:c0 + side, c0:c0 + side] = 1.0

    area_pct = 100.0 * side * side / (box_w * box_h)
    n_views = EOT_BATCH if use_eot else 1

    print(f"\n{label}   {size_m} m -> {side} px   "
          f"{area_pct:.2f}% of box   EOT {'on' if use_eot else 'OFF'}")

    torch.manual_seed(SEED)
    train_rng = np.random.default_rng(SEED)

    patch = torch.rand(3, side, side, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=LR)

    for step in range(STEPS):
        opt.zero_grad()
        running = 0.0

        for _ in range(n_views):
            params = random_view(train_rng) if use_eot else None
            x = place(patch, params)
            scores = net(x)[0][0, 4:, :]
            loss_i = scores.max(dim=0).values.topk(TOPK).values.mean() / n_views
            loss_i.backward()
            running += loss_i.item()

        if TV_WEIGHT > 0:
            dh = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
            dw = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
            (TV_WEIGHT * (dh + dw)).backward()

        opt.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

        if step % 100 == 0 or step == STEPS - 1:
            print(f"    step {step:4d}  loss {running:.4f}")

    grey = torch.full((3, side, side), 0.5, device=DEVICE)
    noise = torch.rand(3, side, side, device=DEVICE)
    variants = [("adversarial", patch.detach()), ("grey", grey), ("noise", noise)]

    eval_rng = np.random.default_rng(SEED + 999)
    trials = [random_view(eval_rng) for _ in range(EVAL_TRIALS)]

    for name, p in variants:
        xn = place(p, None)
        nom_conf = raw_conf(xn)
        rn = wrapper.predict(as_image(xn), verbose=False, conf=CONF)[0]
        nom_det = len(rn.boxes)
        if name == "adversarial":
            cv2.imwrite(os.path.join(OUT_DIR, f"{label}_nominal.png"), rn.plot())

        confs, zero = [], 0
        for prm in trials:
            xr = place(p, prm)
            confs.append(raw_conf(xr))
            if len(wrapper.predict(as_image(xr), verbose=False,
                                   conf=CONF)[0].boxes) == 0:
                zero += 1
        confs = np.array(confs)
        rate = 100.0 * zero / EVAL_TRIALS

        print(f"    {name:12s} nominal {nom_conf:.4f} ({nom_det} det)   "
              f"views mean {confs.mean():.4f}  max {confs.max():.4f}   "
              f"suppressed {zero}/{EVAL_TRIALS} ({rate:.0f}%)")

        all_rows.append(dict(
            run=label, patch_m=size_m, side_px=side,
            area_pct=f"{area_pct:.2f}", eot="yes" if use_eot else "no",
            variant=name,
            clean_conf=f"{clean_conf:.4f}",
            nominal_conf=f"{nom_conf:.4f}", nominal_det=nom_det,
            views_mean=f"{confs.mean():.4f}",
            views_median=f"{np.median(confs):.4f}",
            views_max=f"{confs.max():.4f}",
            suppress_pct=f"{rate:.1f}"))

    arr = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT_DIR, f"{label}_patch.png"), arr)
    cv2.imwrite(os.path.join(OUT_DIR, f"{label}_patch_print.png"),
                cv2.resize(arr, (2048, 2048), interpolation=cv2.INTER_NEAREST))

csv_path = os.path.join(OUT_DIR, "sweep_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)

print("\n" + "=" * 78)
print("SUMMARY  (adversarial patch only)\n")
print(f"{'run':14s} {'area%':>7s} {'EOT':>4s} {'nominal':>9s} "
      f"{'views mean':>11s} {'suppressed':>11s}")
for r in all_rows:
    if r["variant"] != "adversarial":
        continue
    print(f"{r['run']:14s} {r['area_pct']:>7s} {r['eot']:>4s} "
          f"{r['nominal_conf']:>9s} {r['views_mean']:>11s} "
          f"{r['suppress_pct']:>10s}%")

print(f"\nclean image confidence: {clean_conf:.4f}")
print(f"\nfull table -> {csv_path}")
print(f"images     -> {OUT_DIR}")
