#Single-image patch attack: trains an adversarial patch on one tank image, compares it with grey and noise patches of the same size.

import os, glob, math, csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

#Change these two lines for each patch size: 0.086 with None (about 2 m, box centre),
#0.0484 with None (about 1.5 m, box centre)).
PATCH_AREA_FRAC = 0.086
PLACE_XY = None

ROOT = r"D:\Security of Safety Critical Systems"
WEIGHTS = os.path.join(ROOT, "runs", "baseline_B20", "weights", "best.pt")
OUT_DIR = os.path.join(ROOT, f"patch_core_{PATCH_AREA_FRAC}")
IMG_DIR = os.path.join(ROOT, "uas", "valid", "images")

NAMES = ["Aircraft", "Building", "Truck", "Tank"]
PREFER = "tank-29"

OBJECT_LENGTH_M = 7.0

USE_EOT = True
EOT_BATCH = 8
ROT_DEG, SCALE_JIT, SQUASH_JIT = 45.0, 0.25, 0.15
SHIFT, BRIGHT_JIT, CONTRAST_JIT, NOISE_STD = 0.06, 0.20, 0.20, 0.02

IMGSZ = 640
STEPS = 600
LR = 0.03
LR_MIN = 0.003
TOPK = 20

EVAL_TRIALS = 50
CONF = 0.25
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

img_path = label_line = None
for p in sorted(glob.glob(os.path.join(IMG_DIR, "*"))):
    lab = os.path.splitext(p.replace(os.sep + "images" + os.sep,
                                     os.sep + "labels" + os.sep))[0] + ".txt"
    lines = [l for l in open(lab) if l.strip()]
    if len(lines) == 1 and PREFER in os.path.basename(p).lower():
        img_path, label_line = p, lines[0]
        break
if img_path is None:
    raise SystemExit(f"no single-object image matching '{PREFER}'")

cls_id, cx, cy, bw, bh = label_line.split()
cls_id = int(cls_id)
cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)

box_w, box_h = bw * IMGSZ, bh * IMGSZ
side = max(12, int(round(math.sqrt(PATCH_AREA_FRAC * box_w * box_h))))
area_pct = 100.0 * side * side / (box_w * box_h)
m_per_px = OBJECT_LENGTH_M / max(box_w, box_h)
equiv_m = side * m_per_px

print(f"image : {os.path.basename(img_path)}")
print(f"target: {NAMES[cls_id]}   box {box_w:.0f} x {box_h:.0f} px")
print(f"patch : {side} px = {area_pct:.2f}% of box area "
      f"(~{equiv_m:.2f} m if the vehicle is {OBJECT_LENGTH_M} m)")
print(f"EOT   : {'on, ' + str(EOT_BATCH) + ' views per update' if USE_EOT else 'OFF'}\n")

wrapper = YOLO(WEIGHTS)
net = wrapper.model.to(DEVICE).eval()
for prm in net.parameters():
    prm.requires_grad_(False)

rgb = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (IMGSZ, IMGSZ)),
                   cv2.COLOR_BGR2RGB)
base = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1).unsqueeze(0)
base = base.to(DEVICE) / 255.0

c0 = IMGSZ // 2 - side // 2
mask_tmpl = torch.zeros(1, 1, IMGSZ, IMGSZ, device=DEVICE)
mask_tmpl[0, :, c0:c0 + side, c0:c0 + side] = 1.0

if PLACE_XY is None:
    px_t, py_t = 2.0 * cx - 1.0, 2.0 * cy - 1.0
else:
    px_t = 2.0 * PLACE_XY[0] / IMGSZ - 1.0
    py_t = 2.0 * PLACE_XY[1] / IMGSZ - 1.0

def place(patch, view):
    canvas = torch.zeros(1, 3, IMGSZ, IMGSZ, device=DEVICE)
    canvas[0, :, c0:c0 + side, c0:c0 + side] = patch
    mask = mask_tmpl

    if view is None:
        rot, sx, sy, dx, dy, bright, contrast = 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0
    else:
        rot, sx, sy, dx, dy, bright, contrast = view

    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)

    A = [[ca / sx, sa / sx],
         [-sa / sy, ca / sy]]

    gx, gy = px_t + dx, py_t + dy
    tx = -(A[0][0] * gx + A[0][1] * gy)
    ty = -(A[1][0] * gx + A[1][1] * gy)

    theta = torch.tensor([[[A[0][0], A[0][1], tx],
                           [A[1][0], A[1][1], ty]]],
                         dtype=torch.float32, device=DEVICE)

    grid = F.affine_grid(theta, canvas.shape, align_corners=False)
    canvas = F.grid_sample(canvas, grid, align_corners=False)
    mask = F.grid_sample(mask, grid, align_corners=False)

    if view is not None:
        canvas = canvas * contrast + bright
        canvas = canvas + torch.randn_like(canvas) * NOISE_STD
    canvas = canvas.clamp(0.0, 1.0)
    mask = mask.clamp(0.0, 1.0)

    return base * (1.0 - mask) + canvas * mask

def random_view():
    return (rng.uniform(-ROT_DEG, ROT_DEG),
            1.0 + rng.uniform(-SCALE_JIT, SCALE_JIT),
            1.0 + rng.uniform(-SQUASH_JIT, SQUASH_JIT),
            rng.uniform(-SHIFT, SHIFT), rng.uniform(-SHIFT, SHIFT),
            rng.uniform(-BRIGHT_JIT, BRIGHT_JIT),
            1.0 + rng.uniform(-CONTRAST_JIT, CONTRAST_JIT))

def as_image(t):
    a = t[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR)

def confidence(x):
    with torch.no_grad():
        return net(x)[0][0, 4:, :].max().item()

def detections(x):
    return wrapper.predict(as_image(x), verbose=False, conf=CONF)[0]

clean_c = confidence(base)
clean_r = detections(base)
cv2.imwrite(os.path.join(OUT_DIR, "0_clean.png"), clean_r.plot())
print(f"CLEAN: conf {clean_c:.4f}, {len(clean_r.boxes)} detection(s)\n")

patch = torch.rand(3, side, side, device=DEVICE, requires_grad=True)
opt = torch.optim.Adam([patch], lr=LR)

n_views = EOT_BATCH if USE_EOT else 1

print("optimising...")
for step in range(STEPS):
    t = step / max(1, STEPS - 1)
    for g in opt.param_groups:
        g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * t))

    opt.zero_grad()
    running = 0.0

    for _ in range(n_views):
        x = place(patch, random_view() if USE_EOT else None)
        scores = net(x)[0][0, 4:, :]
        loss = scores.max(dim=0).values.topk(TOPK).values.mean() / n_views
        loss.backward()
        running += loss.item()

    opt.step()
    with torch.no_grad():
        patch.clamp_(0.0, 1.0)

    if step % 50 == 0 or step == STEPS - 1:
        print(f"  step {step:4d}  loss {running:.4f}")

grey = torch.full((3, side, side), 0.5, device=DEVICE)
noise = torch.rand(3, side, side, device=DEVICE)
variants = [("adversarial", patch.detach()), ("grey", grey), ("noise", noise)]

print("\nnominal view (no transform):")
nominal = {}
for name, p in variants:
    x = place(p, None)
    r = detections(x)
    c = confidence(x)
    nominal[name] = (c, len(r.boxes))
    cv2.imwrite(os.path.join(OUT_DIR, f"1_nominal_{name}.png"), r.plot())
    print(f"  {name:12s} conf {c:.4f}   {len(r.boxes)} detection(s)")

ev = np.random.default_rng(SEED + 999)
saved = rng.bit_generator.state
rng.bit_generator.state = ev.bit_generator.state
trials = [random_view() for _ in range(EVAL_TRIALS)]
rng.bit_generator.state = saved

print(f"\n{EVAL_TRIALS} random views:")
rows = []
for name, p in variants:
    confs, zero = [], 0
    for v in trials:
        x = place(p, v)
        confs.append(confidence(x))
        if len(detections(x).boxes) == 0:
            zero += 1
    confs = np.array(confs)
    rate = 100.0 * zero / EVAL_TRIALS
    print(f"  {name:12s} mean {confs.mean():.4f}  max {confs.max():.4f}   "
          f"suppressed {zero}/{EVAL_TRIALS} ({rate:.0f}%)")
    rows.append(dict(variant=name, side_px=side, area_pct=f"{area_pct:.2f}",
                     equiv_m=f"{equiv_m:.2f}", eot_batch=n_views, steps=STEPS,
                     clean_conf=f"{clean_c:.4f}",
                     nominal_conf=f"{nominal[name][0]:.4f}",
                     nominal_det=nominal[name][1],
                     views_mean=f"{confs.mean():.4f}",
                     views_max=f"{confs.max():.4f}",
                     suppress_pct=f"{rate:.1f}"))

with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

arr = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(OUT_DIR, "2_patch.png"), arr)
cv2.imwrite(os.path.join(OUT_DIR, "2_patch_print.png"),
            cv2.resize(arr, (2048, 2048), interpolation=cv2.INTER_NEAREST))

print(f"\nwritten to {OUT_DIR}")
