#Step 1 of the universal patch:finds every tank, outlines it with SAM and works out where a patch can sit on it.
#Uses 400 training images for training, 200 other training images as a holdout set, and all valid and test images.

import os, glob, json, math, random
import cv2
import numpy as np
from ultralytics import SAM

ROOT = os.environ.get("AE5510_ROOT", r"D:\Security of Safety Critical Systems")
UAS = os.path.join(ROOT, "uas")
OUT = os.path.join(ROOT, "universal")
SAM_WEIGHTS = "mobile_sam.pt"

TANK = 3
IMGSZ = 640

N_TRAIN_IMAGES = 400
N_HOLDOUT_IMAGES = 200
MIN_TANK_PX = 32
MIN_BOX_PX = 24
MIN_MASK_FRAC = 0.15
CHECK_PER_SET = 4
SEED = 0

check_dir = os.path.join(OUT, "placement_check")
os.makedirs(check_dir, exist_ok=True)
for old in glob.glob(os.path.join(check_dir, "*.png")):
    os.remove(old)
random.seed(SEED)

def label_path(img_path):
    p = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return os.path.splitext(p)[0] + ".txt"

def tank_boxes(img_path):
    boxes = []
    lab = label_path(img_path)
    if not os.path.exists(lab):
        return boxes
    for line in open(lab):
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) != TANK:
            continue
        cx, cy, w, h = (float(v) * IMGSZ for v in parts[1:5])
        boxes.append([max(0.0, cx - w / 2), max(0.0, cy - h / 2),
                      min(IMGSZ, cx + w / 2), min(IMGSZ, cy + h / 2)])
    return boxes

def short_side(b):
    return min(b[2] - b[0], b[3] - b[1])

def box_fallback_mask(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    m = np.zeros((IMGSZ, IMGSZ), np.uint8)
    m[int(y1 + 0.2 * h):int(y2 - 0.2 * h) + 1,
      int(x1 + 0.2 * w):int(x2 - 0.2 * w) + 1] = 1
    return m

def best_spot(mask):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    r = float(dist.max())
    return float(x), float(y), r, r * math.sqrt(2.0)

def images_with_tanks(split):
    files = sorted(glob.glob(os.path.join(UAS, split, "images", "*")))
    return [p for p in files if tank_boxes(p)]

usable = [p for p in images_with_tanks("train")
          if any(short_side(b) >= MIN_TANK_PX for b in tank_boxes(p))]
random.shuffle(usable)
sets = {
    "valid": images_with_tanks("valid"),
    "test": images_with_tanks("test"),
    "train": sorted(usable[:N_TRAIN_IMAGES]),
    "holdout": sorted(usable[N_TRAIN_IMAGES:N_TRAIN_IMAGES + N_HOLDOUT_IMAGES]),
}
print(f"training images with a usable tank: {len(usable)}  "
      f"-> {len(sets['train'])} train, {len(sets['holdout'])} holdout\n")

sam = SAM(SAM_WEIGHTS)
data = {"meta": dict(imgsz=IMGSZ, tank_class=TANK, seed=SEED,
                     min_tank_px=MIN_TANK_PX)}

for name, paths in sets.items():
    items, n_tanks, n_usable, n_sam, n_box, n_check = [], 0, 0, 0, 0, 0
    fits_frac = []

    for k, p in enumerate(paths):
        img = cv2.resize(cv2.imread(p), (IMGSZ, IMGSZ))
        boxes = tank_boxes(p)

        big = [b for b in boxes if short_side(b) >= MIN_BOX_PX]
        sam_masks = {}
        if big:
            res = sam(img, bboxes=big, verbose=False)[0]
            if res.masks is not None:
                md = res.masks.data.cpu().numpy()
                for b, m in zip(big, md):
                    if m.shape != (IMGSZ, IMGSZ):
                        m = cv2.resize(m.astype(np.uint8), (IMGSZ, IMGSZ),
                                       interpolation=cv2.INTER_NEAREST)
                    sam_masks[tuple(b)] = (m > 0.5).astype(np.uint8)

        tanks = []
        checked_this_image = False
        for b in boxes:
            x1, y1, x2, y2 = b
            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            m = sam_masks.get(tuple(b))
            method = "sam"
            if m is not None:
                clip = np.zeros_like(m)
                clip[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = 1
                m = m * clip
                if m.sum() < MIN_MASK_FRAC * box_area:
                    m = None
            if m is None:
                m = box_fallback_mask(b)
                method = "box"

            px, py, r, max_side = best_spot(m)
            tanks.append(dict(box=[round(v, 1) for v in b],
                              x=round(px, 1), y=round(py, 1),
                              max_side=round(max_side, 1), method=method))

            n_tanks += 1
            if short_side(b) < MIN_TANK_PX:
                continue
            n_usable += 1
            n_sam += method == "sam"
            n_box += method == "box"
            fits_frac.append(max_side ** 2 / box_area)

            if n_check < CHECK_PER_SET and not checked_this_image:
                vis = img.copy()
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 128, 0), 1)
                h = max_side / 2
                cv2.rectangle(vis, (int(px - h), int(py - h)),
                              (int(px + h), int(py + h)), (0, 255, 255), 2)
                cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)
                cv2.putText(vis, f"{name} {method}", (5, 20), 0, 0.6,
                            (255, 255, 255), 2)
                stem = os.path.splitext(os.path.basename(p))[0]
                cv2.imwrite(os.path.join(check_dir, f"{name}_{n_check}_{stem}.png"), vis)
                n_check += 1
                checked_this_image = True

        items.append(dict(image=p, tanks=tanks))
        if (k + 1) % 50 == 0:
            print(f"  {name}: {k + 1}/{len(paths)} images")

    data[name] = items
    ff = np.array(fits_frac) if fits_frac else np.array([0.0])
    print(f"{name:7s}: {len(items):4d} images, {n_tanks:5d} tanks, "
          f"{n_usable:5d} usable (>= {MIN_TANK_PX} px)   "
          f"outline from SAM {n_sam}, from box {n_box}   "
          f"largest patch that fits: median {100*np.median(ff):.1f}% of box area")

with open(os.path.join(OUT, "tanks.json"), "w") as f:
    json.dump(data, f)

print(f"\nsaved   {os.path.join(OUT, 'tanks.json')}")
print(f"check   {check_dir}")
print("        green = vehicle outline   yellow = largest patch that fits"
      "   red = patch centre")
