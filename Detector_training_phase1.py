# Phase I pipeline: prepares and checks the aerial image dataset, then trains and
# evaluates the YOLO11s baseline detectors (model A and model B).

import os
import re
import glob
import shutil
from pathlib import Path
from collections import Counter


SRC = r"D:\Security of Safety Critical Systems\Aerial Image Dataset"
DST = r"D:\Security of Safety Critical Systems\uas"

NAMES = ["Aircraft", "Building", "Truck", "Tank"]

MODEL_WEIGHTS = r"D:\Security of Safety Critical Systems\yolo11s.pt"
EPOCHS = 20
IMGSZ = 640
BATCH = 8
FREEZE = 10

YAML_PATH = os.path.join(DST, "data.yaml")
RUNS_DIR = os.path.join(os.path.dirname(DST), "runs")
CHECK_DIR = os.path.join(os.path.dirname(DST), "check")

SPLITS = {"train": "Training Data",
          "valid": "Validation Data",
          "test": "Testing Data"}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_STEMS = {"classes", "readme", "data", "notes"}


def images_dir(split):
    return os.path.join(DST, split, "images")


def labels_dir(split):
    return os.path.join(DST, split, "labels")


def list_images(split):
    return sorted(glob.glob(os.path.join(images_dir(split), "*")))


def list_labels(split):
    return sorted(glob.glob(os.path.join(labels_dir(split), "*.txt")))


def label_for(img_path):
    p = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return os.path.splitext(p)[0] + ".txt"


def require(condition, message):
    if not condition:
        raise SystemExit("STOP: " + message)


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def step1_environment():
    banner("STEP 1  environment")
    import torch
    print("torch:", torch.__version__)
    cuda = torch.cuda.is_available()
    print("CUDA available:", cuda)
    if cuda:
        print("GPU:", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem:.1f} GB")
    else:
        print("  -> running on CPU. Training will be slow.")
        print("  -> if the version string ends in +cpu you have the wrong build.")

    print("\nSRC exists:", os.path.isdir(SRC))
    print("DST exists:", os.path.isdir(DST))


def step2_explore():
    banner("STEP 2  raw dataset structure")
    require(os.path.isdir(SRC), f"SRC not found: {SRC}")
    for root, dirs, files in os.walk(SRC):
        depth = root.replace(SRC, "").count(os.sep)
        if depth <= 3:
            print("  " * depth, os.path.basename(root) or SRC, f"({len(files)} files)")


def step3_organise():
    banner("STEP 3  reorganise into DST")
    require(os.path.isdir(SRC), f"SRC not found: {SRC}")

    for split, folder in SPLITS.items():
        src_dir = os.path.join(SRC, folder)
        require(os.path.isdir(src_dir), f"missing source folder: {src_dir}")

        os.makedirs(images_dir(split), exist_ok=True)
        os.makedirs(labels_dir(split), exist_ok=True)

        images, labels = {}, {}
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                stem, ext = Path(f).stem, Path(f).suffix.lower()
                path = os.path.join(root, f)
                if ext in IMG_EXT:
                    images[stem] = path
                elif ext == ".txt" and stem.lower() not in SKIP_STEMS:
                    labels[stem] = path

        matched = 0
        for stem, ipath in images.items():
            if stem in labels:
                shutil.copy2(ipath, os.path.join(images_dir(split), Path(ipath).name))
                shutil.copy2(labels[stem], os.path.join(labels_dir(split), stem + ".txt"))
                matched += 1

        print(f"{split:6s}: {len(images):5d} images, {len(labels):5d} labels, {matched:5d} matched")
        if matched < len(images):
            print("   WARNING: some images had no matching label file")


def step4_inspect_labels():
    banner("STEP 4  label inspection")
    labs = list_labels("train")
    require(labs, f"no label files in {labels_dir('train')} - run step 3 first")

    print("label files:", len(labs))
    print("example file:", os.path.basename(labs[0]))
    print("contents:")
    print(open(labs[0]).read().strip())

    counts, empty, total_lines = Counter(), 0, 0
    for p in labs:
        lines = [l for l in open(p) if l.strip()]
        if not lines:
            empty += 1
        total_lines += len(lines)
        for line in lines:
            counts[int(line.split()[0])] += 1

    print("\nclass id counts:")
    for cid in sorted(counts):
        name = NAMES[cid] if cid < len(NAMES) else "UNKNOWN ID"
        print(f"   {cid}  {name:16s} {counts[cid]:6d}")
    print("empty label files:", empty)
    print(f"objects per image: {total_lines / len(labs):.1f}")

    print("\nsample image filenames:")
    for p in list_images("train")[:8]:
        print("  ", os.path.basename(p))


def step5_diagnose_coords():
    banner("STEP 5  coordinate diagnostic (read only, nothing is modified)")
    worst, over_one, under_zero, checked = [], 0, 0, 0

    for split in SPLITS:
        for p in list_labels(split):
            for line in open(p):
                if not line.strip():
                    continue
                checked += 1
                vals = list(map(float, line.split()[1:5]))
                if any(v < 0 or v > 1 for v in vals):
                    if any(v > 1 for v in vals):
                        over_one += 1
                    if any(v < 0 for v in vals):
                        under_zero += 1
                    worst.append((max(vals), min(vals), split, Path(p).name, line.strip()))

    print(f"lines checked: {checked}")
    print(f"out of range : {len(worst)}   (>1: {over_one}, <0: {under_zero})")

    if not worst:
        print("all coordinates are normalised and in range - nothing to fix")
        return

    worst.sort(key=lambda t: -t[0])
    print("\nmost extreme by max value:")
    for mx, mn, split, name, line in worst[:8]:
        print(f"   max={mx:.4f} min={mn:.4f}  [{split}] {name}")
        print(f"      {line}")

    worst.sort(key=lambda t: t[1])
    print("\nmost extreme by min value:")
    for mx, mn, split, name, line in worst[:8]:
        print(f"   max={mx:.4f} min={mn:.4f}  [{split}] {name}")
        print(f"      {line}")

    biggest = max(t[0] for t in worst)
    print("\nINTERPRETATION")
    if biggest <= 1.05:
        print("  Values only just exceed 1. This is edge spillover from the")
        print("  rotation / zoom augmentation. Safe to clip (step 6).")
    else:
        print("  Values are well above 1. The labels may be in PIXELS, not")
        print("  normalised. DO NOT clip - they need dividing by the image")
        print("  width/height instead. Check before proceeding.")


def step6_fix_coords(min_visible=0.25):
    banner("STEP 6  FIX COORDINATES - this rewrites label files")

    stats = {s: Counter() for s in SPLITS}
    for split in SPLITS:
        for p in list_labels(split):
            out = []
            for line in open(p):
                if not line.strip():
                    continue
                parts = line.split()
                c = parts[0]
                cx, cy, w, h = map(float, parts[1:5])

                x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                orig = max(w * h, 1e-12)

                nx1, ny1 = max(0.0, x1), max(0.0, y1)
                nx2, ny2 = min(1.0, x2), min(1.0, y2)
                nw, nh = nx2 - nx1, ny2 - ny1

                if nw <= 0 or nh <= 0:
                    stats[split]["dropped_outside"] += 1
                    continue
                if (nw * nh) / orig < min_visible:
                    stats[split]["dropped_sliver"] += 1
                    continue

                if (nx1, ny1, nx2, ny2) != (x1, y1, x2, y2):
                    stats[split]["adjusted"] += 1
                else:
                    stats[split]["unchanged"] += 1

                out.append(f"{c} {(nx1+nx2)/2:.6f} {(ny1+ny2)/2:.6f} {nw:.6f} {nh:.6f}")

            open(p, "w").write("\n".join(out) + ("\n" if out else ""))

    for split in SPLITS:
        s = stats[split]
        print(f"{split:6s} unchanged {s['unchanged']:6d}  adjusted {s['adjusted']:4d}  "
              f"dropped_outside {s['dropped_outside']:4d}  dropped_sliver {s['dropped_sliver']:4d}")
    print(f"\nmin_visible = {min_visible} - record this in your report")


def step7_visual_check(train_samples=10):
    banner("STEP 7  visual check")
    import cv2
    import random

    os.makedirs(CHECK_DIR, exist_ok=True)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

    def annotate(img_path, out_path):
        img = cv2.imread(img_path)
        if img is None:
            print("   could not read", img_path)
            return 0
        h, w = img.shape[:2]
        lab = label_for(img_path)
        n = 0
        for line in open(lab):
            if not line.strip():
                continue
            parts = line.split()
            c = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            col = colors[c % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, NAMES[c], (x1, max(y1 - 5, 12)), 0, 0.5, col, 2)
            n += 1
        cv2.imwrite(out_path, img)
        return n

    for split in ["valid", "test", "train"]:
        files = list_images(split)
        require(files, f"no images in {images_dir(split)}")
        picks = files if split != "train" else random.sample(files, min(train_samples, len(files)))
        print(f"\n{split}:")
        for p in picks:
            n = annotate(p, os.path.join(CHECK_DIR, f"{split}_{os.path.basename(p)}"))
            if split != "train":
                print(f"   {os.path.basename(p):40s} {n:3d} boxes")

    print("\nOpen this folder and look at every image:")
    print("  ", CHECK_DIR)
    print("Check: boxes sit on objects, class names are correct.")


AUG_PATTERN = r"(_|-)?(brightness|flip(ud|lr)?|haze|motion[_-]?blur|blur|rot(ation)?|shadow|zoom([_-]?(in|out))?|scale|contrast)[_-]?\d*"


def source_stem(path):
    s = Path(path).stem
    prev = None
    while prev != s:
        prev = s
        s = re.sub(AUG_PATTERN + r"$", "", s, flags=re.I)
    return s.strip("_- ")


def step8_leakage():
    banner("STEP 8  leakage audit")
    import numpy as np
    from PIL import Image

    tr, va, te = list_images("train"), list_images("valid"), list_images("test")
    require(tr, "no training images - run step 3 first")

    TR = {source_stem(p) for p in tr}
    VA = {source_stem(p) for p in va}
    TE = {source_stem(p) for p in te}

    print(f"train images: {len(tr):5d}   distinct source stems: {len(TR)}")
    print(f"valid images: {len(va):5d}   distinct source stems: {len(VA)}")
    print(f"test  images: {len(te):5d}   distinct source stems: {len(TE)}")
    if TR:
        print(f"average augmented variants per source: {len(tr) / len(TR):.1f}")

    print("\nfilename overlap")
    print("  train & valid:", sorted(TR & VA) or "none")
    print("  train & test :", sorted(TR & TE) or "none")

    def phash(p):
        a = np.asarray(Image.open(p).convert("L").resize((16, 16)), dtype=float)
        return tuple((a > a.mean()).flatten())

    print("\ncomputing perceptual hashes for the training set...")
    trh = {}
    for p in tr:
        trh.setdefault(phash(p), []).append(os.path.basename(p))

    print("pixel-level overlap")
    for name, group in [("valid", va), ("test", te)]:
        hits = []
        for p in group:
            h = phash(p)
            if h in trh:
                hits.append((os.path.basename(p), trh[h][0]))
        print(f"  {name}: {len(hits)}/{len(group)} images match a training image")
        for a, b in hits[:5]:
            print(f"      {a}  ~=  {b}")

    print("\nWrite these numbers down. If there is leakage you still proceed,")
    print("but you report it and interpret the baseline accordingly.")


def step9_write_yaml():
    banner("STEP 9  data.yaml")
    path_fwd = DST.replace("\\", "/")
    lines = [f"path: {path_fwd}",
             "train: train/images",
             "val: valid/images",
             "test: test/images",
             "names:"]
    for i, n in enumerate(NAMES):
        lines.append(f"  {i}: {n}")
    text = "\n".join(lines) + "\n"

    os.makedirs(DST, exist_ok=True)
    open(YAML_PATH, "w").write(text)
    print("written to", YAML_PATH)
    print()
    print(text)


def step10_smoke():
    banner("STEP 10  smoke test - 3 epochs")
    from ultralytics import YOLO
    import torch
    require(os.path.isfile(YAML_PATH), "data.yaml missing - run step 9 first")

    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=YAML_PATH,
        epochs=3, imgsz=IMGSZ, batch=BATCH,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0, amp=True, seed=0, patience=0,
        name="smoke", project=RUNS_DIR, exist_ok=True,
    )
    print("\nIf loss decreased and no errors appeared, you are ready to train.")


def train_baseline(run_name, static_only):
    banner(f"TRAIN  {run_name}  (static_only={static_only})")
    from ultralytics import YOLO
    import torch
    require(os.path.isfile(YAML_PATH), "data.yaml missing - run step 9 first")

    kwargs = dict(
        data=YAML_PATH,
        epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, freeze=FREEZE,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0, amp=True, seed=0, patience=0,
        name=run_name, project=RUNS_DIR, exist_ok=True,
    )
    if static_only:
        kwargs.update(
            mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            degrees=0.0, translate=0.0, scale=0.0,
            fliplr=0.0, flipud=0.0,
        )
    YOLO(MODEL_WEIGHTS).train(**kwargs)


def step12_evaluate(run_name):
    banner(f"EVALUATE  {run_name}")
    from ultralytics import YOLO
    weights = os.path.join(RUNS_DIR, run_name, "weights", "best.pt")
    require(os.path.isfile(weights), f"weights not found: {weights}")

    model = YOLO(weights)
    rows = []
    for split in ["val", "test"]:
        r = model.val(data=YAML_PATH, split=split, plots=True,
                      project=RUNS_DIR, name=f"{run_name}_{split}", exist_ok=True)
        print(f"\n--- {split} ---")
        print(f"precision {r.box.mp:.4f}   recall {r.box.mr:.4f}")
        print(f"mAP@0.5   {r.box.map50:.4f}   mAP@0.5:0.95 {r.box.map:.4f}")
        for i, name in enumerate(NAMES):
            print(f"   {name:16s} AP50 {r.box.ap50[i]:.4f}   AP50-95 {r.box.ap[i]:.4f}")
            rows.append([run_name, split, name,
                         f"{r.box.ap50[i]:.4f}", f"{r.box.ap[i]:.4f}"])
        rows.append([run_name, split, "ALL",
                     f"{r.box.map50:.4f}", f"{r.box.map:.4f}"])

    out = os.path.join(RUNS_DIR, f"{run_name}_baseline.csv")
    with open(out, "w") as f:
        f.write("run,split,class,AP50,AP50_95\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print("\nsaved baseline table to", out)


def step7b_class_mapping():
    from collections import defaultdict
    banner("STEP 7b  class id <-> filename prefix")

    table = defaultdict(Counter)
    for split in ["valid", "test"]:
        for p in list_labels(split):
            prefix = Path(p).stem.split("-")[0].lower()
            for line in open(p):
                if line.strip():
                    table[prefix][int(line.split()[0])] += 1

    ids = sorted({c for row in table.values() for c in row})
    print("prefix".ljust(12) + "".join(f"id {c}".rjust(9) for c in ids) + "    total")
    for pre in sorted(table):
        row = table[pre]
        tot = sum(row.values())
        print(pre.ljust(12) + "".join(f"{100*row[c]/tot:8.1f}%" for c in ids) + f"{tot:9d}")
    print("\nFor each prefix the dominant id should be that object's class.")


if __name__ == "__main__":
    train_baseline("baseline_A20", static_only=True)
    step12_evaluate("baseline_A20")
    print("\ndone")
