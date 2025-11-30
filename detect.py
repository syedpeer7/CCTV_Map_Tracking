#- auto-label suspect images (place your images data in a folder suspect_images)
import os, glob, json
from ultralytics import YOLO
import shutil

# CONFIG - change these paths if using Drive
SUSPECT_IMG_DIR = "/content/suspect_images"   # put your uploaded suspect images here
DATASET_ROOT = "/content/suspect_dataset"     # dataset will be created here
TRAIN_RATIO = 0.8

os.makedirs(SUSPECT_IMG_DIR, exist_ok=True)
os.makedirs(DATASET_ROOT, exist_ok=True)

# create folders
for sub in ["train/images","val/images","train/labels","val/labels"]:
    os.makedirs(os.path.join(DATASET_ROOT, sub), exist_ok=True)

# load pretrained COCO person detector (yolov8n)
detector = YOLO("yolov8n.pt")

# collect images
img_paths = sorted([p for ext in ("*.jpg","*.jpeg","*.png") for p in glob.glob(os.path.join(SUSPECT_IMG_DIR, ext))])
if len(img_paths) < 2:
    raise SystemExit("Please upload at least 2 suspect images into SUSPECT_IMG_DIR.")

# split train/val
n_train = int(len(img_paths) * TRAIN_RATIO)
train_imgs = img_paths[:n_train]
val_imgs = img_paths[n_train:]

def create_label_for_image(src_path, dst_img_path, dst_label_path):
    # run person detector on image and take the largest person bbox
    res = detector.predict(src_path, imgsz=640, conf=0.3, device='cpu')  # <-- FIXED
    try:
        boxes = res[0].boxes
    except:
        boxes = None

    if boxes is None or len(boxes) == 0:
        print("No person detected in", src_path, " — skipping auto-label.")
        return False

    bboxes = []
    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        x1,y1,x2,y2 = xyxy
        conf = float(b.conf)
        bboxes.append((x1,y1,x2,y2,conf))

    # pick largest bbox
    bboxes_sorted = sorted(bboxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
    x1,y1,x2,y2,conf = bboxes_sorted[0]

    import cv2
    img = cv2.imread(src_path)
    h,w = img.shape[:2]

    cx = ((x1+x2)/2)/w
    cy = ((y1+y2)/2)/h
    bw = (x2-x1)/w
    bh = (y2-y1)/h

    # save label + image
    shutil.copy2(src_path, dst_img_path)
    with open(dst_label_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    return True


# process train images
for i, p in enumerate(train_imgs):
    dst_img = os.path.join(DATASET_ROOT, "train/images", os.path.basename(p))
    dst_lbl = os.path.join(DATASET_ROOT, "train/labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
    ok = create_label_for_image(p, dst_img, dst_lbl)
    if not ok:
        print("Consider manually labeling", p)

for i, p in enumerate(val_imgs):
    dst_img = os.path.join(DATASET_ROOT, "val/images", os.path.basename(p))
    dst_lbl = os.path.join(DATASET_ROOT, "val/labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
    ok = create_label_for_image(p, dst_img, dst_lbl)
    if not ok:
        print("Consider manually labeling", p)

# create data.yaml for YOLO training
data_yaml = {
    "names": ["suspect"],
    "nc": 1,
    "train": os.path.join(DATASET_ROOT, "train/images"),
    "val": os.path.join(DATASET_ROOT, "val/images")
}
with open(os.path.join(DATASET_ROOT, "data.yaml"), "w") as f:
    import yaml
    yaml.dump(data_yaml, f)

print("Dataset ready at", DATASET_ROOT)
