# === Colab cell: sequential detection (one video at a time) + mapping === (detection code)
import os, csv, json, math
from datetime import datetime, timedelta
import pandas as pd
import folium
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# CONFIG - update if different
WEIGHTS = "runs/detect/suspect_train/weights/best.pt"  # from training step
CAMERA_CONFIG = "/content/config/camera_config.json"
OUTPUT_CSV = "/content/detections.csv"
MAP_HTML = "/content/suspect_path_map.html"
CONF_THRESH = 0.4
DEVICE = "cpu"  # use "cuda" if GPU available and you want to enable it
VERIFY_FIRST_N = 3   # set to 3 to "verify 3 videos alternatively"; set to None to process all

# load camera config
if not os.path.exists(CAMERA_CONFIG):
    raise SystemExit("camera_config.json missing at /content/config/camera_config.json. Create it with video paths and lat/lon.")
with open(CAMERA_CONFIG, "r") as f:
    cams_cfg = json.load(f).get("videos", [])

print("Cameras found in config:")
for idx, c in enumerate(cams_cfg, start=1):
    print(f" {idx}. {c['id']} => {c['path']}")

# load model
if not os.path.exists(WEIGHTS):
    raise SystemExit(f"Weights not found at: {WEIGHTS}")
model = YOLO(WEIGHTS)

def frame_time_from_idx(frame_idx, fps, start_time=None):
    secs = frame_idx / (fps or 25.0)
    if start_time:
        try:
            base = datetime.fromisoformat(start_time)
            return (base + timedelta(seconds=secs)).isoformat()
        except Exception:
            return str(timedelta(seconds=int(secs)))
    else:
        return str(timedelta(seconds=int(secs)))

def detect_in_video(video_meta):
    vid_path = video_meta["path"]
    cam_id = video_meta.get("id","unknown")
    start_time = video_meta.get("start_time")  # optional ISO8601
    rows = []

    if not os.path.exists(vid_path):
        print(f"[WARN] Video not found for {cam_id}: {vid_path}")
        return rows

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_idx = 0
    pbar = tqdm(total=total, desc=f"Detect {cam_id}", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run detection (frame-by-frame)
        # use DEVICE variable above (cpu or cuda)
        res = model.predict(frame, imgsz=640, conf=CONF_THRESH, verbose=False, device=DEVICE)
        if len(res) > 0:
            r = res[0]
            boxes = getattr(r, "boxes", [])
            for b in boxes:
                conf = float(b.conf)
                if conf < CONF_THRESH:
                    continue
                # get bbox
                # safe access to tensor -> numpy (handles CPU-only)
                try:
                    xyxy = b.xyxy[0].cpu().numpy()
                except Exception:
                    xyxy = b.xyxy[0].numpy()
                x1,y1,x2,y2 = map(float, xyxy)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                ts = frame_time_from_idx(frame_idx, fps, start_time)
                rows.append({
                    "video_path": vid_path,
                    "camera_id": cam_id,
                    "camera_lat": video_meta.get("lat"),
                    "camera_lon": video_meta.get("lon"),
                    "timestamp": ts,
                    "frame_idx": frame_idx,
                    "conf": conf,
                    "bbox_x1": x1, "bbox_y1": y1,
                    "bbox_x2": x2, "bbox_y2": y2,
                    "bbox_cx": cx, "bbox_cy": cy
                })

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return rows

# === Run videos sequentially (one by one) ===
all_rows = []
to_process = cams_cfg if VERIFY_FIRST_N is None else cams_cfg[:VERIFY_FIRST_N]

print(f"\nProcessing {len(to_process)} video(s) sequentially (one at a time)...\n")
for meta in to_process:
    cam_id = meta.get("id", os.path.basename(meta.get("path","unknown")))
    print(f"--> Processing {cam_id} ...")
    rows = detect_in_video(meta)
    all_rows.extend(rows)

    # print concise status check (check mark if any detection)
    if len(rows) > 0:
        print(f"   ✅ Suspect detected in {cam_id} — detections: {len(rows)}")
    else:
        print(f"   ❌ No suspect detected in {cam_id}")

# Always write CSV (even when empty)
df = pd.DataFrame(all_rows, columns=[
    "video_path","camera_id","camera_lat","camera_lon","timestamp","frame_idx","conf",
    "bbox_x1","bbox_y1","bbox_x2","bbox_y2","bbox_cx","bbox_cy"
])
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[INFO] Saved detections CSV at {OUTPUT_CSV}, rows = {len(df)}")

# --- Map generation (same behavior as before) ---
def build_map_from_df(df, cam_cfg_list, out_html):
    # center roughly on mean of camera locations
    mean_lat = sum([c["lat"] for c in cam_cfg_list]) / len(cam_cfg_list)
    mean_lon = sum([c["lon"] for c in cam_cfg_list]) / len(cam_cfg_list)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=16)

    if df.empty:
        print("[INFO] No detections in CSV. Plotting camera markers only.")
        for c in cam_cfg_list:
            folium.Marker([c["lat"], c["lon"]], popup=f"{c['id']} (no suspect detected)").add_to(m)
        m.save(out_html)
        return out_html

    # prepare chronological route of unique camera hits
    df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df_sorted = df.sort_values('timestamp_parsed')

    route = []
    seen = set()
    for _, r in df_sorted.iterrows():
        cid = r["camera_id"]
        if cid in seen:
            continue
        seen.add(cid)
        route.append((float(r["camera_lat"]), float(r["camera_lon"]), cid, r["timestamp_parsed"]))

    detected_cam_ids = {c[2] for c in route}

    # draw markers
    for c in cam_cfg_list:
        cid = c["id"]
        popup = f"{cid}<br>{c.get('desc','')}"
        if cid in detected_cam_ids:
            folium.CircleMarker([c["lat"], c["lon"]], radius=7, color="red", fill=True, fill_opacity=0.9, popup=popup).add_to(m)
        else:
            folium.CircleMarker([c["lat"], c["lon"]], radius=5, color="blue", fill=True, fill_opacity=0.6, popup=popup).add_to(m)

    # draw polyline for route (if >=2 points)
    coords = [(lat, lon) for (lat, lon, _, _) in route]
    if len(coords) >= 2:
        folium.PolyLine(locations=coords, weight=4, color="red", opacity=0.8).add_to(m)

    # add numbered popups for order
    for idx, (lat, lon, cid, ts) in enumerate(route, start=1):
        folium.map.Marker([lat, lon], popup=f"{idx}. {cid} - {ts}",
                          icon=folium.DivIcon(html=f"""<div style="font-size:12px;color:black;
                              background:rgba(255,255,255,0.7);padding:2px;border-radius:3px">{idx}</div>""")).add_to(m)

    m.save(out_html)
    return out_html

map_file = build_map_from_df(df, cams_cfg, MAP_HTML)
print("[INFO] Map saved to:", map_file)

# display in colab if available
from IPython.display import IFrame
IFrame(map_file, width=1000, height=600)
