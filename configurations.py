# create a sample camera_config
camera_config = {
  "videos": [
    {"path": "/content/videos/cam1.mp4", "id":"cam1", "lat":12.9715987, "lon":77.594566, "desc":"Gate cam", "start_time": None},
    {"path": "/content/videos/cam2.mp4", "id":"cam2", "lat":12.967864, "lon":77.582076, "desc":"Street cam 2", "start_time": None},
    {"path": "/content/videos/cam3.mp4", "id":"cam3", "lat":12.9736, "lon":77.5960, "desc":"Alley cam", "start_time": None},
    # add more cameras if you want; the detection script will process three videos at a time
  ]
}
import json, os
os.makedirs("/content/config", exist_ok=True)
with open("/content/config/camera_config.json","w") as f:
    json.dump(camera_config, f, indent=2)
print("Saved /content/config/camera_config.json - edit lat/lon and paths as needed.")
