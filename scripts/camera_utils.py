#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Tuple
import pyrealsense2 as rs
from tunables import CAMERA_SERIALS_FILE, FALLBACK_DEPTH_SCALE

# share cam2 depth-scale
import threading
_CAM2_DEPTH_SCALE_LOCK = threading.Lock()
_CAM2_DEPTH_SCALE = None

def set_cam2_depth_scale(val: float):
    global _CAM2_DEPTH_SCALE
    with _CAM2_DEPTH_SCALE_LOCK:
        _CAM2_DEPTH_SCALE = val

def get_cam2_depth_scale():
    with _CAM2_DEPTH_SCALE_LOCK:
        return _CAM2_DEPTH_SCALE

def load_serial_map() -> Dict[str, str]:
    """
    camera_serials.txt format:
      cam1: <serial>
      cam2: <serial>
      cam3: <serial>
    We'll invert to serial->label.
    """
    serial_to_label: Dict[str, str] = {}
    try:
        with open(CAMERA_SERIALS_FILE, "r") as f:
            for line in f:
                if ":" in line:
                    label, serial = line.strip().split(":")
                    serial_to_label[serial.strip()] = label.strip()
    except Exception:
        pass
    return serial_to_label

def write_camera_info(profile: rs.pipeline_profile, serial: str, label: str,
                      out_txt: Path, out_json: Path) -> Tuple[dict, float]:
    device = profile.get_device()
    sensors = device.query_sensors()

    intr_json = {}
    with open(out_txt, "w") as f:
        f.write(f"Camera Label: {label}\n")
        f.write(f"Serial Number: {serial}\n")
        f.write(f"Firmware Version: {device.get_info(rs.camera_info.firmware_version)}\n")
        f.write(f"USB Port ID: {device.get_info(rs.camera_info.physical_port)}\n")
        f.write(f"Product Line: {device.get_info(rs.camera_info.product_line)}\n\n")
        for sensor in sensors:
            f.write(f"[Sensor: {sensor.get_info(rs.camera_info.name)}]\n")
            for opt in sensor.get_supported_options():
                try:
                    val = sensor.get_option(opt)
                    f.write(f"  {opt.name}: {val}\n")
                except Exception:
                    continue
            f.write("\n")
        f.write("[Active Streams]\n")

    for s in profile.get_streams():
        try:
            vs = s.as_video_stream_profile()
            intr = vs.get_intrinsics()
            stream_key = f"{s.stream_type().name.lower()}_{s.format().name.lower()}"
            intr_json[stream_key] = {
                "width": vs.width(),
                "height": vs.height(),
                "fps": vs.fps(),
                "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                "model": intr.model.name, "coeffs": list(intr.coeffs),
            }
        except Exception:
            pass

    # depth scale
    try:
        depth_scale = float(device.first_depth_sensor().get_depth_scale())
    except Exception:
        depth_scale = None
    if depth_scale is None or depth_scale <= 0:
        depth_scale = FALLBACK_DEPTH_SCALE

    intr_json["alignment"] = {"depth_to_color": True, "alignment_target": "color", "use_intrinsics": "color"}
    intr_json["depth_scale_m"] = depth_scale

    with open(out_json, "w") as jf:
        json.dump(intr_json, jf, indent=2)

    return intr_json, depth_scale
