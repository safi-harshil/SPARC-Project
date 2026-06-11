#!/usr/bin/env python3
"""
Main script to orchestrate:
- Detect connected RealSense cameras
- Start per-cam capture threads
- Start optional movement, emotion, and object trigger threads
- Start optional audio threads
- Optional unified preview window (2x2 grid)
Controls (terminal):
  • SPACE → toggle Pause/Resume
  • g     → reopen the unified preview grid (if closed)
  • q     → close ONLY the preview grid (pipeline continues; handled in PreviewGrid)
  • ESC   → Stop gracefully (terminal or window)
  • Ctrl+C → Stop gracefully (SIGINT)
"""

import os, sys, time, threading, queue, signal, argparse
from pathlib import Path
from typing import Dict, Set
import subprocess

import pyrealsense2 as rs

from tunables import *
from preview_grid import PreviewGrid
from capture_worker import capture_worker
from movement_processor import processor_worker
from emotion_processor import emotion_worker
from object_worker import object_worker  # ✅ NEW
from camera_utils import load_serial_map
from audio_worker import audio_worker
from eye_tracking import eye_tracking_worker  # ✅ NEW
import certifi
# import rclpy

# from ros_publisher_node import ROS2PublisherNode


# shared control flags
from control_flags import pause_event, stop_event

# centralized logging
from logger_utils import (
    get_preview_logger,
    get_object_trigger_logger,
    log_exception,
)

# event triggers
from event_triggers import ObjectUntouchedTrigger


def _sig_stop(signum, frame):
    print("\n[⛔] SIGINT → stopping…", flush=True)
    stop_event.set()
signal.signal(signal.SIGINT, _sig_stop)



# --- Terminal keyboard listener (SPACE / 'g' / ESC) ---
def start_keyboard_listener(preview_ref):
    """Listen on the terminal for SPACE (pause), 'g' (reopen grid), ESC (stop)."""
    if not sys.stdin.isatty():
        print("[INFO] Keyboard listener disabled (stdin is not a TTY).", flush=True)
        return None

    import termios, tty, select
    fd = sys.stdin.fileno()
    try:
        old_attrs = termios.tcgetattr(fd)
    except Exception:
        print("[WARN] Could not configure terminal keyboard listener.", flush=True)
        return None

    def _run():
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == ' ':
                    if pause_event.is_set():
                        pause_event.clear()
                        print("[▶] Resume (space)", flush=True)
                    else:
                        pause_event.set()
                        print("[⏸] Pause (space)", flush=True)
                elif ch in ('g', 'G'):
                    if preview_ref is not None:
                        try:
                            preview_ref.reopen_window()
                            print("[🪟] Preview reopened.", flush=True)
                        except Exception:
                            pass
                elif ch == '\x1b':  # ESC
                    print("[⛔] ESC pressed (terminal) → stopping.", flush=True)
                    stop_event.set()
                    try:
                        os.kill(os.getpid(), signal.SIGINT)
                    except Exception:
                        pass
                    break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass

    t = threading.Thread(target=_run, name="kbd-listener", daemon=True)
    t.start()
    return t


def build_argparser():
    ap = argparse.ArgumentParser(description="Real-time unified capture+process+audio pipeline (modular)")
    ap.add_argument("--output-dir", required=True, help="Base output directory")
    ap.add_argument("--duration-sec", type=float, required=True, help="ACTIVE duration in seconds")
    ap.add_argument("--save-every", type=int, default=1, help="Save raw color/depth every Nth frame (0=off)")
    ap.add_argument("--filters", choices=["on","off"], default="off", help="Depth filters on/off")
    ap.add_argument("--viz-live", choices=["off","on"], default="on", help="Preview window (unified 2x2 grid)")
    ap.add_argument("--force-flip", choices=["flip","same"], default="flip", help="Global handedness flip baseline")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame for movement")
    ap.add_argument("--backpressure", choices=["drop-latest","block"], default="drop-latest", help="Processor queue policy")
    ap.add_argument("--csv-flush", type=int, default=30, help="CSV flush interval (frames) for movement landmarks")
    ap.add_argument("--log-flush-sec", type=int, default=5, help="Logger flush interval")
    ap.add_argument("--emo-history", type=int, default=240, help="Frames kept in on-screen VA plot (per cam)")
    ap.add_argument("--emo-stride", type=int, default=1, help="Process every Nth frame for emotion")
    ap.add_argument("--emo-csv-flush", type=int, default=30, help="CSV flush interval (frames) for emotion")
    

    #eye tracking
    ap.add_argument("--eye-stride", type=int, default=1, help="Process every Nth frame for eye tracking")  # ✅ NEW
    ap.add_argument("--eye-csv-flush", type=int, default=30, help="CSV flush interval (frames) for eye tracking")  # ✅ NEW
    ap.add_argument(
    "--process-eye-cams",
    nargs="*",
    help="Labels to process for eye tracking."
)

    # camera selection
    ap.add_argument("--process-mov-cams", nargs="*", help="Labels to process for hand movement.")
    ap.add_argument("--process-emo-cams", nargs="*", help="Labels to process for emotion (valence/arousal).")
    ap.add_argument("--process-obj-cams", nargs="*", help="Labels to process for object interaction triggers.")  # ✅ NEW

    # audio
    ap.add_argument("--audio-out", default=None, help="Audio output directory (default: <output-dir>/audio)")
    ap.add_argument("--audio-duration-sec", type=float, default=0, help="Active duration for audio (0=off)")
    ap.add_argument("--rate", type=int, choices=[44100,48000], default=44100, help="Audio sample rate")

    # annotated movement preview saver
    ap.add_argument("--viz-save-every", type=int, default=3,
                    help="Save annotated movement preview every N frames (0 = OFF)")

    # notes UI
    ap.add_argument("--notes", choices=["off","on"], default="on",
                    help="Notes UI (chat messenger panel) on/off")

    # event checker toggles
    ap.add_argument("--no-event-checker", action="store_true",
                    help="Disable the R0 expected-speed event checker (default: enabled)")
    ap.add_argument("--no-object-trigger", action="store_true",
                    help="Disable object untouched trigger (default: enabled)")
    return ap


def main():

    # # Initialize rclpy once, get singleton node
    # node = ROS2PublisherNode.get_instance()

    # # Run ROS2 event loop in a background thread (so rest of code runs)
    # spin_thread = threading.Thread(
    #     target=rclpy.spin,
    #     args=(node,),
    #     daemon=True,
    #     name="ros2-spin-thread"
    # )
    # spin_thread.start()

    try:


        args = build_argparser().parse_args()

        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Central loggers
        preview_logger = get_preview_logger(out_dir, flush_sec=max(1, args.log_flush_sec))

        # Discover cams
        serial_to_label = load_serial_map()
        ctx = rs.context()
        connected = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
        active = [(s, serial_to_label.get(s, f"cam_{s[-4:]}")) for s in connected]
        if not active:
            print("[ERROR] No RealSense cameras found.")
            return
        print(f"[INFO] Connected cams: {', '.join([f'{lab}({s})' for s,lab in active])}")

        # Camera selections
        process_set_mov = set([lab.strip() for lab in (args.process_mov_cams or []) if lab.strip()])
        process_set_emo = set([lab.strip() for lab in (args.process_emo_cams or []) if lab.strip()])
        process_set_obj = set([lab.strip() for lab in (args.process_obj_cams or []) if lab.strip()])
        process_set_eye = set([lab.strip() for lab in (args.process_eye_cams or []) if lab.strip()])  # ✅ NEW

        print(f"[INFO] Movement cams: {sorted(process_set_mov) if process_set_mov else 'NONE'}")
        print(f"[INFO] Emotion cams:  {sorted(process_set_emo) if process_set_emo else 'NONE'}")
        print(f"[INFO] Object-trigger cams: {sorted(process_set_obj) if process_set_obj else 'NONE'}")
        print(f"[INFO] Eye-tracking cams: {sorted(process_set_eye) if process_set_eye else 'NONE'}") #Eye tracking

        # Speed trigger
        if not args.no_event_checker and process_set_mov:
            try:
                refcsv = SPEED_TRIGGER_REFCSV_PATH
            except NameError:
                refcsv = "(default in trigger)"
            print(f"[INFO] Speed-trigger checker: ENABLED (reference CSV: {refcsv})")
        else:
            print("[INFO] Speed-trigger checker: DISABLED")

        # Object trigger setup
        obj_trigs = {}
        if not args.no_object_trigger:
            if process_set_obj:
                try:
                    for lab in sorted(process_set_obj):
                        cam_dir = out_dir / lab
                        obj_log = get_object_trigger_logger(cam_dir, flush_sec=max(1, args.log_flush_sec))
                        cam_dir = out_dir / lab
                        trig = ObjectUntouchedTrigger(cam_dir, cam_label=lab, logger=obj_log)
                        setattr(trig, "target_cam_labels", [lab])
                        obj_trigs[lab] = trig
                    print(f"[INFO] Object-trigger checker: ENABLED for {len(obj_trigs)} cam(s): {', '.join(obj_trigs.keys())}")
                except Exception as e:
                    print(f"[WARN] Object-trigger init failed: {e}")
            else:
                print("[INFO] Object-trigger checker: ENABLED but no object cams selected")
        else:
            print("[INFO] Object-trigger checker: DISABLED")

        # Unified preview
        PREVIEW = None
        if args.viz_live == "on":
            PREVIEW = PreviewGrid(
                title="Unified Preview",
                history_len=max(240, args.emo_history),
                target_fps=30,
                logger=preview_logger,
                enable_event_overlay=(not args.no_event_checker),
                enable_object_overlay=(not args.no_object_trigger),
                show_cv2_window=(args.notes != "on"),   # ✅ IMPORTANT
            )
            PREVIEW.start()
            # ✅ ensure the X-axis lock in the cumulative plot matches runtime duration exactly
            try:
                PREVIEW.set_total_duration(float(args.duration_sec))
            except Exception:
                pass

        # ✅ Notes UI (chat messenger panel) — runs in its own GUI loop (typing does not affect terminal keys)
        NOTES_APP = None
        NOTES_WIN = None

        if (args.notes == "on") and (PREVIEW is not None):
            try:
                from notes_ui import start_notes_ui
                NOTES_APP, NOTES_WIN = start_notes_ui(PREVIEW, out_dir, logger=preview_logger)
                preview_logger.info("Notes UI started")
                preview_logger.periodic_flush()
            except Exception as e:
                print(f"[WARN] Notes UI not started: {e}", flush=True)

        _kbd = start_keyboard_listener(PREVIEW)

        # Queues & threads
        cap_threads, proc_threads = [], []
        queues_mov: Dict[str, queue.Queue] = {}
        queues_emo: Dict[str, queue.Queue] = {}
        queues_obj: Dict[str, queue.Queue] = {}  # ✅ NEW
        queues_eye: Dict[str, queue.Queue] = {}  # for eye tracking 

        obj_threads = []   # <-- ADD THIS

        for serial, label in active:
            q_mov = None
            q_emo = None
            q_obj = None  # ✅ NEW
            q_eye = None  # for eye tracking 

            # Movement
            if label in process_set_mov:
                q_mov = queue.Queue(maxsize=8)
                queues_mov[label] = q_mov
                t_proc_mov = threading.Thread(
                    target=processor_worker,
                    args=(
                        label,
                        q_mov,
                        out_dir,
                        args.force_flip,
                        max(1, args.stride),
                        max(1, args.csv_flush),
                        max(1, args.log_flush_sec),
                        PREVIEW,
                        max(0, args.viz_save_every),
                        float(args.duration_sec),  # 👈 NEW: total task duration (T_total)
                    ),
                    kwargs=dict(event_checker_enabled=(not args.no_event_checker)),
                    daemon=True,
                    name=f"proc-mov-{label}",
                )
                t_proc_mov.start()
                proc_threads.append(t_proc_mov)

            # Emotion
            if label in process_set_emo:
                q_emo = queue.Queue(maxsize=8)
                queues_emo[label] = q_emo
                t_proc_emo = threading.Thread(
                    target=emotion_worker,
                    args=(label, q_emo, out_dir, max(1, args.emo_stride),
                        max(1, args.emo_csv_flush), max(1, args.log_flush_sec),
                        max(10, args.emo_history), PREVIEW),
                    daemon=True, name=f"proc-emo-{label}"
                )
                t_proc_emo.start()
                proc_threads.append(t_proc_emo)

            #EYE Tracking
            if label in process_set_eye:
                q_eye = queue.Queue(maxsize=8)
                queues_eye[label] = q_eye
                t_proc_eye = threading.Thread(
                    target=eye_tracking_worker,
                    args=(label, q_eye, out_dir),
                    daemon=True, name=f"proc-eye-{label}"
                )
                t_proc_eye.start()
                proc_threads.append(t_proc_eye)

            # Object interaction trigger lane
            if label in process_set_obj:
                q_obj = queue.Queue(maxsize=8)
                queues_obj[label] = q_obj
                trig = obj_trigs.get(label)
                t_proc_obj = threading.Thread(
                    target=object_worker,
                    args=(label, q_obj, out_dir, max(1, args.log_flush_sec), trig),
                    kwargs=dict(preview=PREVIEW),  # ✅ forward preview for mask overlay
                    daemon=True, name=f"proc-obj-{label}"
                )
                t_proc_obj.start()
                obj_threads.append(t_proc_obj)

            # Capture
            # t_cap = threading.Thread(
            #     target=capture_worker,
            #     args=(serial, label, out_dir, float(args.duration_sec), max(0, args.save_every),
            #         (args.filters == "on"), queues_mov.get(label, None),
            #         queues_emo.get(label, None), args.backpressure,
            #         queues_obj.get(label, None)),  # ✅ NEW
            #         queues_eye.get(label, None),  # for eye tracking (optional, not yet implemented in capture_worker) --- IGNORE for now
            #     daemon=True, name=f"cap-{label}"
            # )
            # t_cap.start()
            # cap_threads.append(t_cap)

            # Capture
            t_cap = threading.Thread(
                target=capture_worker,
                args=(
                    serial,
                    label,
                    out_dir,
                    float(args.duration_sec),
                    max(0, args.save_every),
                    (args.filters == "on"),
                    queues_mov.get(label, None),
                    queues_emo.get(label, None),
                    args.backpressure,
                    queues_obj.get(label, None),
                    queues_eye.get(label, None),   # ✅ eye queue
                ),
                daemon=True,
                name=f"cap-{label}"
            )

            t_cap.start()
            cap_threads.append(t_cap) #thread list for capture threads (one per cam)

        # --- Audio ---
        def _list_alsa_hw_devices():
            try:
                out = subprocess.check_output(["arecord", "-l"], stderr=subprocess.STDOUT, text=True)
            except Exception:
                return []
            found = []
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("card "):
                    try:
                        parts = [p.strip() for p in line.split(",")]
                        cidx = int(parts[0].split()[1].rstrip(":"))
                        didx = int(parts[1].split()[1].rstrip(":"))
                        found.append(f"hw:{cidx},{didx}")
                    except Exception:
                        continue
            return found

        # ✅ Minimal addition:
        # Some devices (like RODE Wireless GO II RX) are better addressed by CARD=... style.
        # arecord -l won’t list them in that exact string form, so we also scan arecord -L.
        def _list_alsa_named_devices():
            try:
                out = subprocess.check_output(["arecord", "-L"], stderr=subprocess.STDOUT, text=True)
            except Exception:
                return []
            found = []
            for line in out.splitlines():
                line = line.strip()
                # Keep only hw:CARD=...,DEV=... style entries
                if line.startswith("hw:CARD=") and ",DEV=" in line:
                    found.append(line)
            return found

        audio_threads = []
        aud_dir = Path(args.audio_out) if args.audio_out else (out_dir / "audio")
        if args.audio_duration_sec > 0:
            present = set(_list_alsa_hw_devices())
            present_named = set(_list_alsa_named_devices())

            enabled = [d for d in VALID_MIC_IDS if (d in present) or (d in present_named)]
            if not enabled:
                print("[INFO] 🎙 No whitelisted mics detected; skipping audio.")
            else:
                for dev in enabled:
                    t = threading.Thread(
                        target=audio_worker,
                        args=(dev, aud_dir, float(args.audio_duration_sec), int(args.rate), preview_logger),
                        daemon=True, name=f"aud-{dev}"
                    )
                    audio_threads.append(t)
                for t in audio_threads:
                    t.start()
                print(f"[INFO] 🎙 Audio enabled on {len(audio_threads)} mic(s): {', '.join(enabled)} → {aud_dir}")

        # --- Wait and cleanup ---
        try:
            while True:
                alive = any(t.is_alive() for t in cap_threads)
                if not alive:
                    break

                if stop_event.is_set():
                    break

                # ✅ keep Qt responsive
                if NOTES_APP is not None:
                    try:
                        NOTES_APP.processEvents()
                    except Exception:
                        pass

                time.sleep(0.01)
        except Exception as e:
            log_exception(preview_logger, "Error while joining capture threads", e)
        finally:
            for q in list(queues_mov.values()) + list(queues_emo.values()) + list(queues_obj.values()) + list(queues_eye.values()):
                setattr(q, "closed", True)

            for t in proc_threads + obj_threads:
                try:
                    t.join()
                except Exception as e:
                    log_exception(preview_logger, "Error while joining processor threads", e)

            for t in audio_threads:
                try:
                    t.join()
                except Exception as e:
                    log_exception(preview_logger, "Error while joining audio threads", e)

            # Notes UI cleanup
            if NOTES_WIN is not None:
                try:
                    NOTES_WIN.close()
                except Exception:
                    pass

            if PREVIEW is not None:
                PREVIEW.stop()

            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

        # --- Force log flush ---
        preview_logger.periodic_flush(force=True)
        for _lab, _trig in obj_trigs.items():
            try:
                _trig.logger.periodic_flush(force=True)
            except Exception:
                pass
        print("[🏁] Done.")

    except KeyboardInterrupt:
        print("\n[⛔] KeyboardInterrupt → stopping…", flush=True)

    # finally:
    #     # --- Shutdown ROS2 node gracefully ---
    #     print("[🧹] Shutting down ROS2...")
    #     ROS2PublisherNode.shutdown()
    #     try:
    #         spin_thread.join(timeout=1.0)
    #     except Exception:
    #         pass
    #     print("[✅] ROS2 shutdown complete.")


if __name__ == "__main__":
    main()