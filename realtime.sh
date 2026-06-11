#!/usr/bin/env bash
# realtime.sh — Interactive wrapper for realtime_capture.py (movement + emotion + object trigger)
# Simplified: only main run arguments are interactive; rest under advanced controls.

# source /opt/ros/humble/setup.bash
# export ROS_DOMAIN_ID=0

set -euo pipefail

# ───────────────────────── Conda environment ─────────────────────────
NEED_DEACTIVATE=0
cleanup() {
  if [[ "${NEED_DEACTIVATE:-0}" -eq 1 ]]; then
    set +e
    conda deactivate >/dev/null 2>&1
    set -e
  fi
}
trap cleanup EXIT INT TERM

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "realtime_v2" ]]; then
      conda activate realtime_v2
      NEED_DEACTIVATE=1
    fi
  else
    echo "[ERROR] Conda environment not initialized. Run: conda init bash"
    exit 1
  fi
else
  echo "[ERROR] 'conda' not found in PATH."
  exit 1
fi

# ───────────────────────────── Script setup ───────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_PATH="${SCRIPT_PATH:-./scripts/realtime_capture.py}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] Script not found at $SCRIPT_PATH"
  exit 1
fi



# ────────────────────────────── Helpers ───────────────────────────────
ask() { local p="$1"; local d="${2:-}"; local r; read -r -p "$p [$d]: " r || true; echo "${r:-$d}"; }
ask_yn() { local p="$1"; local d="${2:-y}"; local a; while true; do a="$(ask "$p (y/n)" "$d")"; case "${a,,}" in y|yes) echo "y"; return;; n|no) echo "n"; return;; esac; done; }

# ─────────────────────────────── Banner ───────────────────────────────
cat <<'BANNER'
────────────────────────────────────────────────────────────────
   Real-Time Unified Pipeline — Interactive Launcher
   (captures all cams; selectively process Movement, Emotion, Object triggers)
────────────────────────────────────────────────────────────────
Controls during run:
  • SPACE  → Pause/Resume
  • q      → Close grid (pipeline continues)
  • g      → Reopen grid
  • ESC/Ctrl+C → Stop gracefully
────────────────────────────────────────────────────────────────
BANNER

# ─────────────────────────── Basic Controls ───────────────────────────
OUTPUT_DIR="$(ask "Output directory" "./run_$(date +%Y%m%d_%H%M%S)")"
DUR_SEC="$(ask "Active recording duration (sec)" "60")"
SAVE_EVERY="$(ask "Save raw frames? Enter N (0 = OFF)" "1")"
VIZ_LIVE="$(ask "Live preview window? (off/on)" "on")"

echo
NOTES_ENABLE="$(ask_yn "Enable Notes UI (chat panel)?" "y")"   # ← NEW

echo
echo "Which camera(s) to PROCESS for MOVEMENT?"
echo "  - Comma-separated labels (e.g., cam2 or cam1,cam3)"
echo "  - Enter 'all' to process all"
echo "  - Enter 'none' for capture-only"
PROC_MOV_INPUT="$(ask "Movement cams" "cam2")"

echo
echo "Which camera(s) to PROCESS for EMOTION (valence/arousal)?"
echo "  - Comma-separated labels (e.g., cam3 or cam1,cam2)"
echo "  - Enter 'none' to disable"
PROC_EMO_INPUT="$(ask "Emotion cams" "cam1")"

echo
echo "Which camera(s) to PROCESS for EYE TRACKING?"
echo "  - Comma-separated labels (e.g., cam1)"
echo "  - Enter 'none' to disable"
PROC_EYE_INPUT="$(ask "Eye tracking cams" "cam1")"

echo
echo "Which camera(s) to PROCESS for OBJECT TRIGGERS?"
echo "  - Comma-separated labels (e.g., cam2 or cam1,cam3)"
echo "  - Enter 'none' to disable"
PROC_OBJ_INPUT="$(ask "Object trigger cams" "cam2")"   # ← NEW

EVENT_CHECKER_ENABLE="$(ask_yn "Enable R0 expected-speed event checker?" "y")"
OBJ_TRIGGER_ENABLE="$(ask_yn "Enable object untouched trigger?" "y")"

# Heads-up for where logs will land
if [[ "${EVENT_CHECKER_ENABLE,,}" == "y" ]]; then
  echo "[INFO] Speed-trigger: ENABLED → per-cam logs at <cam_dir>/logs/speed_trigger.txt"
else
  echo "[INFO] Speed-trigger: DISABLED"
fi
if [[ "${OBJ_TRIGGER_ENABLE,,}" == "y" ]]; then
  echo "[INFO] Object-trigger: ENABLED → logs at <output_dir>/logs/object_trigger.txt"
else
  echo "[INFO] Object-trigger: DISABLED"
fi

ADVANCED="$(ask_yn "Show advanced controls?" "n")"

# ───────────────────────────── Defaults ───────────────────────────────
FILTERS="off"
FORCE_FLIP="flip"
STRIDE="1"
BKP_POLICY="drop-latest"
VIZ_SAVE_EVERY="3"
CSV_FLUSH="30"
LOG_FLUSH_SEC="5"
EMO_HISTORY="240"
EMO_STRIDE="1"
EMO_CSV_FLUSH="30"
AUDIO_ENABLE="n"
AUDIO_OUT="$OUTPUT_DIR/audio"
AUDIO_DUR="$DUR_SEC"
RATE="44100"

# ─────────────────────────── Advanced Controls ────────────────────────
if [[ "$ADVANCED" == "y" ]]; then
  echo "── Advanced Controls ───────────────────────────"
  FILTERS="$(ask "Depth filters (on/off)" "off")"
  FORCE_FLIP="$(ask "Handedness flip baseline (flip/same)" "flip")"
  STRIDE="$(ask "Movement processing stride" "1")"
  BKP_POLICY="$(ask "Backpressure policy (drop-latest/block)" "drop-latest")"
  VIZ_SAVE_EVERY="$(ask "Save annotated previews every N frames (0=off)" "3")"
  CSV_FLUSH="$(ask "Movement CSV flush interval (frames)" "30")"
  LOG_FLUSH_SEC="$(ask "Logger flush interval (sec)" "5")"
  EMO_HISTORY="$(ask "Emotion plot history (frames)" "240")"
  EMO_STRIDE="$(ask "Emotion stride" "1")"
  EMO_CSV_FLUSH="$(ask "Emotion CSV flush interval" "30")"
  AUDIO_ENABLE="$(ask_yn "Record microphones?" "n")"
  if [[ "$AUDIO_ENABLE" == "y" ]]; then
    AUDIO_OUT="$(ask "Audio output dir" "$AUDIO_OUT")"
    AUDIO_DUR="$(ask "Audio duration (sec)" "$AUDIO_DUR")"
    RATE="$(ask "Audio sample rate (44100/48000)" "44100")"
  fi
  echo "────────────────────────────────────────────────"
fi

# ─────────────────────────── Parse Cam Lists ──────────────────────────
declare -a CMD
CMD+=("$PYTHON_BIN" "$SCRIPT_PATH" "--output-dir" "$OUTPUT_DIR" "--duration-sec" "$DUR_SEC" "--save-every" "$SAVE_EVERY")
CMD+=("--viz-live" "$VIZ_LIVE" "--filters" "$FILTERS" "--force-flip" "$FORCE_FLIP")
CMD+=("--stride" "$STRIDE" "--backpressure" "$BKP_POLICY" "--viz-save-every" "$VIZ_SAVE_EVERY")
CMD+=("--csv-flush" "$CSV_FLUSH" "--log-flush-sec" "$LOG_FLUSH_SEC")
CMD+=("--emo-history" "$EMO_HISTORY" "--emo-stride" "$EMO_STRIDE" "--emo-csv-flush" "$EMO_CSV_FLUSH")

# Eye tracking cams
IFS=',' read -r -a EYE_ARR <<<"${PROC_EYE_INPUT// /}"
if [[ "${PROC_EYE_INPUT,,}" != "none" && "${PROC_EYE_INPUT,,}" != "" ]]; then
  CMD+=("--process-eye-cams" "${EYE_ARR[@]}")
fi

# Notes toggle  ← NEW
if [[ "${NOTES_ENABLE,,}" == "y" ]]; then
  CMD+=("--notes" "on")
else
  CMD+=("--notes" "off")
fi

# Movement cams
IFS=',' read -r -a MOV_ARR <<<"${PROC_MOV_INPUT// /}"
if [[ "${PROC_MOV_INPUT,,}" != "none" && "${PROC_MOV_INPUT,,}" != "all" ]]; then
  CMD+=("--process-mov-cams" "${MOV_ARR[@]}")
elif [[ "${PROC_MOV_INPUT,,}" == "none" ]]; then
  CMD+=("--process-mov-cams")
fi

# Emotion cams
IFS=',' read -r -a EMO_ARR <<<"${PROC_EMO_INPUT// /}"
if [[ "${PROC_EMO_INPUT,,}" != "none" && "${PROC_EMO_INPUT,,}" != "" ]]; then
  CMD+=("--process-emo-cams" "${EMO_ARR[@]}")
fi

# Object trigger cams  ← NEW
IFS=',' read -r -a OBJ_ARR <<<"${PROC_OBJ_INPUT// /}"
if [[ "${PROC_OBJ_INPUT,,}" != "none" && "${PROC_OBJ_INPUT,,}" != "" ]]; then
  CMD+=("--process-obj-cams" "${OBJ_ARR[@]}")
fi

# Audio controls
if [[ "$AUDIO_ENABLE" == "y" ]]; then
  CMD+=("--audio-out" "$AUDIO_OUT" "--audio-duration-sec" "$AUDIO_DUR" "--rate" "$RATE")
else
  CMD+=("--audio-duration-sec" "0")
fi

# Event checker toggles
if [[ "${EVENT_CHECKER_ENABLE,,}" != "y" ]]; then
  CMD+=("--no-event-checker")
fi
if [[ "${OBJ_TRIGGER_ENABLE,,}" != "y" ]]; then
  CMD+=("--no-object-trigger")
fi

# ─────────────────────────── Summary ───────────────────────────
echo
echo "──────────────── RUN SUMMARY ────────────────"
echo "Output dir        : $OUTPUT_DIR"
echo "Duration (sec)    : $DUR_SEC"
echo "Save-every (raw)  : $SAVE_EVERY"
echo "Live preview      : $VIZ_LIVE"
echo "Notes UI          : $([[ "${NOTES_ENABLE,,}" == "y" ]] && echo "ENABLED" || echo "DISABLED")"
echo "Movement cams     : $PROC_MOV_INPUT"
echo "Emotion cams      : $PROC_EMO_INPUT"
echo "Object cams       : $PROC_OBJ_INPUT"
echo "Eye tracking cams : $PROC_EYE_INPUT"
echo "Speed-trigger     : $([[ "${EVENT_CHECKER_ENABLE,,}" == "y" ]] && echo "ENABLED" || echo "DISABLED")"
echo "Object-trigger    : $([[ "${OBJ_TRIGGER_ENABLE,,}" == "y" ]] && echo "ENABLED" || echo "DISABLED")"
if [[ "$ADVANCED" == "y" ]]; then
  echo "Depth filters     : $FILTERS"
  echo "Flip baseline     : $FORCE_FLIP"
  echo "Stride            : $STRIDE"
  echo "Backpressure      : $BKP_POLICY"
  echo "CSV flush         : $CSV_FLUSH"
  echo "Log flush (sec)   : $LOG_FLUSH_SEC"
  echo "Emotion history   : $EMO_HISTORY"
  echo "Emotion stride    : $EMO_STRIDE"
  echo "Emotion CSV flush : $EMO_CSV_FLUSH"
  echo "Audio enabled     : $AUDIO_ENABLE"
fi
echo "─────────────────────────────────────────────"

read -r -p "Proceed? (y/n) [y]: " CONFIRM
CONFIRM="${CONFIRM:-y}"
[[ "${CONFIRM,,}" != "y" ]] && { echo "Aborted."; exit 0; }

mkdir -p "$OUTPUT_DIR"
[[ "$AUDIO_ENABLE" == "y" ]] && mkdir -p "$AUDIO_OUT"

# ─────────────────────────── Execute ───────────────────────────
set +e
"${CMD[@]}"
EXIT_CODE=$?
set -e
exit "$EXIT_CODE"
