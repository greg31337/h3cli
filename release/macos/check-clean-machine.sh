#!/usr/bin/env bash
# Copy this script beside the downloaded CLI on the clean Mac. No Python/Xcode.
set -euo pipefail
h3_clean_binary=''
h3_clean_models=''
h3_clean_model=''
h3_clean_output=''
h3_clean_port=31987
h3_clean_pid=''
while (($#)); do
  case "$1" in
    --binary|--models-path|--model|--output|--port)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --binary) h3_clean_binary="$2" ;;
        --models-path) h3_clean_models="$2" ;;
        --model) h3_clean_model="$2" ;;
        --output) h3_clean_output="$2" ;;
        --port) h3_clean_port="$2" ;;
      esac
      shift 2 ;;
    --help|-h)
      echo 'Usage: check-clean-machine.sh --binary /path/to/h3cli-macos-arm64 --models-path /models --output /fresh/output [--model /custom/main] [--port 31987]'
      echo 'Runs offline help, a 256x256/56-frame/two-step render and a server job. Preserves quarantine; does not claim fresh-trust qualification.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$(/usr/bin/uname -s)" == Darwin ]] || { echo 'Requires macOS' >&2; exit 2; }
[[ "$h3_clean_binary" == /* && -f "$h3_clean_binary" ]] || { echo 'Supply an absolute executable path' >&2; exit 2; }
[[ "$h3_clean_models" == /* && -d "$h3_clean_models" ]] || { echo 'Supply an absolute installed models path' >&2; exit 2; }
[[ "$h3_clean_output" == /* && ! -e "$h3_clean_output" ]] || { echo 'Supply a fresh absolute output directory' >&2; exit 2; }
[[ "$h3_clean_port" =~ ^[0-9]{1,5}$ ]] && ((10#$h3_clean_port > 1024 && 10#$h3_clean_port < 65536)) || { echo 'Invalid port' >&2; exit 2; }
if /usr/bin/nc -z 127.0.0.1 "$h3_clean_port"; then echo 'Port already in use; choose --port' >&2; exit 2; fi
mkdir -p "$h3_clean_output"
exec > >(tee "$h3_clean_output/check.log") 2>&1
/usr/bin/sw_vers
/usr/bin/uname -m
/usr/bin/shasum -a 256 "$h3_clean_binary"
/usr/bin/xattr -l "$h3_clean_binary" || true
chmod +x "$h3_clean_binary"
h3_clean_args=(--offline --models-path "$h3_clean_models")
if [[ -n "$h3_clean_model" ]]; then h3_clean_args+=(-d "$h3_clean_model"); fi
"$h3_clean_binary" --help
"$h3_clean_binary" "${h3_clean_args[@]}" \
  -p 'A small wooden boat on a quiet pond.' \
  --width 256 --height 256 --frames 56 --steps 2 -o "$h3_clean_output/smoke.mp4"
cleanup() {
  if [[ -n "$h3_clean_pid" ]]; then
    kill "$h3_clean_pid" 2>/dev/null || true
    wait "$h3_clean_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$h3_clean_binary" "${h3_clean_args[@]}" --server --server-host 127.0.0.1 \
  --server-port "$h3_clean_port" --server-state-dir "$h3_clean_output/server-state" \
  > "$h3_clean_output/server.log" 2>&1 &
h3_clean_pid=$!
h3_clean_url="http://127.0.0.1:$h3_clean_port"
h3_clean_ready=false
for ((h3_clean_attempt=0; h3_clean_attempt<60; h3_clean_attempt++)); do
  kill -0 "$h3_clean_pid" || { echo 'Server exited; see server.log' >&2; exit 1; }
  if /usr/bin/curl --max-time 2 -fsS "$h3_clean_url/health" > "$h3_clean_output/health.json" 2>/dev/null; then
    h3_clean_ready=true
    break
  fi
  sleep 1
done
[[ "$h3_clean_ready" == true ]] || { echo 'Server startup timed out' >&2; exit 1; }
/usr/bin/curl --max-time 30 -fsS "$h3_clean_url/v1/videos" -H 'Content-Type: application/json' \
  -d '{"prompt":"A small wooden boat on a quiet pond.","h3cli":"--width 256 --height 256 --frames 56 --steps 2"}' \
  > "$h3_clean_output/job.json"
h3_clean_job="$(/usr/bin/plutil -extract id raw -o - "$h3_clean_output/job.json")"
[[ "$h3_clean_job" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid job ID' >&2; exit 1; }
h3_clean_done=false
for ((h3_clean_attempt=0; h3_clean_attempt<900; h3_clean_attempt++)); do
  kill -0 "$h3_clean_pid" || { echo 'Server exited' >&2; exit 1; }
  /usr/bin/curl --max-time 30 -fsS "$h3_clean_url/v1/videos/$h3_clean_job" > "$h3_clean_output/status.json"
  h3_clean_status="$(/usr/bin/plutil -extract status raw -o - "$h3_clean_output/status.json")"
  case "$h3_clean_status" in
    completed) h3_clean_done=true; break ;;
    queued|in_progress) sleep 2 ;;
    *) echo "Job ended with status: $h3_clean_status" >&2; exit 1 ;;
  esac
done
[[ "$h3_clean_done" == true ]] || { echo 'Server job timed out' >&2; exit 1; }
/usr/bin/curl --max-time 60 -fsS "$h3_clean_url/v1/videos/$h3_clean_job/content" -o "$h3_clean_output/server.mp4"
[[ -s "$h3_clean_output/smoke.mp4" && -s "$h3_clean_output/server.mp4" ]]
echo 'Runtime smoke checks passed. Record the clean-machine/trust conditions separately. Video inspection is optional.'
