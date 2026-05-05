#!/usr/bin/env bash

SERVER_PID=""
ACTIVE_PORT=""

kill_tree() {
    local pid=$1
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        kill_tree "$child"
    done
    kill "$pid" 2>/dev/null
}

cleanup() {
    [ -n "$SERVER_PID" ] && kill_tree "$SERVER_PID"
}

trap cleanup EXIT SIGHUP SIGTERM SIGINT

if [ ! -t 0 ]; then
  echo "Error: must be run in an interactive terminal" >&2
  exit 1
fi

MODE="dev"
[[ "$1" == "s" ]] && MODE="staging"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/venv/bin/python"
[ ! -x "$PY" ] && PY="python3"

port_open() {
    lsof -ti:"$1" >/dev/null 2>&1
}

find_new_port() {
    local before="$1" tries=0
    shift
    while [ $tries -lt 30 ]; do
        for p in "$@"; do
            if port_open "$p" && ! echo "$before" | grep -qw "$p"; then
                echo "$p"
                return 0
            fi
        done
        sleep 0.3
        tries=$((tries + 1))
    done
    return 1
}

kill_servers() {
    [ -n "$SERVER_PID" ] && kill_tree "$SERVER_PID" && SERVER_PID=""
    sleep 0.5
    local pids
    pids=$(lsof -ti:8000 2>/dev/null)
    [ -n "$pids" ] && echo "$pids" | xargs kill 2>/dev/null && sleep 0.3
}

start() {
    clear
    echo ""
    echo "  Starting Vehicle-Maintenance backend... [$MODE]"
    echo ""

    local busy_ports=""
    for p in 8000; do
        port_open "$p" && busy_ports="$busy_ports $p"
    done

    local BE_existing=false
    if echo "$busy_ports" | grep -qw "8000"; then
        BE_existing=true
    else
        cd "$ROOT_DIR"
        if [[ "$MODE" == "staging" ]]; then
            APP_ENV=staging "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level error >/dev/null 2>&1 </dev/null &
        else
            "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level error >/dev/null 2>&1 </dev/null &
        fi
        SERVER_PID=$!
    fi

    if [ "$BE_existing" = true ]; then
        echo "  Backend   → http://localhost:8000 (existing)"
        ACTIVE_PORT=8000
    else
        local actual_port
        actual_port=$(find_new_port "$busy_ports" 8000)
        if [ -n "$actual_port" ]; then
            ACTIVE_PORT=$actual_port
            echo "  Backend   → http://localhost:$actual_port"
        else
            echo "  Backend   ✗ didn't start"
        fi
    fi

    echo ""
    echo "  ✅ Vehicle-Maintenance backend ready [$MODE]"
    echo ""
    echo "  r restart · q quit"
    echo ""
}

start

while true; do
    [ ! -t 0 ] && cleanup && exit 0
    key=""
    read -rsn1 -t 3 key || true
    case "$key" in
        r|R)
            kill_servers
            start
            ;;
        q|Q) exit 0 ;;
        "")
            server_ok=true
            [ -n "$ACTIVE_PORT" ] && ! port_open "$ACTIVE_PORT" && server_ok=false
            if [ "$server_ok" = false ]; then
                echo ""
                echo "  ⚠ Server :$ACTIVE_PORT died"
                ACTIVE_PORT=""
                echo "  r restart · q quit"
                echo ""
            fi
            ;;
    esac
done
