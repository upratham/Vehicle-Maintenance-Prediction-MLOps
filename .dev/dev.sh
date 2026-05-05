#!/usr/bin/env bash

BE_PID=""
FE_PID=""
ACTIVE_BE_PORT=""
ACTIVE_FE_PORT=""

kill_tree() {
    local pid=$1
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        kill_tree "$child"
    done
    kill "$pid" 2>/dev/null
}

cleanup() {
    [ -n "$BE_PID" ] && kill_tree "$BE_PID"
    [ -n "$FE_PID" ] && kill_tree "$FE_PID"
}

trap cleanup EXIT SIGHUP SIGTERM SIGINT

if [ ! -t 0 ]; then
  echo "Error: must be run in an interactive terminal" >&2
  exit 1
fi

MODE="dev"
[[ "$1" == "s" ]] && MODE="staging"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FE_DIR="$ROOT_DIR/frontend"
PY="$ROOT_DIR/venv/bin/python"
[ ! -x "$PY" ] && PY="python3"

# first-run setup: venv + deps + train models if missing
setup() {
    cd "$ROOT_DIR"

    if [ ! -d venv ]; then
        echo "  creating venv..."
        python3 -m venv venv
    fi
    PY="$ROOT_DIR/venv/bin/python"

    if ! "$PY" -c "import fastapi" 2>/dev/null; then
        echo "  installing python deps..."
        "$PY" -m pip install -q -r requirements.txt
    fi

    if [ ! -d "$FE_DIR/node_modules" ]; then
        echo "  installing frontend deps..."
        (cd "$FE_DIR" && npm install --silent)
    fi

    local need_train=false
    for p in vehicle_maintenance cars_hyundai engine_data; do
        if ! ls artifact/$p/*/model_trainer/trained_model/model.pkl >/dev/null 2>&1; then
            need_train=true
            break
        fi
    done
    if [ "$need_train" = true ]; then
        echo "  training models (one-time, ~5-15 min)..."
        "$PY" demo.py
    fi

    if [ ! -d "$FE_DIR/dist" ]; then
        echo "  building frontend..."
        (cd "$FE_DIR" && npm run build --silent)
    fi
}

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
    [ -n "$BE_PID" ] && kill_tree "$BE_PID" && BE_PID=""
    [ -n "$FE_PID" ] && kill_tree "$FE_PID" && FE_PID=""
    sleep 0.5
    local pids
    pids=$(lsof -ti:8000,3000,3001,3002,3003,3004,3005,5173 2>/dev/null)
    [ -n "$pids" ] && echo "$pids" | xargs kill 2>/dev/null && sleep 0.3
}

start() {
    clear
    echo ""
    echo "  Starting Vehicle-Maintenance... [$MODE]"
    echo ""

    local busy_ports=""
    for p in 8000 3000 3001 3002 3003 3004 3005 5173; do
        port_open "$p" && busy_ports="$busy_ports $p"
    done

    # Backend
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
        BE_PID=$!
    fi

    # Frontend
    local FE_existing=false
    if echo "$busy_ports" | grep -qw "3000"; then
        FE_existing=true
    else
        cd "$FE_DIR"
        if [[ "$MODE" == "staging" ]]; then
            VITE_APP_ENV=staging npx vite --port 3000 >/dev/null 2>&1 </dev/null &
        else
            npx vite --port 3000 >/dev/null 2>&1 </dev/null &
        fi
        FE_PID=$!
    fi

    if [ "$BE_existing" = true ]; then
        echo "  Backend   → http://localhost:8000 (existing)"
        ACTIVE_BE_PORT=8000
    else
        local be_port
        be_port=$(find_new_port "$busy_ports" 8000)
        if [ -n "$be_port" ]; then
            ACTIVE_BE_PORT=$be_port
            echo "  Backend   → http://localhost:$be_port"
        else
            echo "  Backend   ✗ didn't start"
        fi
    fi

    if [ "$FE_existing" = true ]; then
        echo "  Frontend  → http://localhost:3000 (existing)"
        ACTIVE_FE_PORT=3000
    else
        local fe_port
        fe_port=$(find_new_port "$busy_ports" 3000 3001 3002 3003 3004 3005 5173)
        if [ -n "$fe_port" ]; then
            ACTIVE_FE_PORT=$fe_port
            echo "  Frontend  → http://localhost:$fe_port"
        else
            echo "  Frontend  ✗ didn't start"
        fi
    fi

    echo ""
    echo "  ✅ Vehicle-Maintenance ready [$MODE]"
    echo ""
    echo "  r restart · q quit"
    echo ""
}

setup
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
            be_ok=true; fe_ok=true
            [ -n "$ACTIVE_BE_PORT" ] && ! port_open "$ACTIVE_BE_PORT" && be_ok=false
            [ -n "$ACTIVE_FE_PORT" ] && ! port_open "$ACTIVE_FE_PORT" && fe_ok=false
            if [ "$be_ok" = false ] || [ "$fe_ok" = false ]; then
                echo ""
                if [ "$be_ok" = false ]; then
                    echo "  ⚠ Backend :$ACTIVE_BE_PORT died"
                    ACTIVE_BE_PORT=""
                fi
                if [ "$fe_ok" = false ]; then
                    echo "  ⚠ Frontend :$ACTIVE_FE_PORT died"
                    ACTIVE_FE_PORT=""
                fi
                echo "  r restart · q quit"
                echo ""
            fi
            ;;
    esac
done
