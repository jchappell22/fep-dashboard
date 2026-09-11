#!/usr/bin/env bash
# Launch the FEP dashboard.
#
# Safe to run while campaigns are in flight: the dashboard only ever READS
# the campaign directories. Campaign drivers are detached (their own process
# group), so starting, stopping, or crashing this UI cannot affect a running
# calculation.
#
# Run it in its own terminal; streamlit stays in the foreground. Ctrl-C to stop.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${FEPDASH_PORT:-8578}"

# Bound to loopback deliberately: this box is shared and there is NO auth in
# front of the dashboard -- and unlike a read-only status page, this one can
# launch GPU jobs. Browse from the box's own desktop, or forward the port:
#
#     ssh -L 8578:localhost:8578 <user>@conifer
#
# Change to 0.0.0.0 only if you have decided the LAN should be able to start
# jobs on these cards.
ADDRESS="${FEPDASH_ADDRESS:-127.0.0.1}"

export FEPDASH_CONFIG="${FEPDASH_CONFIG:-$REPO/config.toml}"

if [ ! -f "$FEPDASH_CONFIG" ]; then
    echo "WARNING: no config at $FEPDASH_CONFIG -- falling back to repo-relative"
    echo "         defaults, which write campaigns into $REPO/runs."
    echo "         Copy config.example.toml to config.toml and set runs_root."
    echo
fi

# ---- find streamlit ------------------------------------------------------
# Resolved in this order so the script works whether or not you remembered to
# activate anything: an already-active env, then the repo's own conda env,
# then a local .venv.
#
# Note this is about the DASHBOARD's env only. The engines need no activation
# at all -- `openfe` and `tmd-submit` are launchers in /usr/local/bin that
# activate their own /opt envs internally.
ENV_NAME="${FEPDASH_ENV:-fep-dash}"

find_streamlit() {
    command -v streamlit 2>/dev/null && return 0
    for root in "${CONDA_ROOT:-}" "$HOME/miniconda3" "$HOME/mambaforge" \
                "$HOME/miniforge3" /opt/conda; do
        [ -n "$root" ] && [ -x "$root/envs/$ENV_NAME/bin/streamlit" ] \
            && echo "$root/envs/$ENV_NAME/bin/streamlit" && return 0
    done
    [ -x "$REPO/.venv/bin/streamlit" ] && echo "$REPO/.venv/bin/streamlit" && return 0
    return 1
}

STREAMLIT="$(find_streamlit)" || {
    echo "ERROR: streamlit not found (looked on PATH, in conda env '$ENV_NAME',"
    echo "       and in $REPO/.venv)."
    echo
    echo "  mamba env create -f environment.yml     # creates '$ENV_NAME'"
    echo "  mamba activate $ENV_NAME"
    echo
    echo "Or set FEPDASH_ENV=<name> if your env is called something else."
    exit 1
}

echo "=============================================================="
echo "FEP dashboard"
echo "  config : $FEPDASH_CONFIG"
echo "  python : $STREAMLIT"
echo "  URL    : http://localhost:$PORT"
echo
echo "  Launch  -- start a campaign (shows the exact command first)"
echo "  Runs    -- live leg progress, logs, kill / resume"
echo "  Results -- ranked ligands, edge table, network consistency"
echo
echo "Engine availability:"
for bin in openfe tmd-submit; do
    if command -v "$bin" >/dev/null 2>&1; then
        echo "  $bin -> $(command -v "$bin")"
    else
        echo "  $bin -> NOT FOUND (campaigns using it will fail at launch)"
    fi
done
echo "=============================================================="
echo

cd "$REPO"
# --server.headless suppresses the first-run email prompt and stops streamlit
# trying to open a browser on a headless box.
exec "$STREAMLIT" run src/fepdash/app.py \
    --server.port "$PORT" \
    --server.headless true \
    --server.address "$ADDRESS"
