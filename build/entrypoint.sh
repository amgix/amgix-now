#!/bin/bash
set -e

# Function to gracefully shutdown supervisord
shutdown() {
    echo "[entrypoint] Received shutdown signal, shutting down supervisord..."
    if [ -n "$SUPERVISORD_PID" ] && kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
        # Send SIGTERM to supervisord, which will gracefully shut down all children
        kill -TERM "$SUPERVISORD_PID"
        # Wait for supervisord to finish shutting down
        wait $SUPERVISORD_PID 2>/dev/null || true
    fi
    exit 0
}

# Trap SIGTERM and SIGINT to gracefully shutdown
trap shutdown SIGTERM SIGINT

# Tokio reads TOKIO_WORKER_THREADS; AMGIX_NOW_WORKERS is the image knob (see Dockerfile ENV).
if [ -n "${AMGIX_NOW_WORKERS:-}" ]; then
    export TOKIO_WORKER_THREADS="${AMGIX_NOW_WORKERS}"
fi

if [ "${AMGIX_DATABASE_URL}" != "${AMGIX_DEFAULT_DATABASE_URL}" ]; then
    echo "[entrypoint] AMGIX_DATABASE_URL != AMGIX_DEFAULT_DATABASE_URL — disabling embedded Qdrant supervisord program"
    sed -i '/^\[program:qdrant\]/,/^\[program:amgix-now\]/ s/^autostart=true$/autostart=false/' /etc/supervisor/conf.d/amgix-now.conf
else
    # Fix permissions for mounted volumes (especially when /data is mounted from host)
    mkdir -p /data/qdrant
    chmod 755 /data/qdrant
fi

# Increase the maximum number of open files limit to 65536
ulimit -n 65536

/usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf > >(
    while IFS= read -r line; do
        if [[ "$line" =~ ^\[.*\] ]]; then
            echo "$line"
        else
            echo "[supervisord] $line"
        fi
    done
) 2>&1 &
SUPERVISORD_PID=$!

wait $SUPERVISORD_PID
