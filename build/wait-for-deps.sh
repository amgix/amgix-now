#!/bin/bash
# Wait for embedded Qdrant when AMGIX_DATABASE_URL matches baked-in default.
set -e

echo "Waiting for dependencies to be ready..."

if [ "${AMGIX_DATABASE_URL}" = "${AMGIX_DEFAULT_DATABASE_URL}" ]; then
    echo -n "Checking Qdrant..."
    for i in {1..30}; do
        if timeout 1 bash -c "echo > /dev/tcp/localhost/6334" 2>/dev/null; then
            echo " ready!"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo " FAILED - timeout after 30s"
            exit 1
        fi
        echo -n "."
        sleep 1
    done
else
    echo "Skipping Qdrant wait (external AMGIX_DATABASE_URL)"
fi

echo "All dependencies ready!"
exec "$@"
