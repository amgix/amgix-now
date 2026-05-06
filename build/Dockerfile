# amgix-now: single Rust binary + embedded Qdrant (no Python, RabbitMQ, or separate encoders)
#
# Build from the amgix-now crate root:
#   docker build -f build/Dockerfile .

FROM rust:1-bookworm AS rust-builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --locked --release

# -----------------------------------------------------------------------------

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Embedded Qdrant (same pattern as amgix-one)
ARG QDRANT_VERSION=v1.17.1
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then QDRANT_ARCH="x86_64"; else QDRANT_ARCH="$ARCH"; fi && \
    curl -fsSL "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-${QDRANT_ARCH}-unknown-linux-musl.tar.gz" -o /tmp/qdrant.tar.gz && \
    tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin && \
    rm /tmp/qdrant.tar.gz && \
    chmod +x /usr/local/bin/qdrant

COPY --from=rust-builder /app/target/release/amgix-now /app/amgix-now

COPY build/prefix-log-exec.sh build/wait-for-deps.sh build/entrypoint.sh /app/
COPY build/supervisord-now.conf /etc/supervisor/conf.d/amgix-now.conf
COPY build/config.yaml /app/config.yaml

RUN chmod +x /app/amgix-now /app/prefix-log-exec.sh /app/wait-for-deps.sh /app/entrypoint.sh \
    && mkdir -p /data/qdrant \
    && chmod 755 /data /data/qdrant

# Default: talk to embedded Qdrant gRPC (paths/ports: config.yaml + Qdrant defaults)
ENV AMGIX_DEFAULT_DATABASE_URL=qdrant://localhost:6334 \
    AMGIX_DATABASE_URL=qdrant://localhost:6334

ARG AMGIX_VERSION=${AMGIX_VERSION:-0.1.0-dev}
ENV AMGIX_VERSION=${AMGIX_VERSION}

ENV AMGIX_VARIANT='Amgix-Now'
ENV AMGIX_NOW_WORKERS=4

EXPOSE 8235

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl --silent --fail --max-time 3 http://127.0.0.1:8235/v1/health/ready || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
