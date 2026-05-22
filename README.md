# **Amgix Now** - Hybrid Search Engine

**Amgix Now** is a high-performance hybrid search engine from the Amgix family. Send it text, get back ranked results.

Part of the [Amgix](https://amgix.io) family.

## Benchmarks

<img src="https://amgix.io/images/amgix-now/amgix-now-bench-mix.png" align="right" width="350" hspace="10"/>

Visit our detailed [Amgix Now Benchmarks Series](https://amgix.io/blog/2026/05/21/amgix-now-bench-series/) (in context of Typesense, Meilisearch, and Elasticsearch).

<br clear="both"/>

## Get Started

```bash
docker run -d -p 8235:8235 -v <path/on/host>:/data amgixio/amgix-now:0
```

## The Amgix Family

| | Amgix Now | Amgix One | Amgix |
|---|:---:|:---:|:---:|
| Hybrid search | engine | system | system |
| Single container | ✓ | ✓ | |
| Async ingestion pipeline | | ✓ | ✓ |
| Dashboard & metrics | | ✓ | ✓ |
| PostgreSQL/MariaDB | | ✓ | ✓ |
| High availability | limited* | awkward** | ✓ |
| Modular scaling | | | ✓ |
| Model self-orchestration | | | ✓ |
| **Throughput** | high | medium | scales with cluster |
| **Latency** | lowest | medium | medium |
| **Operational complexity** | lowest | medium | highest |

- \* Amgix Now can be configured as a primary/fallback behind a proxy and a shared external Qdrant instance
- \*\* Multiple Amgix One instances can be deployed with external RabbitMQ and database instances, but it's awkward and better served by a full-scale Amgix cluster deployment

Same REST API and storage format across all three.

## Documentation

[docs.amgix.io](https://docs.amgix.io)

