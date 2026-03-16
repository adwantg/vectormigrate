from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    command: list[str]


SERVICES = [
    ServiceSpec(
        name="vectormigrate-opensearch-live",
        command=[
            "docker",
            "run",
            "-d",
            "--name",
            "vectormigrate-opensearch-live",
            "-p",
            "19200:9200",
            "-e",
            "discovery.type=single-node",
            "-e",
            "bootstrap.memory_lock=true",
            "-e",
            "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m",
            "-e",
            "DISABLE_SECURITY_PLUGIN=true",
            "-e",
            "DISABLE_INSTALL_DEMO_CONFIG=true",
            "opensearchproject/opensearch:3.5.0",
        ],
    ),
    ServiceSpec(
        name="vectormigrate-weaviate-live",
        command=[
            "docker",
            "run",
            "-d",
            "--name",
            "vectormigrate-weaviate-live",
            "-p",
            "18080:8080",
            "-e",
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
            "-e",
            "PERSISTENCE_DATA_PATH=/var/lib/weaviate",
            "-e",
            "QUERY_DEFAULTS_LIMIT=25",
            "-e",
            "DEFAULT_VECTORIZER_MODULE=none",
            "-e",
            "CLUSTER_HOSTNAME=node1",
            "semitechnologies/weaviate:1.36.5",
        ],
    ),
    ServiceSpec(
        name="vectormigrate-qdrant-live",
        command=[
            "docker",
            "run",
            "-d",
            "--name",
            "vectormigrate-qdrant-live",
            "-p",
            "16333:6333",
            "qdrant/qdrant:v1.17.0",
        ],
    ),
    ServiceSpec(
        name="vectormigrate-pgvector-live",
        command=[
            "docker",
            "run",
            "-d",
            "--name",
            "vectormigrate-pgvector-live",
            "-p",
            "15432:5432",
            "-e",
            "POSTGRES_DB=vectormigrate",
            "-e",
            "POSTGRES_USER=vectormigrate",
            "-e",
            "POSTGRES_PASSWORD=vectormigrate",
            "pgvector/pgvector:pg17",
        ],
    ),
]


def _run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _cleanup_services() -> None:
    for service in SERVICES:
        subprocess.run(
            ["docker", "rm", "-f", service.name],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )


def _wait_http(url: str, timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _wait_pg(
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    timeout_seconds: int = 180,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with psycopg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=dbname,
                connect_timeout=5,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                return
        except psycopg.Error as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for PostgreSQL: {last_error}")


def main() -> int:
    result_summary: dict[str, str] = {}
    env = os.environ.copy()
    env.update(
        {
            "VECTORMIGRATE_OPENSEARCH_URL": "http://127.0.0.1:19200",
            "VECTORMIGRATE_WEAVIATE_URL": "http://127.0.0.1:18080",
            "VECTORMIGRATE_QDRANT_URL": "http://127.0.0.1:16333",
            "VECTORMIGRATE_PGVECTOR_DSN": "postgresql://vectormigrate:vectormigrate@127.0.0.1:15432/vectormigrate",
            "VECTORMIGRATE_RUN_LIVE_BACKENDS": "1",
        }
    )

    _cleanup_services()

    try:
        for service in SERVICES:
            started = _run(service.command)
            result_summary[service.name] = started.stdout.strip()

        _wait_http("http://127.0.0.1:19200/_cluster/health")
        _wait_http("http://127.0.0.1:18080/v1/.well-known/ready")
        _wait_http("http://127.0.0.1:16333/readyz")
        _wait_pg("127.0.0.1", 15432, "vectormigrate", "vectormigrate", "vectormigrate")

        pytest_run = _run([sys.executable, "-m", "pytest"], env=env)
        result_summary["pytest"] = pytest_run.stdout.strip()
        print(json.dumps(result_summary, indent=2))
        return 0
    finally:
        for service in SERVICES:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "20", service.name],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )
            combined = (logs.stdout + logs.stderr).strip()
            if combined:
                print(f"===== {service.name} logs =====")
                print(combined)
        _cleanup_services()


if __name__ == "__main__":
    raise SystemExit(main())
