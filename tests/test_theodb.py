"""Tests for the TheoDB client.

TheoDB (https://github.com/usetheoai/theo-db) is an own-code, PostgreSQL-compatible
database. Its vector pillar is a pgrx extension that ships a `vector` type wire-compatible
with pgvector, plus its own `theodb_hnsw` access method (aliased as `hnsw`).

The tests split in two groups:

  * config tests — pure, no database. They cover the one thing this client does that the
    other Postgres clients do not: it REFUSES a build parameter it cannot honour, instead
    of accepting it and quietly building something else.
  * live tests — skipped unless a TheoDB instance is reachable. They exercise the hot path
    the benchmark actually times: binary COPY, index build, and a k-NN scan.

Live tests require:

    docker run -d --name theodb-b035 \
      -e POSTGRES_PASSWORD=theo -e POSTGRES_DB=theo -p 55435:5432 theodb:b034

Usage:
    pytest tests/test_theodb.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType

THEODB_HOST = os.environ.get("THEODB_HOST", "127.0.0.1")
THEODB_PORT = int(os.environ.get("THEODB_PORT", "55435"))
THEODB_USER = os.environ.get("THEODB_USER", "postgres")
THEODB_PASSWORD = os.environ.get("THEODB_PASSWORD", "theo")
THEODB_DBNAME = os.environ.get("THEODB_DBNAME", "theo")

DIM = 64
N_ROWS = 5000
K = 10


def _theodb_reachable() -> bool:
    """True when a TheoDB instance answers on the configured endpoint."""
    try:
        import psycopg

        with psycopg.connect(
            host=THEODB_HOST,
            port=THEODB_PORT,
            user=THEODB_USER,
            password=THEODB_PASSWORD,
            dbname=THEODB_DBNAME,
            connect_timeout=3,
        ) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


needs_theodb = pytest.mark.skipif(
    not _theodb_reachable(),
    reason=f"no TheoDB reachable at {THEODB_HOST}:{THEODB_PORT}",
)


# --------------------------------------------------------------------------------------
# T1.1 — registry
# --------------------------------------------------------------------------------------


def test_theodb_is_registered_and_resolves():
    assert DB.TheoDB.value == "TheoDB"
    assert DB.TheoDB.init_cls.__name__ == "TheoDB"
    assert DB.TheoDB.config_cls.__name__ == "TheoDBConfig"
    assert DB.TheoDB.case_config_cls(IndexType.HNSW).__name__ == "TheoDBHNSWConfig"


def test_theodb_cli_command_is_exposed():
    from vectordb_bench.cli.vectordbbench import cli

    assert "theodbhnsw" in cli.commands  # upstream convention: no dash (pgvectorhnsw)


# --------------------------------------------------------------------------------------
# T1.3 — a build parameter we cannot honour is refused, loudly and early
# --------------------------------------------------------------------------------------


def test_unsupported_m_is_refused_at_config_time():
    from vectordb_bench.backend.clients.theodb.config import (
        TheoDBHNSWConfig,
        UnsupportedBuildParameterError,
    )

    with pytest.raises(UnsupportedBuildParameterError) as exc:
        TheoDBHNSWConfig(metric_type=MetricType.L2, m=32)

    message = str(exc.value)
    assert all(token in message for token in ("m", "32", "16", "B-036")), message


def test_unsupported_ef_construction_is_refused_at_config_time():
    from vectordb_bench.backend.clients.theodb.config import (
        TheoDBHNSWConfig,
        UnsupportedBuildParameterError,
    )

    with pytest.raises(UnsupportedBuildParameterError) as exc:
        TheoDBHNSWConfig(metric_type=MetricType.L2, ef_construction=200)

    assert "ef_construction" in str(exc.value)


def test_honored_values_are_accepted_and_never_forwarded():
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    config = TheoDBHNSWConfig(metric_type=MetricType.L2, m=16, ef_construction=64)
    assert config.index_param()["options"] == {}


def test_unset_build_params_are_accepted():
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    assert TheoDBHNSWConfig(metric_type=MetricType.L2).index_param()["options"] == {}


def test_create_index_statement_carries_no_with_clause():
    """TheoDB's hnsw AM rejects `m` and `ef_construction`; the SQL must not emit WITH."""
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    statement = TheoDB.render_create_index(
        table_name="t",
        index_name="t_idx",
        vector_field="embedding",
        case_config=TheoDBHNSWConfig(metric_type=MetricType.L2),
    )
    assert "WITH (" not in statement
    assert "USING hnsw" in statement
    assert "vector_l2_ops" in statement


# --------------------------------------------------------------------------------------
# T1.4 — metric mapping
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "opclass", "operator"),
    [
        (MetricType.L2, "vector_l2_ops", "<->"),
        (MetricType.COSINE, "vector_cosine_ops", "<=>"),
        (MetricType.IP, "vector_ip_ops", "<#>"),
    ],
)
def test_metric_maps_to_opclass_and_operator(metric, opclass, operator):
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    config = TheoDBHNSWConfig(metric_type=metric)
    assert config.index_param()["metric"] == opclass
    assert config.search_param()["metric_fun_op"] == operator


def test_unsupported_metric_is_refused():
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    with pytest.raises(ValueError, match="HAMMING"):
        TheoDBHNSWConfig(metric_type=MetricType.HAMMING).index_param()


def test_cosine_is_native_so_the_dataset_is_never_normalized():
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB

    assert TheoDB.need_normalize_cosine(TheoDB) is False


# --------------------------------------------------------------------------------------
# Concurrency contract — the runner deep-copies the client across processes
# --------------------------------------------------------------------------------------


def test_client_declares_itself_not_thread_safe():
    """A psycopg connection cannot be shared across threads; the runner must clone."""
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB

    assert TheoDB.thread_safe is False


def test_only_non_filter_is_declared_supported():
    from vectordb_bench.backend.filter import FilterOp
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB

    assert TheoDB.supported_filter_types == [FilterOp.NonFilter]


# --------------------------------------------------------------------------------------
# T1.2 — live path
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> tuple[np.ndarray, np.ndarray, set[int]]:
    rng = np.random.default_rng(20260812)
    vectors = rng.random((N_ROWS, DIM), dtype=np.float32)
    query = rng.random(DIM, dtype=np.float32)
    truth = set(np.argsort(((vectors - query) ** 2).sum(axis=1))[:K].tolist())
    return vectors, query, truth


def _live_client(drop_old: bool = True):
    from vectordb_bench.backend.clients.theodb.config import TheoDBConfig, TheoDBHNSWConfig
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from pydantic import SecretStr

    db_config = TheoDBConfig(
        user_name=SecretStr(THEODB_USER),
        password=SecretStr(THEODB_PASSWORD),
        host=THEODB_HOST,
        port=THEODB_PORT,
        db_name=THEODB_DBNAME,
    )
    return TheoDB(
        dim=DIM,
        db_config=db_config.to_dict(),
        db_case_config=TheoDBHNSWConfig(metric_type=MetricType.L2),
        collection_name="vdbb_theodb_live",
        drop_old=drop_old,
    )


@needs_theodb
def test_live_client_loads_indexes_and_searches(corpus):
    vectors, query, truth = corpus
    client = _live_client()

    with client.init():
        count, error = client.insert_embeddings(vectors.tolist(), list(range(N_ROWS)))
        assert error is None, error
        assert count == N_ROWS

        client.optimize(data_size=N_ROWS)

        got = client.search_embedding(query.tolist(), k=K)
        assert len(got) == K
        # recall, not row count: a client that returns 10 arbitrary ids is not searching
        assert len(set(got) & truth) >= 5, f"recall too low: {len(set(got) & truth)}/{K}"


@needs_theodb
def test_live_search_uses_the_index(corpus):
    """A sequential scan would still return correct rows — and measure nothing useful."""
    vectors, query, _ = corpus
    client = _live_client(drop_old=False)

    with client.init():
        plan = client.explain_search(query.tolist(), k=K)

    assert "Index Scan" in plan, plan


@needs_theodb
def test_live_session_applies_ef_search(corpus):
    """The session GUC must actually reach the backend under its native name."""
    from vectordb_bench.backend.clients.theodb.config import TheoDBConfig, TheoDBHNSWConfig
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from pydantic import SecretStr

    db_config = TheoDBConfig(
        user_name=SecretStr(THEODB_USER),
        password=SecretStr(THEODB_PASSWORD),
        host=THEODB_HOST,
        port=THEODB_PORT,
        db_name=THEODB_DBNAME,
    )
    client = TheoDB(
        dim=DIM,
        db_config=db_config.to_dict(),
        db_case_config=TheoDBHNSWConfig(metric_type=MetricType.L2, ef_search=321),
        collection_name="vdbb_theodb_live",
        drop_old=False,
    )
    with client.init():
        assert client.current_setting("theodb_hnsw.ef_search") == "321"


@needs_theodb
def test_client_is_deepcopyable_when_idle(corpus):
    """The concurrent runner deep-copies the client; a live connection breaks pickling.

    Upstream issue #756 hit exactly this on the pgvector client.
    """
    import copy

    client = _live_client(drop_old=False)
    clone = copy.deepcopy(client)
    assert clone.table_name == client.table_name
