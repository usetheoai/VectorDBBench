"""Tests for TheoDB's full-text (BM25) path.

TheoDB exposes BM25 through two SQL functions rather than an index access method:
`bm25_build(index_id, table, id_col, text_col)` builds the lexical index over a table
that is already loaded, and `bm25_search(index_id, query, k)` returns `(id, score)`
ordered by score descending.

That shape maps onto the benchmark's document contract exactly: `insert_documents` does
the COPY, `optimize` builds, `search_documents` searches. The build is a full rebuild, so
it belongs in `optimize` and nowhere else — building per batch would be quadratic and
would make the harness's build-time metric meaningless.

Measured limits this client does NOT paper over: no stemming (`jumping` does not match
`jumps`), no query operators (phrase, boolean, exclusion, prefix are all treated as bare
terms), and no `k1`/`b` knobs. All three are reported next to the numbers instead.

Live tests require:

    docker run -d --name vdbb-theodb \
      -e POSTGRES_PASSWORD=theo -e POSTGRES_DB=theo -p 55435:5432 theodb:b034

Usage:
    pytest tests/test_theodb_fts.py -v
"""

from __future__ import annotations

import copy
import os
import threading

import pytest
from pydantic import SecretStr

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.payload import PayloadProfile

THEODB_HOST = os.environ.get("THEODB_HOST", "127.0.0.1")
THEODB_PORT = int(os.environ.get("THEODB_PORT", "55435"))
THEODB_USER = os.environ.get("THEODB_USER", "postgres")
THEODB_PASSWORD = os.environ.get("THEODB_PASSWORD", "theo")
THEODB_DBNAME = os.environ.get("THEODB_DBNAME", "theo")

# Relevance is known by construction: A carries both query terms, B one, C none.
DOC_A = "the lazy dog sleeps all day in the warm sun"
DOC_B = "a quick brown fox jumps over the fence"
DOC_C = "postgresql is an advanced open source relational database"
DOCS = [DOC_A, DOC_B, DOC_C]
DOC_IDS = ["a", "b", "c"]
QUERY = "lazy dog"


def _theodb_reachable() -> bool:
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


def _fts_client(drop_old: bool = True, collection: str = "vdbb_fts_live"):
    from vectordb_bench.backend.clients.theodb.config import TheoDBConfig, TheoDBFTSConfig
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB

    db_config = TheoDBConfig(
        user_name=SecretStr(THEODB_USER),
        password=SecretStr(THEODB_PASSWORD),
        host=THEODB_HOST,
        port=THEODB_PORT,
        db_name=THEODB_DBNAME,
    )
    return TheoDB(
        dim=0,
        db_config=db_config.to_dict(),
        db_case_config=TheoDBFTSConfig(metric_type=MetricType.BM25),
        collection_name=collection,
        drop_old=drop_old,
    )


# --------------------------------------------------------------------------------------
# T1.1 — the contract the harness looks for
# --------------------------------------------------------------------------------------


def test_theodb_declares_full_text_support():
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB

    assert TheoDB.supports_full_text_search() is True


@needs_theodb
def test_theodb_declares_a_text_field_and_the_text_payload_profile():
    client = _fts_client(drop_old=True)
    assert client.has_text_field() is True
    assert client.supports_document_payload_profile(PayloadProfile.TEXT) is True
    assert client.supports_document_payload_profile(PayloadProfile.IDS_ONLY) is True


# --------------------------------------------------------------------------------------
# T1.4 — a BM25 parameter TheoDB cannot honour is refused
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "value"), [("k1", 1.5), ("b", 0.6)])
def test_bm25_parameters_are_refused(field, value):
    from vectordb_bench.backend.clients.theodb.config import (
        TheoDBFTSConfig,
        UnsupportedBuildParameterError,
    )

    with pytest.raises(UnsupportedBuildParameterError) as exc:
        TheoDBFTSConfig(metric_type=MetricType.BM25, **{field: value})

    message = str(exc.value)
    assert field in message
    assert "B-040" in message


def test_defaults_are_accepted_and_emit_no_options():
    from vectordb_bench.backend.clients.theodb.config import TheoDBFTSConfig

    assert TheoDBFTSConfig(metric_type=MetricType.BM25).index_param()["options"] == {}


# --------------------------------------------------------------------------------------
# T1.2 — load, build, search against the real engine
# --------------------------------------------------------------------------------------


@needs_theodb
def test_fts_load_build_and_search():
    client = _fts_client(drop_old=True)
    with client.init():
        inserted, error = client.insert_documents(DOCS, DOC_IDS)
        assert error is None, error
        assert inserted == len(DOCS)

        client.optimize(data_size=len(DOCS))

        got = client.search_documents(QUERY, k=5)
        assert got, "empty result after build"
        assert set(got) <= set(DOC_IDS)
        assert all(isinstance(doc_id, str) for doc_id in got)


@needs_theodb
def test_search_before_build_fails_loudly():
    """An unbuilt index must raise, not return [].

    An empty list is indistinguishable from "nothing matched", which would let a whole
    run report recall 0 as if it were a measurement.
    """
    client = _fts_client(drop_old=True, collection="vdbb_fts_unbuilt")
    with client.init():
        client.insert_documents(DOCS, DOC_IDS)
        with pytest.raises(Exception):  # noqa: B017 — engine-defined error type
            client.search_documents(QUERY, k=5)


# --------------------------------------------------------------------------------------
# T1.3 — ranking, which is what NDCG and MRR are computed from
# --------------------------------------------------------------------------------------


@needs_theodb
def test_ranking_puts_the_best_document_first():
    client = _fts_client(drop_old=True)
    with client.init():
        client.insert_documents(DOCS, DOC_IDS)
        client.optimize(data_size=len(DOCS))

        got = client.search_documents(QUERY, k=3)
        assert got[0] == "a", f"expected the two-term document first, got {got}"
        assert "c" not in got, f"document with no query term must not appear: {got}"


@needs_theodb
def test_k_is_respected():
    client = _fts_client(drop_old=False)
    with client.init():
        assert len(client.search_documents("lazy dog sun fox database", k=2)) <= 2


@needs_theodb
def test_query_with_no_matching_term_returns_empty_without_error():
    client = _fts_client(drop_old=False)
    with client.init():
        assert client.search_documents("zebra xylophone", k=5) == []


# --------------------------------------------------------------------------------------
# Concurrency — the runner searches in parallel over a shared lexical index
# --------------------------------------------------------------------------------------


@needs_theodb
def test_concurrent_search_documents_agree():
    """Two threads, two connections, one shared lexical index — same top result."""
    client = _fts_client(drop_old=True)
    with client.init():
        client.insert_documents(DOCS, DOC_IDS)
        client.optimize(data_size=len(DOCS))

    tops: list[list[str]] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            worker = _fts_client(drop_old=False)
            with worker.init():
                tops.append(worker.search_documents(QUERY, k=3))
        except BaseException as exc:  # noqa: BLE001 — surfaced through `errors`
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(tops) == 2
    assert tops[0] == tops[1], f"parallel searches disagreed: {tops}"


@needs_theodb
def test_fts_client_is_deepcopyable_when_idle():
    client = _fts_client(drop_old=False)
    clone = copy.deepcopy(client)
    assert clone.table_name == client.table_name
