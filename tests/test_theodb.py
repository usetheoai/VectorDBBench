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

# A plain PostgreSQL with pgvector installed, used as the negative control: it answers,
# it has the `vector` type and an `hnsw` access method, and it is NOT TheoDB.
IMPOSTOR_PORT = int(os.environ.get("IMPOSTOR_PORT", "55436"))

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


def test_build_params_are_carried_by_the_config_not_refused_by_a_constant():
    """B-046: `m`/`ef_construction` sao reloptions honradas desde o B-036 (cecd388).

    O guard do B-035 comparava contra a constante THEODB_HNSW_M = 16 e recusava tudo o
    mais. A constante deixou de descrever o motor, e um cliente que decide por ela mede o
    ROTULO em vez da CAPACIDADE — a classe de erro que o b047 documentou. A decisao passa
    para uma sonda contra o servidor.
    """
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    config = TheoDBHNSWConfig(metric_type=MetricType.L2, m=32, ef_construction=200)
    assert config.index_param()["options"] == {"m": 32, "ef_construction": 200}


def test_unset_build_params_emit_no_options():
    """Sem pedido explicito, nada e emitido — o indice nasce com o default do motor."""
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    assert TheoDBHNSWConfig(metric_type=MetricType.L2).index_param()["options"] == {}


def test_create_index_statement_carries_the_with_clause():
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    statement = TheoDB.render_create_index(
        table_name="t",
        index_name="t_idx",
        vector_field="embedding",
        case_config=TheoDBHNSWConfig(metric_type=MetricType.L2, m=32, ef_construction=200),
    )
    assert "WITH (m = 32, ef_construction = 200)" in statement
    assert "USING hnsw" in statement
    assert "vector_l2_ops" in statement


def test_create_index_statement_omits_the_with_clause_when_nothing_was_asked():
    """Emitir WITH () vazio e erro de sintaxe; omitir e o comportamento correto."""
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    statement = TheoDB.render_create_index(
        table_name="t",
        index_name="t_idx",
        vector_field="embedding",
        case_config=TheoDBHNSWConfig(metric_type=MetricType.L2),
    )
    assert "WITH" not in statement


def test_the_probe_sql_asks_for_exactly_what_was_requested():
    """A sonda tem de testar as opcoes PEDIDAS, nao um par fixo.

    Uma sonda que sempre testasse m=16 passaria contra qualquer servidor e liberaria
    m=32 num motor que nao o suporta — precisamente o defeito que o guard existe para
    impedir, com um passo a mais de indirecao.
    """
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    sql = TheoDB.render_build_param_probe(
        TheoDBHNSWConfig(metric_type=MetricType.L2, m=32, ef_construction=200)
    )
    assert "m = 32" in sql
    assert "ef_construction = 200" in sql
    assert "USING hnsw" in sql


def test_probe_translates_a_server_refusal_into_the_typed_error():
    """Sem banco: o tradutor de erro e puro e testavel isoladamente."""
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from vectordb_bench.backend.clients.theodb.config import (
        TheoDBHNSWConfig,
        UnsupportedBuildParameterError,
    )

    class _RefusingCursor:
        def execute(self, sql, *a, **k):
            raise RuntimeError('unrecognized parameter "m"')

    with pytest.raises(UnsupportedBuildParameterError) as exc:
        TheoDB.probe_build_params(
            _RefusingCursor(), TheoDBHNSWConfig(metric_type=MetricType.L2, m=32)
        )
    message = str(exc.value)
    assert "unrecognized parameter" in message, message
    # A mensagem cita o que o SERVIDOR disse, nao a constante do cliente. Sem esta
    # assercao o teste passaria com o guard antigo, que e a regressao a impedir.
    assert "build.rs:22-23" not in message, message


def test_probe_is_skipped_when_no_build_param_was_requested():
    """Sem m/ef_construction pedidos nao ha o que sondar — e sondar custaria uma
    transacao por corrida sem responder pergunta nenhuma."""
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    class _ExplodingCursor:
        def execute(self, *a, **k):
            raise AssertionError("a sonda nao devia ter rodado")

    TheoDB.probe_build_params(_ExplodingCursor(), TheoDBHNSWConfig(metric_type=MetricType.L2))


def test_probe_rolls_back_and_leaves_nothing_behind():
    """A sonda cria uma tabela temporaria e um indice. Se algum deles sobreviver, a
    proxima corrida herda estado — e uma corrida que herda estado nao e reproduzivel."""
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB
    from vectordb_bench.backend.clients.theodb.config import TheoDBHNSWConfig

    executed: list[str] = []

    class _RecordingCursor:
        def execute(self, sql, *a, **k):
            executed.append(str(sql))

    TheoDB.probe_build_params(
        _RecordingCursor(), TheoDBHNSWConfig(metric_type=MetricType.L2, m=32)
    )
    joined = " ".join(executed).upper()
    assert "SAVEPOINT" in joined, executed
    assert "ROLLBACK TO" in joined, executed


@needs_theodb
def test_build_params_reach_the_catalog_against_a_real_server(corpus):
    """T1.2 — a prova e o CATALOGO, nao o SQL emitido.

    Um servidor que aceitasse a clausula e a ignorasse devolveria reloptions NULL aqui, e
    e exatamente esse caso que separa "aceito" de "honrado" — a distincao que o B-034
    pagou para aprender.
    """
    import psycopg
    from pydantic import SecretStr
    from vectordb_bench.backend.clients.theodb.config import TheoDBConfig, TheoDBHNSWConfig
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB

    client = TheoDB(
        dim=DIM,
        db_config=TheoDBConfig(
            user_name=SecretStr(THEODB_USER), password=SecretStr(THEODB_PASSWORD),
            host=THEODB_HOST, port=THEODB_PORT, db_name=THEODB_DBNAME,
        ).to_dict(),
        db_case_config=TheoDBHNSWConfig(metric_type=MetricType.L2, m=32, ef_construction=200),
        collection_name="vdbb_b046_params",
        drop_old=True,
    )
    with client.init():
        client.insert_embeddings(corpus[0][:500], list(range(500)))
        client.optimize()

    with psycopg.connect(
        host=THEODB_HOST, port=THEODB_PORT, user=THEODB_USER,
        password=THEODB_PASSWORD, dbname=THEODB_DBNAME,
    ) as conn:
        # O JOIN em pg_am nao e enfeite: sem ele a consulta tambem casa o indice de
        # chave primaria (`..._pkey`, btree), cujas reloptions sao legitimamente NULL —
        # e a primeira versao deste teste reprovou por ler essa linha, acusando o codigo
        # de um defeito que era da assercao. Filtrar pelo access method faz o teste
        # afirmar tambem que o indice esta no `hnsw`, que e parte do que se quer provar.
        rows = conn.execute(
            "SELECT c.relname, c.reloptions FROM pg_class c "
            "JOIN pg_am am ON am.oid = c.relam "
            "WHERE c.relname LIKE %s AND c.relkind = 'i' AND am.amname = 'hnsw'",
            ("%b046_params%",),
        ).fetchall()

    assert len(rows) == 1, f"esperado exatamente 1 indice hnsw, veio {rows}"
    _, options = rows[0]
    assert options is not None, "o indice nasceu sem reloptions: a clausula WITH nao chegou"
    assert sorted(options) == ["ef_construction=200", "m=32"], options


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


# --------------------------------------------------------------------------------------
# Identity — the client must not measure a database that is not TheoDB
# --------------------------------------------------------------------------------------


def _impostor_reachable() -> bool:
    try:
        import psycopg

        with psycopg.connect(
            host=THEODB_HOST,
            port=IMPOSTOR_PORT,
            user=THEODB_USER,
            password=THEODB_PASSWORD,
            dbname=THEODB_DBNAME,
            connect_timeout=3,
        ) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _impostor_reachable(), reason=f"no impostor DB at :{IMPOSTOR_PORT}")
def test_client_refuses_a_database_that_is_not_theodb():
    """pgvector answers every probe the client would otherwise make.

    It has the `vector` type, an `hnsw` access method and `vector_l2_ops`. Without an
    identity check the run completes and its numbers are published under the TheoDB
    label — a mislabelled measurement, which is worse than a failed one.
    """
    from pydantic import SecretStr

    from vectordb_bench.backend.clients.theodb.config import (
        NotATheoDBError,
        TheoDBConfig,
        TheoDBHNSWConfig,
    )
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB

    config = TheoDBConfig(
        user_name=SecretStr(THEODB_USER),
        password=SecretStr(THEODB_PASSWORD),
        host=THEODB_HOST,
        port=IMPOSTOR_PORT,
        db_name=THEODB_DBNAME,
    )
    with pytest.raises(NotATheoDBError) as exc:
        TheoDB(
            dim=DIM,
            db_config=config.to_dict(),
            db_case_config=TheoDBHNSWConfig(metric_type=MetricType.L2),
            collection_name="vdbb_theodb_identity",
            drop_old=True,
        )

    message = str(exc.value)
    assert "theodb_hnsw" in message
    assert str(IMPOSTOR_PORT) in message


@needs_theodb
def test_client_accepts_a_real_theodb():
    client = _live_client(drop_old=False)
    assert client.table_name == "vdbb_theodb_live"


@needs_theodb
def test_client_works_against_a_database_without_the_extension():
    """The client must create the extension before registering the vector adapters.

    `register_vector` looks the type up in the catalogue and raises when it is absent, so
    calling it before `CREATE EXTENSION` is a chicken-and-egg failure. It stays invisible
    on the shipped TheoDB image, which installs the extension into template1 so every new
    database inherits it — and it is exactly what broke the upstream pgvector client on a
    clean machine (`pgvector.py:93` registers before `pgvector.py:61` creates).
    """
    import psycopg
    from pydantic import SecretStr

    from vectordb_bench.backend.clients.theodb.config import TheoDBConfig, TheoDBHNSWConfig
    from vectordb_bench.backend.clients.theodb.theodb import TheoDB

    admin = psycopg.connect(
        host=THEODB_HOST, port=THEODB_PORT, user=THEODB_USER,
        password=THEODB_PASSWORD, dbname=THEODB_DBNAME, autocommit=True,
    )
    admin.execute("DROP DATABASE IF EXISTS vdbb_no_ext")
    admin.execute("CREATE DATABASE vdbb_no_ext")
    try:
        bare = psycopg.connect(
            host=THEODB_HOST, port=THEODB_PORT, user=THEODB_USER,
            password=THEODB_PASSWORD, dbname="vdbb_no_ext", autocommit=True,
        )
        bare.execute("DROP EXTENSION IF EXISTS vector CASCADE")
        # theodb_rs owns the `vector` TYPE; dropping only the shim leaves the type
        # behind and the scenario under test never materialises. Measured.
        bare.execute("DROP EXTENSION IF EXISTS theodb CASCADE")
        bare.execute("DROP EXTENSION IF EXISTS theodb_rs CASCADE")
        assert bare.execute("SELECT count(*) FROM pg_type WHERE typname='vector'").fetchone()[0] == 0
        bare.close()

        config = TheoDBConfig(
            user_name=SecretStr(THEODB_USER), password=SecretStr(THEODB_PASSWORD),
            host=THEODB_HOST, port=THEODB_PORT, db_name="vdbb_no_ext",
        )
        client = TheoDB(
            dim=DIM, db_config=config.to_dict(),
            db_case_config=TheoDBHNSWConfig(metric_type=MetricType.L2),
            collection_name="vdbb_no_ext_t", drop_old=True,
        )
        assert client.table_name == "vdbb_no_ext_t"
    finally:
        admin.execute("DROP DATABASE IF EXISTS vdbb_no_ext WITH (FORCE)")
        admin.close()
