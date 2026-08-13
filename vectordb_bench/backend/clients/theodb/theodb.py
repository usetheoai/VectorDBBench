"""Wrapper around TheoDB over the VectorDB port.

TheoDB speaks the PostgreSQL wire protocol and ships a `vector` type binary-compatible
with pgvector, so the fast paths the benchmark cares about — `COPY … FORMAT BINARY` for
load and a prepared, binary k-NN query for search — work through the stock
`pgvector-python` adapters.

One ordering detail is load-bearing: `register_vector` must run BEFORE the cursor is
created. psycopg freezes the adapter map into the cursor's Transformer at construction,
so a cursor made first never sees the `vector` dumpers and every binary path fails with
a misleading "cannot adapt type 'ndarray'".
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile

from ..api import IndexType, VectorDB
from .config import (
    THEODB_NATIVE_ACCESS_METHOD,
    NotATheoDBError,
    UnsupportedBuildParameterError,
)

if TYPE_CHECKING:  # imports used only in annotations (PEP 563 via __future__)
    from collections.abc import Generator

    from .config import TheoDBConfigDict, TheoDBIndexConfig

log = logging.getLogger(__name__)


def _lexical_index_id_for(collection_name: str) -> int:
    """Stable, collision-resistant lexical index id derived from the collection name.

    bm25_build takes a bigint id rather than a name. Hashing the collection keeps a run
    repeatable (a re-run rebuilds the same id instead of leaking a new index every time)
    and keeps two collections in one database apart. blake2b rather than hash() because
    Python randomises string hashing per process, and the runner spawns several.
    """
    digest = hashlib.blake2b(collection_name.encode("utf-8"), digest_size=6).digest()
    return int.from_bytes(digest, "big")


# Nome fixo da tabela temporária da sonda. Fixo de propósito: um nome aleatório tornaria
# o SQL emitido não-asserível, e a tabela é TEMP + dentro de um savepoint revertido.
_PROBE_TABLE = "_theodb_build_param_probe"


def _rollback_probe(cursor) -> None:
    """Desfaz a sonda. Best-effort: se o próprio rollback falhar, o erro que importa é o
    da sonda, e mascará-lo com o do rollback trocaria o diagnóstico pelo sintoma."""
    try:
        cursor.execute("ROLLBACK TO SAVEPOINT theodb_build_param_probe")
    except Exception:  # noqa: BLE001, S110
        pass


class TheoDB(VectorDB):
    """VectorDB adapter for TheoDB (https://github.com/usetheoai/theo-db)."""

    # A psycopg connection cannot be shared across threads; the concurrent runner
    # clones the client and calls init() per thread instead.
    thread_safe: bool = False

    # Filtered search is deliberately not claimed. TheoDB has a filter surface
    # (`theodb.enable_vecfilter`, `theodb_ivfflat_label_ops`) that this client has not
    # measured, and declaring support for an unmeasured path is how a benchmark starts
    # reporting numbers nobody verified.
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    def __init__(
        self,
        dim: int,
        db_config: TheoDBConfigDict,
        db_case_config: TheoDBIndexConfig,
        collection_name: str = "theodb_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "TheoDB"
        self.dim = dim
        self.connect_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name

        self._index_name = f"{collection_name}_theodb_idx"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._text_field = "body"
        # bm25_build needs a BIGINT key; the harness gives opaque string ids. The
        # surrogate carries the first, the doc_id column carries the second, and
        # search_documents joins back so the engine's ranking is what comes out.
        self._rowid_field = "rowid"
        self._doc_id_field = "doc_id"

        # The full-text case has no vector column and no ANN index; it is driven by
        # bm25_build/bm25_search instead. Deciding once here keeps every branch below
        # explicit rather than inferring the mode from whichever field happens to be set.
        self._is_fts = getattr(self.case_config, "index", None) == IndexType.FTS
        # A stable id makes a run repeatable: re-running rebuilds the same index rather
        # than accumulating one per run. Derived from the collection so two collections
        # in one database do not collide.
        self._lexical_index_id = _lexical_index_id_for(collection_name)
        self._lexical_index_built = False

        if not self._is_fts and not (
            self.case_config.create_index_before_load or self.case_config.create_index_after_load
        ):
            msg = (
                f"{self.name} needs create_index_before_load or create_index_after_load; "
                f"a run with no index measures a sequential scan, not the engine."
            )
            raise RuntimeError(msg)

        self.conn, self.cursor = self._create_connection(**self.connect_config)
        self._assert_is_theodb()
        # Antes da carga, que é a única hora em que uma corrida mal-direcionada ainda é
        # barata de parar. Depois dela, descobrir que o parâmetro não é honrado custa o
        # dataset inteiro — foi a lição que o B-035 pagou.
        self.probe_build_params(self.cursor, self.case_config)

        if drop_old:
            if self._is_fts:
                self._drop_table()
                self._create_document_table()
            else:
                self._drop_index()
                self._drop_table()
                self._create_table()
                if self.case_config.create_index_before_load:
                    self._create_index()

        # Release before the runner deep-copies this instance across processes: a live
        # psycopg connection is not picklable (upstream issue #756).
        self._close_connection()

    # ---------------------------------------------------------------- connection

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        # The extension comes first: register_vector looks the `vector` type up in the
        # catalogue and raises when it is absent, so registering before creating is a
        # chicken-and-egg failure — the one that breaks the upstream pgvector client on a
        # clean database (`pgvector.py:93` registers, `pgvector.py:61` creates). CASCADE
        # because TheoDB's `vector` shim requires `theodb_rs`.
        #
        # On the shipped TheoDB image both extensions live in template1, so every new
        # database inherits them and this ordering never bites. It matters anyway: on a
        # database without them, getting past this line is what lets the identity check
        # below report "this is not TheoDB" instead of an opaque "vector type not found".
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector CASCADE")
        conn.commit()
        # BEFORE the cursor — see the module docstring.
        register_vector(conn)
        conn.autocommit = False
        return conn, conn.cursor()

    def _assert_is_theodb(self) -> None:
        """Refuse an endpoint that answers but is not TheoDB.

        Checked once, at construction, before the dataset is loaded — the cheapest point
        at which a misdirected run can still be stopped.
        """
        found = self.cursor.execute(
            "SELECT 1 FROM pg_am WHERE amname = %s",
            (THEODB_NATIVE_ACCESS_METHOD,),
        ).fetchone()
        if found is None:
            self._close_connection()
            raise NotATheoDBError(
                host=self.connect_config["host"],
                port=self.connect_config["port"],
                dbname=self.connect_config["dbname"],
            )

    def _close_connection(self) -> None:
        if self.cursor is not None:
            self.cursor.close()
        if self.conn is not None:
            self.conn.close()
        self.cursor = None
        self.conn = None

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn, self.cursor = self._create_connection(**self.connect_config)
        for name, value in self.case_config.session_param().items():
            statement = sql.SQL("SET {name} = {value}").format(
                name=sql.Identifier(name),
                value=sql.Literal(value),
            )
            log.debug(statement.as_string(self.cursor))
            self.cursor.execute(statement)
        self.conn.commit()
        try:
            yield
        finally:
            self._close_connection()

    # ---------------------------------------------------------------- DDL

    def _drop_table(self) -> None:
        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table}").format(table=sql.Identifier(self.table_name)),
        )
        self.conn.commit()

    def _drop_index(self) -> None:
        self.cursor.execute(
            sql.SQL("DROP INDEX IF EXISTS {index}").format(index=sql.Identifier(self._index_name)),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ full text

    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True

    def has_text_field(self) -> bool:
        return self._is_fts

    def _create_document_table(self) -> None:
        """Table for the full-text case: a surrogate bigint key beside the real id.

        `bm25_build` requires the id column to be BIGINT — measured: a TEXT id fails with
        `invalid input syntax for type bigint`. The harness's document ids are opaque
        strings (MS MARCO passage ids are numeric, HotpotQA's are not), so coercing them
        would work for one dataset and corrupt the next.

        The surrogate is `GENERATED ALWAYS AS IDENTITY` rather than a client-side counter
        because the concurrent insert runner deep-copies this client across processes and
        two counters would collide. Assigning keys is the database's job.
        """
        self.cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS public.{table} ("
                "{rowid} BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                "{doc_id} TEXT NOT NULL, "
                "{body} TEXT NOT NULL)",
            ).format(
                table=sql.Identifier(self.table_name),
                rowid=sql.Identifier(self._rowid_field),
                doc_id=sql.Identifier(self._doc_id_field),
                body=sql.Identifier(self._text_field),
            ),
        )
        self.conn.commit()

    def insert_documents(
        self,
        texts: list[str],
        doc_ids: list[str],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        """Load only. The lexical index is built in optimize() — see the class docstring."""
        try:
            with self.cursor.copy(
                sql.SQL("COPY public.{table} ({doc_id}, {body}) FROM STDIN (FORMAT BINARY)").format(
                    table=sql.Identifier(self.table_name),
                    doc_id=sql.Identifier(self._doc_id_field),
                    body=sql.Identifier(self._text_field),
                ),
            ) as copy:
                copy.set_types(["text", "text"])
                for doc_id, text in zip(doc_ids, texts, strict=True):
                    copy.write_row((doc_id, text))
            self.conn.commit()
        except Exception as exc:
            log.warning(f"{self.name} failed to insert documents into {self.table_name}: {exc}")
            return 0, exc
        return len(doc_ids), None

    def _build_lexical_index(self) -> int:
        """Full rebuild over the loaded table. Returns the indexed document count."""
        indexed = self.cursor.execute(
            "SELECT bm25_build(%s, %s, %s, %s)",
            (self._lexical_index_id, self.table_name, self._rowid_field, self._text_field),
        ).fetchone()[0]
        self.conn.commit()
        self._lexical_index_built = True
        log.info(f"{self.name} built lexical index {self._lexical_index_id}: {indexed} documents")
        return int(indexed)

    def search_documents(
        self,
        query: str,
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        **kwargs: Any,
    ) -> list[str]:
        """Ranked document ids, best first.

        `bm25_search` returns (id, score) ordered by score descending, so the order is
        the engine's — this method never re-sorts. NDCG and MRR are computed from this
        order, so re-ranking here would measure the client rather than the engine.
        """
        self._assert_lexical_index_built()
        rows = self.cursor.execute(
            sql.SQL(
                "SELECT t.{doc_id} FROM bm25_search(%s, %s, %s) AS s "
                "JOIN public.{table} AS t ON t.{rowid} = s.id "
                "ORDER BY s.score DESC",
            ).format(
                doc_id=sql.Identifier(self._doc_id_field),
                table=sql.Identifier(self.table_name),
                rowid=sql.Identifier(self._rowid_field),
            ),
            (self._lexical_index_id, query, k),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _assert_lexical_index_built(self) -> None:
        """Refuse to search an index that was never built.

        Measured: `bm25_search` over an unbuilt index_id returns zero rows and no error,
        which is indistinguishable from "nothing matched". A whole run would then report
        recall 0 as if it were a measurement. The catalogue answers the question as fact
        rather than trusting a client-side flag that a deep-copied process may not carry.
        """
        built = self.cursor.execute(
            "SELECT 1 FROM theodb.lexical_index_meta WHERE index_id = %s",
            (self._lexical_index_id,),
        ).fetchone()
        if built is None:
            msg = (
                f"lexical index {self._lexical_index_id} for collection "
                f"'{self.table_name}' was never built: bm25_search would return an empty "
                f"result that is indistinguishable from 'nothing matched'. Call optimize() "
                f"after loading the documents."
            )
            raise RuntimeError(msg)

    def _create_table(self) -> None:
        self.cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS public.{table} ({primary} BIGINT PRIMARY KEY, {vector} vector({dim}))",
            ).format(
                table=sql.Identifier(self.table_name),
                primary=sql.Identifier(self._primary_field),
                vector=sql.Identifier(self._vector_field),
                dim=sql.Literal(self.dim),
            ),
        )
        # PLAIN keeps vectors off TOAST, matching what the pgvector client does so the
        # two engines are compared on the same storage decision.
        self.cursor.execute(
            sql.SQL("ALTER TABLE public.{table} ALTER COLUMN {vector} SET STORAGE PLAIN").format(
                table=sql.Identifier(self.table_name),
                vector=sql.Identifier(self._vector_field),
            ),
        )
        self.conn.commit()

    @staticmethod
    def _render_with_clause(options: dict) -> str:
        """`WITH (k = v, ...)`, or the empty string when nothing was asked.

        The empty case is not cosmetic: `WITH ()` is a syntax error, so a client that
        always emitted the clause would break every default run.
        """
        if not options:
            return ""
        rendered = ", ".join(f"{name} = {int(value)}" for name, value in options.items())
        return f" WITH ({rendered})"

    @staticmethod
    def render_create_index(
        table_name: str,
        index_name: str,
        vector_field: str,
        case_config: TheoDBIndexConfig,
    ) -> str:
        """Render the CREATE INDEX statement.

        Pure and static so the emitted SQL can be asserted without a database. Since
        B-036 the `WITH` clause carries whatever build knobs the case asked for; whether
        the server honours them was already settled by `probe_build_params`, before the
        load — so reaching this point with an unsupported knob is impossible by
        construction, not by hope.
        """
        index_param = case_config.index_param()
        return (
            f'CREATE INDEX IF NOT EXISTS "{index_name}" ON public."{table_name}" '
            f'USING {index_param["index_type"]} ("{vector_field}" {index_param["metric"]})'
            f"{TheoDB._render_with_clause(index_param['options'])}"
        )

    @staticmethod
    def render_build_param_probe(case_config: TheoDBIndexConfig) -> str:
        """The `CREATE INDEX` the probe will try, asking for EXACTLY what was requested.

        Emitting a fixed pair here would pass against any server and then let an
        unsupported value through — the original defect with one more layer of
        indirection. The probe's whole worth is that it tests the real request.
        """
        options = case_config.index_param()["options"]
        return (
            f"CREATE INDEX ON {_PROBE_TABLE} "
            f'USING hnsw (e {case_config.index_param()["metric"]})'
            f"{TheoDB._render_with_clause(options)}"
        )

    @staticmethod
    def probe_build_params(cursor, case_config: TheoDBIndexConfig, probe_sql: str | None = None) -> None:
        """Ask the SERVER whether it honours the requested build knobs. Raises if not.

        Runs before the dataset is loaded, inside a savepoint that is always rolled back:
        a probe that left a table behind would make the next run inherit state, and a run
        that inherits state is not reproducible.

        Why a probe and not a version check: a version string describes the build the
        client was compiled against, and comparing it measures the LABEL rather than the
        CAPABILITY. That distinction is not theoretical here — the b047 lexical run
        published a 6.4% advantage that was entirely an artefact of trusting a label.
        """
        requested = case_config.index_param()["options"]
        if not requested:
            return
        sql = probe_sql or TheoDB.render_build_param_probe(case_config)
        try:
            cursor.execute("SAVEPOINT theodb_build_param_probe")
            cursor.execute(f"CREATE TEMP TABLE {_PROBE_TABLE} (e vector(2))")
            cursor.execute(sql)
        except Exception as exc:  # noqa: BLE001 — re-raised as a typed error below
            message = str(exc)
            _rollback_probe(cursor)
            name = next(
                (n for n in requested if f'"{n}"' in message),
                ", ".join(requested),
            )
            raise UnsupportedBuildParameterError(
                name,
                requested.get(name, requested),
                reason=f"the server refused it: {message.strip()}",
                remedy=(
                    "Point the run at a TheoDB that registers this reloption "
                    "(B-036 shipped it), or omit the parameter."
                ),
                issue="B-046",
            ) from exc
        _rollback_probe(cursor)

    def _create_index(self) -> None:
        statement = self.render_create_index(
            table_name=self.table_name,
            index_name=self._index_name,
            vector_field=self._vector_field,
            case_config=self.case_config,
        )
        log.info(f"{self.name} creating index: {statement}")
        self.cursor.execute(statement)
        self.conn.commit()

    def optimize(self, data_size: int | None = None) -> None:
        if self._is_fts:
            # Full rebuild, once, after the load. Building per batch would be quadratic
            # and would make the harness's build-time metric meaningless.
            self._build_lexical_index()
            return
        # Exactly what the pgvector client does — drop and rebuild. No VACUUM, no
        # ANALYZE: a maintenance step applied to one side only would tilt the comparison
        # while the published table said nothing about it.
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    # ---------------------------------------------------------------- data path

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        try:
            vectors = np.asarray(embeddings, dtype=np.float32)
            with self.cursor.copy(
                sql.SQL("COPY public.{table} FROM STDIN (FORMAT BINARY)").format(
                    table=sql.Identifier(self.table_name),
                ),
            ) as copy:
                copy.set_types(["bigint", "vector"])
                for offset, row_id in enumerate(metadata):
                    copy.write_row((row_id, vectors[offset]))
            self.conn.commit()
        except Exception as exc:
            log.warning(f"{self.name} failed to insert into {self.table_name}: {exc}")
            return 0, exc
        return len(metadata), None

    def prepare_filter(self, filters: Filter) -> None:
        if filters.type != FilterOp.NonFilter:
            msg = f"{self.name} supports NonFilter only, got {filters.type}"
            raise ValueError(msg)

    def _search_statement(self) -> sql.Composed:
        return sql.SQL(
            "SELECT {primary} FROM public.{table} ORDER BY {vector} {operator} %s::vector LIMIT %s::int",
        ).format(
            primary=sql.Identifier(self._primary_field),
            table=sql.Identifier(self.table_name),
            vector=sql.Identifier(self._vector_field),
            operator=sql.SQL(self.case_config.search_param()["metric_fun_op"]),
        )

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs: Any,
    ) -> list[int]:
        result = self.cursor.execute(
            self._search_statement(),
            (np.asarray(query, dtype=np.float32), k),
            prepare=True,
            binary=True,
        )
        return [int(row[0]) for row in result.fetchall()]

    def need_normalize_cosine(self) -> bool:
        # TheoDB implements cosine distance natively (`vector_cosine_ops`, `<=>`), so
        # normalising the dataset would change the input rather than the comparison.
        return False

    # ---------------------------------------------------------------- introspection

    def explain_search(self, query: list[float], k: int = 100) -> str:
        """Plan for the search statement — used to prove the index is actually used."""
        rows = self.cursor.execute(
            sql.SQL("EXPLAIN (COSTS OFF) ") + self._search_statement(),
            (np.asarray(query, dtype=np.float32), k),
            binary=True,
        ).fetchall()
        return "\n".join(row[0] for row in rows)

    def current_setting(self, name: str) -> str:
        return self.cursor.execute("SELECT current_setting(%s)", (name,)).fetchone()[0]
