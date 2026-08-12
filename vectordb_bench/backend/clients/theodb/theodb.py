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

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB

if TYPE_CHECKING:  # imports used only in annotations (PEP 563 via __future__)
    from collections.abc import Generator

    from .config import TheoDBConfigDict, TheoDBIndexConfig

log = logging.getLogger(__name__)


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

        if not (self.case_config.create_index_before_load or self.case_config.create_index_after_load):
            msg = (
                f"{self.name} needs create_index_before_load or create_index_after_load; "
                f"a run with no index measures a sequential scan, not the engine."
            )
            raise RuntimeError(msg)

        self.conn, self.cursor = self._create_connection(**self.connect_config)
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.commit()

        if drop_old:
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
        # BEFORE the cursor — see the module docstring.
        register_vector(conn)
        conn.autocommit = False
        return conn, conn.cursor()

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
    def render_create_index(
        table_name: str,
        index_name: str,
        vector_field: str,
        case_config: TheoDBIndexConfig,
    ) -> str:
        """Render the CREATE INDEX statement.

        Pure and static so the emitted SQL can be asserted without a database. There is
        deliberately no WITH clause: TheoDB's `hnsw` access method rejects `m` and
        `ef_construction` as reloptions, and the config layer has already refused any
        value that would need one.
        """
        index_param = case_config.index_param()
        if index_param["options"]:
            msg = f"TheoDB emits no index options, got {index_param['options']}"
            raise AssertionError(msg)
        return (
            f'CREATE INDEX IF NOT EXISTS "{index_name}" ON public."{table_name}" '
            f'USING {index_param["index_type"]} ("{vector_field}" {index_param["metric"]})'
        )

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
