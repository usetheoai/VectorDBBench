"""Connection and case configuration for the TheoDB client.

TheoDB ships a `vector` type that is wire-compatible with pgvector and its own
`theodb_hnsw` access method, registered under the alias `hnsw` so pgvector DDL works
unchanged. Since B-036 (`cecd388`) it also honours the pgvector *build* knobs: `m` and
`ef_construction` are real reloptions, validated by the server and recorded in
`pg_class.reloptions`.

The shape of this module is still driven by the same fear, and the fear is still right:
a case config that silently dropped an unsupported knob would let a run complete and
report `ef_construction=200` over an index built with 64 — a wrong measurement that
looks right, which is worse than no measurement at all.

What changed is WHO decides. Until B-046 the answer came from a constant mirrored from
`build.rs`; a constant describes the engine the client was written against, not the one
it is pointed at, so it measures the LABEL instead of the CAPABILITY — the failure the
b047 lexical run documented. The decision now comes from a probe against the live server
(`TheoDB.probe_build_params`), still before the dataset is loaded, which is the only
point where a misdirected run is still cheap to stop.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, LiteralString, NamedTuple, TypedDict

from pydantic import BaseModel, SecretStr, model_validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

# TheoDB's build defaults, kept for documentation only — never for a decision.
#
# They are what an index gets when no `WITH` clause is emitted, and they coincide with
# pgvector's own defaults, which is what makes the no-options comparison between the two
# engines like-for-like without any adjustment. They are deliberately NOT used to accept
# or refuse a requested value: that is the server's job (see the module docstring).
THEODB_HNSW_DEFAULT_M = 16
THEODB_HNSW_DEFAULT_EF_CONSTRUCTION = 64

# The build knobs this client will forward. Membership here means "emit it in WITH and
# let the server rule on it", not "the engine supports it".
FORWARDED_BUILD_PARAMS = ("m", "ef_construction")

# Item that made these real reloptions.
_BUILD_PARAM_ISSUE = "B-036"

# Tracking item for exposing BM25's k1/b. TheoDB has no GUC for either — measured against
# pg_settings, which has no bm25/lexical/k1 entry at all.
_BM25_PARAM_ISSUE = "B-040"


class UnsupportedBuildParameterError(ValueError):
    """Raised when a case asks for a build parameter TheoDB cannot honour.

    Deliberately a hard failure rather than a warning: the alternative is a benchmark
    that reports a parameter it did not apply.
    """

    def __init__(
        self,
        name: str,
        requested: object,
        reason: str,
        remedy: str,
        issue: str,
    ) -> None:
        super().__init__(
            f"TheoDB cannot honour {name}={requested}: {reason}, so running with "
            f"{name}={requested} would report a parameter that was never applied. "
            f"{remedy} Tracking item: {issue}."
        )
        self.name = name
        self.requested = requested
        self.issue = issue


class NotATheoDBError(RuntimeError):
    """Raised when the configured endpoint answers but is not TheoDB.

    pgvector satisfies every other probe this client makes: it has the `vector` type, an
    `hnsw` access method and `vector_l2_ops`. Without this check a misdirected run
    completes and publishes another engine's numbers under the TheoDB label — a
    mislabelled measurement, which is worse than a failed one.
    """

    def __init__(self, host: str, port: int, dbname: str) -> None:
        super().__init__(
            f"{host}:{port}/{dbname} answered, but it is not TheoDB: the access method "
            f"'{THEODB_NATIVE_ACCESS_METHOD}' is absent from pg_am. A plain PostgreSQL "
            f"with pgvector looks identical to every other probe this client makes, so "
            f"the run is refused rather than published under the wrong engine name."
        )


# The own-code access method. The `hnsw` alias exists in pgvector too, so it cannot tell
# the two apart; this name only exists in TheoDB.
THEODB_NATIVE_ACCESS_METHOD = "theodb_hnsw"


class MetricSurface(NamedTuple):
    """The two catalogue names a metric maps to: index opclass and distance operator."""

    opclass: str
    operator: LiteralString


# Verified against the running catalogue: `pg_opclass` joined to `pg_am` for `hnsw`, and
# `pg_operator` for left type `vector`. All three pairs exist under the pgvector names.
_METRIC_SURFACE: dict[MetricType, MetricSurface] = {
    MetricType.L2: MetricSurface("vector_l2_ops", "<->"),
    MetricType.COSINE: MetricSurface("vector_cosine_ops", "<=>"),
    MetricType.IP: MetricSurface("vector_ip_ops", "<#>"),
}


class TheoDBConfigDict(TypedDict):
    """Keys are passed straight to `psycopg.connect`, so they must match its API."""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class TheoDBConfig(DBConfig):
    user_name: SecretStr = SecretStr("postgres")
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> TheoDBConfigDict:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": self.user_name.get_secret_value(),
            "password": self.password.get_secret_value(),
        }


class TheoDBIndexConfig(BaseModel, DBCaseConfig):
    """Shared surface for every TheoDB index type.

    Concrete subclasses fill in the three parameter views. Only HNSW exists today;
    `theodb_ivfflat` has no `ivfflat` alias yet, so an IVF subclass would have nothing
    to create an index with.
    """

    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

    def _metric_surface(self) -> MetricSurface:
        surface = _METRIC_SURFACE.get(self.metric_type)
        if surface is None:
            supported = ", ".join(sorted(m.value for m in _METRIC_SURFACE))
            msg = (
                f"TheoDB has no vector operator class for metric "
                f"{self.metric_type.value if self.metric_type else None}; "
                f"supported: {supported}"
            )
            raise ValueError(msg)
        return surface

    def parse_metric(self) -> str:
        return self._metric_surface().opclass

    def parse_metric_fun_op(self) -> LiteralString:
        return self._metric_surface().operator

    @abstractmethod
    def index_param(self) -> dict: ...

    @abstractmethod
    def search_param(self) -> dict: ...

    @abstractmethod
    def session_param(self) -> dict: ...


class TheoDBHNSWConfig(TheoDBIndexConfig):
    """HNSW over `theodb_hnsw`, addressed through its pgvector-compatible `hnsw` alias.

    `m` and `ef_construction` are forwarded verbatim into the `WITH` clause; whether the
    server honours them is decided by `TheoDB.probe_build_params`, before the load.
    `ef_search` is a session GUC and is applied per connection.
    """

    index: IndexType = IndexType.HNSW
    m: int | None = None
    ef_construction: int | None = None
    ef_search: int | None = None

    def requested_build_params(self) -> dict[str, int]:
        """The build knobs this case asked for, in the order the probe will emit them.

        Empty when nothing was asked — which is what lets the probe be skipped entirely
        rather than paying a transaction to answer a question nobody posed.
        """
        return {
            name: value
            for name in FORWARDED_BUILD_PARAMS
            if (value := getattr(self, name, None)) is not None
        }

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value.lower(),
            "options": self.requested_build_params(),
        }

    def search_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "metric_fun_op": self.parse_metric_fun_op(),
        }

    def session_param(self) -> dict:
        # The native GUC name, not the `hnsw.ef_search` alias. The alias carries a
        # precedence rule (specific name wins), and a benchmark should have the fewest
        # possible variables between the knob and the effect.
        if self.ef_search is None:
            return {}
        return {"theodb_hnsw.ef_search": self.ef_search}


_theodb_case_config = {
    IndexType.HNSW: TheoDBHNSWConfig,
}


class TheoDBFTSConfig(TheoDBIndexConfig):
    """Full-text (BM25) case config.

    TheoDB's BM25 has no tunable knobs: `k1` and `b` are not exposed as GUCs (measured —
    `pg_settings` has no bm25/lexical entry), so a run is product-default and nothing
    else. Accepting either value and running anyway would publish a comparison that
    claims a parameterisation which never happened; the harness explicitly warns that
    engines differ here.
    """

    index: IndexType = IndexType.FTS
    metric_type: MetricType | None = MetricType.BM25
    k1: float | None = None
    b: float | None = None

    _unhonourable_params: ClassVar[tuple[str, ...]] = ("k1", "b")

    def __init__(self, **data: object) -> None:
        self._refuse_unhonourable_bm25_params(data)
        super().__init__(**data)

    @classmethod
    def _refuse_unhonourable_bm25_params(cls, values: object) -> None:
        if not isinstance(values, dict):
            return
        for name in cls._unhonourable_params:
            if values.get(name) is not None:
                raise UnsupportedBuildParameterError(
                    name,
                    values[name],
                    reason=(f"TheoDB's BM25 exposes no GUC for {name}, so the ranking function cannot be tuned at all"),
                    remedy=f"Omit {name}; the run is product-default and must be reported as such.",
                    issue=_BM25_PARAM_ISSUE,
                )

    @model_validator(mode="before")
    @classmethod
    def _refuse_on_every_bm25_construction_path(cls, values: object) -> object:
        cls._refuse_unhonourable_bm25_params(values)
        return values

    def _metric_surface(self) -> MetricSurface:
        # BM25 is not a vector distance: there is no opclass and no operator. The base
        # class's mapping deliberately does not cover it.
        msg = "TheoDBFTSConfig has no vector metric surface; full-text search uses bm25_search"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {"metric": "bm25", "index_type": "bm25", "options": {}}

    def search_param(self) -> dict:
        return {"metric": "bm25"}

    def session_param(self) -> dict:
        return {}


_theodb_case_config[IndexType.FTS] = TheoDBFTSConfig
