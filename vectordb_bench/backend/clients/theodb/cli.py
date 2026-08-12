import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class TheoDBTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", default="postgres", show_default=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="TheoDB database password",
            default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
            show_default="$POSTGRES_PASSWORD",
        ),
    ]
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[int, click.option("--port", type=int, help="Db port", default=5432, show_default=True)]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]


class TheoDBHNSWTypedDict(TheoDBTypedDict):
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=int,
            help="Scan candidate-list size (theodb_hnsw.ef_search)",
        ),
    ]
    m: Annotated[
        int,
        click.option(
            "--m",
            type=int,
            help="Graph degree. TheoDB builds at m=16 and REFUSES any other value (item B-036)",
        ),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            help="Build candidate-list size. TheoDB builds at 64 and REFUSES any other value (item B-036)",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TheoDBHNSWTypedDict)
def TheoDBHNSW(**parameters: Unpack[TheoDBHNSWTypedDict]):
    from .config import TheoDBConfig, TheoDBHNSWConfig

    run(
        db=DB.TheoDB,
        db_config=TheoDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=TheoDBHNSWConfig(
            ef_search=parameters["ef_search"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
        ),
        **parameters,
    )
