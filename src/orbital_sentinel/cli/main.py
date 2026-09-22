"""Orbital Sentinel CLI — main entry point."""

import click

from orbital_sentinel.cli.commands.train import train
from orbital_sentinel.cli.commands.predict import predict
from orbital_sentinel.cli.commands.serve import serve
from orbital_sentinel.cli.commands.data import data
from orbital_sentinel.cli.commands.status import status


@click.group()
@click.version_option(version="1.0.0", prog_name="orbital-sentinel")
def cli():
    """Orbital Sentinel — Satellite Conjunction Risk Assessment CLI.

    ML-powered triage system for space debris conjunction analysis.
    """


cli.add_command(train)
cli.add_command(predict)
cli.add_command(serve)
cli.add_command(data)
cli.add_command(status)


if __name__ == "__main__":
    cli()
