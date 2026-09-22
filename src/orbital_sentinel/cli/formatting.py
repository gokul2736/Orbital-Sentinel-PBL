"""CLI output formatting utilities."""

import click


def print_header(text: str) -> None:
    """Print a styled section header."""
    click.echo()
    click.echo(click.style(f"{'=' * 60}", fg="cyan"))
    click.echo(click.style(f"  {text}", fg="white", bold=True))
    click.echo(click.style(f"{'=' * 60}", fg="cyan"))
    click.echo()


def print_subheader(text: str) -> None:
    """Print a styled subsection header."""
    click.echo()
    click.echo(click.style(f"--- {text} ---", fg="cyan"))


def print_metric(label: str, value: str, color: str = "white") -> None:
    """Print a label-value metric pair."""
    click.echo(f"  {click.style(label + ':', fg='bright_black', bold=True):40s} {click.style(str(value), fg=color)}")


def print_risk_badge(category: str) -> str:
    """Return a colored risk category string."""
    colors = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "cyan", "NEGLIGIBLE": "green"}
    color = colors.get(category.upper(), "white")
    return click.style(f"[{category.upper()}]", fg=color, bold=True)


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int] = None) -> None:
    """Print a formatted table."""
    if not col_widths:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]

    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    click.echo(click.style(f"  {header_line}", fg="bright_white", bold=True))
    click.echo(click.style(f"  {''.join('-' * w for w in col_widths)}", fg="bright_black"))

    for row in rows:
        line = "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        click.echo(f"  {line}")


def print_success(msg: str) -> None:
    click.echo(click.style(f"  [OK] {msg}", fg="green"))


def print_error(msg: str) -> None:
    click.echo(click.style(f"  [ERROR] {msg}", fg="red"))


def print_warning(msg: str) -> None:
    click.echo(click.style(f"  [WARN] {msg}", fg="yellow"))


def print_info(msg: str) -> None:
    click.echo(click.style(f"  [INFO] {msg}", fg="cyan"))
