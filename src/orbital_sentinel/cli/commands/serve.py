"""Server commands for API and dashboard."""

import click


@click.group()
def serve():
    """Start API or dashboard servers."""


@serve.command("api")
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", type=int, default=8000, help="Bind port")
@click.option("--reload/--no-reload", default=False, help="Auto-reload on file changes")
def serve_api(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI REST API server."""
    from orbital_sentinel.cli.formatting import print_header, print_metric

    print_header("Orbital Sentinel API Server")
    print_metric("Host", host)
    print_metric("Port", str(port))
    print_metric("Reload", str(reload))
    print_metric("Docs", f"http://{host}:{port}/docs")

    import uvicorn
    uvicorn.run(
        "orbital_sentinel.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@serve.command("dashboard")
@click.option("--port", type=int, default=8501, help="Dashboard port")
def serve_dashboard(port: int) -> None:
    """Start the Streamlit dashboard."""
    from orbital_sentinel.cli.formatting import print_header, print_metric

    print_header("Orbital Sentinel Dashboard")
    print_metric("Port", str(port))

    import subprocess
    import sys
    from pathlib import Path

    dashboard_path = Path(__file__).resolve().parents[3] / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
         "--server.port", str(port)],
        check=True,
    )
