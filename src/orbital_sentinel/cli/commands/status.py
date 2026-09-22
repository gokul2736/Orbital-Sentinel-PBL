"""System status commands."""

import click
from pathlib import Path


@click.group(invoke_without_command=True)
@click.pass_context
def status(ctx) -> None:
    """Show system status."""
    if ctx.invoked_subcommand is None:
        _show_full_status()


def _show_full_status() -> None:
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_success, print_warning

    print_header("Orbital Sentinel — System Status")

    data_dir = Path("data/raw/esa_kelvins")
    train_exists = (data_dir / "train_data.csv").exists()
    test_exists = (data_dir / "test_data.csv").exists()

    if train_exists:
        print_success("Dataset: train_data.csv found")
    else:
        print_warning("Dataset: train_data.csv NOT FOUND")
    if test_exists:
        print_success("Dataset: test_data.csv found")
    else:
        print_warning("Dataset: test_data.csv NOT FOUND")

    models_dir = Path("models_saved")
    if models_dir.exists():
        models = sorted(p.stem.replace("_model", "") for p in models_dir.glob("*_model.joblib"))
        if models:
            print_success(f"Models: {', '.join(models)}")
        else:
            print_warning("Models: none trained")
    else:
        print_warning("Models: models_saved/ directory not found")

    scaler_exists = (models_dir / "scaler.joblib").exists() if models_dir.exists() else False
    features_exists = (models_dir / "feature_names.json").exists() if models_dir.exists() else False
    print_metric("Scaler", "OK" if scaler_exists else "MISSING")
    print_metric("Feature names", "OK" if features_exists else "MISSING")

    results_path = Path("proofs/pipeline_results.json")
    if results_path.exists():
        print_success("Pipeline results: available")
    else:
        print_warning("Pipeline results: not found (run training)")

    env_path = Path(".env")
    if env_path.exists():
        from dotenv import load_dotenv
        import os
        load_dotenv()
        has_st = bool(os.getenv("SPACE_TRACK_USERNAME")) and bool(os.getenv("SPACE_TRACK_PASSWORD"))
        if has_st:
            print_success("Space-Track API: configured")
        else:
            print_warning("Space-Track API: credentials missing in .env")
    else:
        print_warning("Space-Track API: .env file not found")


@status.command("models")
def status_models() -> None:
    """List available trained models."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_warning

    print_header("Trained Models")

    models_dir = Path("models_saved")
    if not models_dir.exists():
        print_warning("No models_saved/ directory")
        return

    for path in sorted(models_dir.glob("*_model.joblib")):
        size_mb = path.stat().st_size / 1024 / 1024
        name = path.stem.replace("_model", "")
        print_metric(name, f"{size_mb:.1f} MB")

    other_files = list(models_dir.glob("*.json")) + list(models_dir.glob("scaler.*"))
    if other_files:
        click.echo()
        click.echo(click.style("  Artifacts:", bold=True))
        for f in other_files:
            print_metric(f.name, f"{f.stat().st_size / 1024:.1f} KB")


@status.command("config")
def status_config() -> None:
    """Show current configuration."""
    from orbital_sentinel.cli.formatting import print_header, print_error

    print_header("Configuration")

    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print_error("configs/config.yaml not found")
        return

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    def _print_dict(d: dict, indent: int = 2) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                click.echo(f"{' ' * indent}{click.style(k + ':', bold=True)}")
                _print_dict(v, indent + 4)
            else:
                click.echo(f"{' ' * indent}{k}: {click.style(str(v), fg='cyan')}")

    _print_dict(config)
