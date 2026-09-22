"""Data management commands."""

import click


@click.group()
def data():
    """Manage datasets and CDM data."""


@data.command("info")
def data_info() -> None:
    """Show dataset information."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_error

    print_header("Dataset Information")

    try:
        from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
        train_df = load_esa_kelvins(split="train")

        print_metric("Rows", f"{len(train_df):,}")
        print_metric("Columns", str(train_df.shape[1]))
        print_metric("Memory", f"{train_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        n_events = train_df["event_id"].nunique() if "event_id" in train_df.columns else 0
        print_metric("Unique events", f"{n_events:,}")

        null_pct = train_df.isnull().mean().mean() * 100
        print_metric("Mean null %", f"{null_pct:.2f}%")

        if "risk" in train_df.columns:
            click.echo()
            click.echo(click.style("  Risk Distribution:", bold=True))
            print_metric("HIGH (> -5)", f"{(train_df['risk'] > -5).sum():,}")
            print_metric("MEDIUM (-7 to -5)", f"{((train_df['risk'] > -7) & (train_df['risk'] <= -5)).sum():,}")
            print_metric("LOW (-15 to -7)", f"{((train_df['risk'] > -15) & (train_df['risk'] <= -7)).sum():,}")
            print_metric("NEGLIGIBLE (< -15)", f"{(train_df['risk'] <= -15).sum():,}")

        try:
            test_df = load_esa_kelvins(split="test")
            click.echo()
            print_metric("Test rows", f"{len(test_df):,}")
        except FileNotFoundError:
            pass

    except FileNotFoundError:
        print_error("Dataset not found at data/raw/esa_kelvins/")


@data.command("validate")
def data_validate() -> None:
    """Validate dataset schema and quality."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_success, print_warning, print_error

    print_header("Data Validation")

    try:
        from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
        from orbital_sentinel.validation.schema import validate_schema
        from orbital_sentinel.validation.quality import assess_quality

        df = load_esa_kelvins(split="train")

        schema_issues = validate_schema(df)
        if schema_issues:
            for issue in schema_issues:
                print_warning(f"Schema: {issue}")
        else:
            print_success("Schema validation passed")

        quality = assess_quality(df)
        print_metric("Null columns", str(quality.get("columns_with_nulls", "N/A")))
        print_metric("Risk floor %", f"{quality.get('risk_floor_pct', 0):.1f}%")
        print_metric("Neg miss distance", str(quality.get("negative_miss_distance_count", 0)))
        print_metric("Neg speed", str(quality.get("negative_speed_count", 0)))

        outliers = quality.get("outlier_columns", [])
        if outliers:
            print_warning(f"Outlier columns ({len(outliers)}): {', '.join(outliers[:10])}")
        else:
            print_success("No outlier columns detected")

    except Exception as e:
        print_error(str(e))


@data.command("fetch-cdm")
@click.option("--count", type=int, default=5, help="Number of CDMs to fetch")
@click.option("--output", "output_path", type=click.Path(), help="Save mapped CDMs to CSV")
def data_fetch_cdm(count: int, output_path: str) -> None:
    """Fetch CDMs from Space-Track.org."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_success, print_error, print_info

    print_header("Fetch Live CDMs")

    try:
        from orbital_sentinel.ingestion.cdm_api import SpaceTrackClient
        from orbital_sentinel.ingestion.cdm_mapper import map_cdm_to_esa_format

        print_info("Connecting to Space-Track.org...")
        client = SpaceTrackClient()
        client.login()
        raw_cdms = client.fetch_cdm(limit=count)
        client.close()

        print_success(f"Fetched {len(raw_cdms)} CDMs")

        for i, cdm in enumerate(raw_cdms):
            sat1 = cdm.get("SAT_1_NAME", "?")
            sat2 = cdm.get("SAT_2_NAME", "?")
            miss = cdm.get("MISS_DISTANCE", "?")
            click.echo(f"  #{i+1}: {sat1} vs {sat2} | Miss: {miss} km | TCA: {cdm.get('TCA', '?')}")

        if output_path:
            import pandas as pd
            mapped = [map_cdm_to_esa_format(c) for c in raw_cdms]
            pd.DataFrame(mapped).to_csv(output_path, index=False)
            print_success(f"Saved to {output_path}")

    except Exception as e:
        print_error(str(e))


@data.command("sample")
@click.option("--n", type=int, default=5, help="Number of rows")
@click.option("--risk-filter", type=click.Choice(["all", "high", "medium", "low"]), default="all")
def data_sample(n: int, risk_filter: str) -> None:
    """Show sample data rows."""
    from orbital_sentinel.cli.formatting import print_header, print_error

    print_header("Sample Data")

    try:
        from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
        df = load_esa_kelvins(split="train")

        if risk_filter == "high":
            df = df[df["risk"] > -5]
        elif risk_filter == "medium":
            df = df[(df["risk"] > -7) & (df["risk"] <= -5)]
        elif risk_filter == "low":
            df = df[(df["risk"] > -15) & (df["risk"] <= -7)]

        sample = df.head(n)
        key_cols = ["event_id", "time_to_tca", "risk", "miss_distance", "relative_speed"]
        avail = [c for c in key_cols if c in sample.columns]

        for _, row in sample.iterrows():
            parts = [f"{c}={row[c]:.4f}" if isinstance(row[c], float) else f"{c}={row[c]}" for c in avail]
            click.echo(f"  {' | '.join(parts)}")

    except Exception as e:
        print_error(str(e))
