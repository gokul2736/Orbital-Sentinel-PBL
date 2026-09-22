"""Prediction commands."""

import click


@click.group()
def predict():
    """Run collision risk predictions."""


@predict.command("single")
@click.option("--event-id", type=int, required=True, help="Event ID from the dataset")
@click.option("--model", default="XGBoost", help="Model name to use")
@click.option("--include-physics/--no-physics", default=True)
@click.option("--include-explanation/--no-explanation", default=True)
def predict_single(event_id: int, model: str, include_physics: bool, include_explanation: bool) -> None:
    """Predict risk for a single conjunction event."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_risk_badge, print_error

    print_header(f"Predicting Event {event_id}")

    try:
        from orbital_sentinel.inference.pipeline import load_inference_artifacts, predict_single_event
        from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins

        artifacts = load_inference_artifacts(model_name=model)
        if artifacts["model"] is None:
            print_error(f"Model '{model}' not found")
            return

        df = load_esa_kelvins(split="train")
        event_rows = df[df["event_id"] == event_id]
        if event_rows.empty:
            print_error(f"Event {event_id} not found in dataset")
            return

        row_dict = event_rows.iloc[-1].to_dict()
        result = predict_single_event(
            row_dict, artifacts["model"], artifacts["scaler"], artifacts["feature_names"],
            include_physics=include_physics, include_explanation=include_explanation,
        )

        print_metric("Predicted Risk", f"{result['prediction']:.4f}")
        click.echo(f"  {'Risk Category:':40s} {print_risk_badge(result['risk_category'])}")
        print_metric("Confidence", f"{result.get('confidence', 'N/A')}")

        interval = result.get("interval")
        if interval:
            print_metric("90% Interval", f"[{interval[0]:.2f}, {interval[1]:.2f}]")

        true_risk = row_dict.get("risk")
        if true_risk is not None:
            print_metric("True Risk", f"{true_risk:.4f}", color="cyan")

        physics = result.get("physics_result")
        if physics:
            click.echo()
            click.echo(click.style("  Physics Verification:", bold=True))
            log_pc = physics.get("log10_pc")
            if log_pc is not None:
                print_metric("Analytic log10(Pc)", f"{log_pc:.2f}")
            print_metric("Covariance Valid", str(physics.get("covariance_valid", "N/A")))

        if result.get("explanation"):
            click.echo()
            click.echo(click.style("  AI Explanation:", bold=True))
            click.echo(f"  {result['explanation'][:500]}")

    except Exception as e:
        print_error(str(e))


@predict.command("batch")
@click.option("--input", "input_path", type=click.Path(exists=True), help="Input CSV file")
@click.option("--output", "output_path", type=click.Path(), help="Output CSV file")
@click.option("--model", default="XGBoost")
@click.option("--limit", type=int, default=None, help="Limit number of rows")
def predict_batch(input_path: str, output_path: str, model: str, limit: int) -> None:
    """Batch prediction on a dataset."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_success, print_error
    import pandas as pd

    print_header("Batch Prediction")

    try:
        from orbital_sentinel.inference.pipeline import load_inference_artifacts, predict_batch as run_batch

        artifacts = load_inference_artifacts(model_name=model)
        if artifacts["model"] is None:
            print_error(f"Model '{model}' not found")
            return

        if input_path:
            df = pd.read_csv(input_path)
        else:
            from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
            df = load_esa_kelvins(split="test")

        if limit:
            df = df.head(limit)

        print_metric("Input rows", f"{len(df):,}")

        results = run_batch(df, artifacts["model"], artifacts["scaler"], artifacts["feature_names"])

        categories = {}
        for r in results:
            cat = r["risk_category"]
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items()):
            print_metric(cat, f"{count:,}")

        if output_path:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print_success(f"Results saved to {output_path}")

    except Exception as e:
        print_error(str(e))


@predict.command("live")
@click.option("--count", type=int, default=5, help="Number of CDMs to fetch")
@click.option("--model", default="XGBoost")
def predict_live(count: int, model: str) -> None:
    """Fetch live CDMs from Space-Track and run predictions."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_risk_badge, print_error, print_info

    print_header("Live CDM Prediction")

    try:
        from orbital_sentinel.ingestion.cdm_api import SpaceTrackClient
        from orbital_sentinel.ingestion.cdm_mapper import map_cdm_to_esa_format
        from orbital_sentinel.inference.pipeline import load_inference_artifacts, predict_single_event

        artifacts = load_inference_artifacts(model_name=model)
        if artifacts["model"] is None:
            print_error(f"Model '{model}' not found")
            return

        print_info("Connecting to Space-Track.org...")
        client = SpaceTrackClient()
        client.login()
        raw_cdms = client.fetch_cdm(limit=count)
        client.close()

        print_metric("CDMs fetched", str(len(raw_cdms)))
        click.echo()

        for i, cdm in enumerate(raw_cdms):
            mapped = map_cdm_to_esa_format(cdm)
            sat1 = cdm.get("SAT_1_NAME", "Object 1")
            sat2 = cdm.get("SAT_2_NAME", "Object 2")

            result = predict_single_event(
                mapped, artifacts["model"], artifacts["scaler"], artifacts["feature_names"],
                include_physics=True, include_explanation=False,
            )

            click.echo(click.style(f"  CDM #{i+1}: {sat1} vs {sat2}", bold=True))
            click.echo(f"    Risk: {result['prediction']:.2f}  {print_risk_badge(result['risk_category'])}  "
                        f"Confidence: {result.get('confidence', 0):.2f}  "
                        f"Miss: {mapped.get('miss_distance', 0):,.0f} m")
            click.echo()

    except Exception as e:
        print_error(str(e))
