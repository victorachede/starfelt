from starfelt.core.benchmark import BenchmarkConfig, BenchmarkReport, BenchmarkResult


def test_benchmark_config_rejects_invalid_values():
    try:
        BenchmarkConfig(steps=0).validate()
    except ValueError as exc:
        assert "steps" in str(exc)
    else:
        raise AssertionError("expected invalid benchmark config to fail")


def test_benchmark_report_uses_median_and_serializes():
    config = BenchmarkConfig(repeats=2)
    report = BenchmarkReport(
        config=config,
        results=[
            BenchmarkResult(
                repeat=1,
                device="cpu",
                steps=10,
                batch_size=4,
                elapsed_s=2.0,
                steps_per_s=5.0,
                samples_per_s=20.0,
                final_loss=1.0,
                cost_usd=0.2,
            ),
            BenchmarkResult(
                repeat=2,
                device="cpu",
                steps=10,
                batch_size=4,
                elapsed_s=1.0,
                steps_per_s=10.0,
                samples_per_s=40.0,
                final_loss=0.9,
                cost_usd=0.1,
            ),
        ],
    )

    assert report.median_elapsed_s == 1.5
    assert report.median_samples_per_s == 30.0
    payload = report.to_dict()
    assert payload["workload"] == "synthetic_mlp_v1"
    assert len(payload["results"]) == 2
