import torch

from kd_sensing.evaluation.complexity import benchmark_forward, parameter_summary


def test_complexity_helpers_report_exact_parameters_and_cpu_timing() -> None:
    model = torch.nn.Linear(3, 2)
    values = torch.ones(4, 3)

    assert parameter_summary(model) == {"parameters_total": 8, "parameters_trainable": 8}
    timing = benchmark_forward(lambda: model(values), device=torch.device("cpu"), batch_size=4, warmup=1, repeats=2)
    assert timing["latency_ms_mean"] > 0
    assert timing["throughput_samples_per_second"] > 0
    assert "peak_memory_mib" not in timing
