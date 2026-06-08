# utils/benchmark.py
import torch, time
from torch.amp.autocast_mode import autocast

def measure_inference(model, device, input_shape=(1,3,32,32), n_runs=50, warmup=10):
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    try:
        dummy = dummy.contiguous(memory_format=torch.channels_last)
    except Exception:
        pass

    with torch.no_grad():
        # warmup under autocast also
        for _ in range(warmup):
            with autocast("cuda"):
                _ = model(dummy)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n_runs):
            with autocast("cuda"):
                _ = model(dummy)
        t1.record()
        torch.cuda.synchronize()
        elapsed_ms = t0.elapsed_time(t1)
        avg_ms = elapsed_ms / n_runs
        throughput = (input_shape[0] / (avg_ms/1000.0))
    return avg_ms, throughput