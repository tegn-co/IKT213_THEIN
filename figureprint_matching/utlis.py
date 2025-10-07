import time
import psutil

def measure_performance(func, *args, **kwargs):

    process = psutil.Process()
    cpu_before = psutil.cpu_percent(interval=None)
    memo_before = process.memory_info().rss / (1024 ** 2)
    start_time = time.perf_counter()

    #runs the function and gathering results
    result = func(*args, **kwargs)

    end_time = time.perf_counter()
    cpu_after = psutil.cpu_percent(interval=None)
    memo_after = process.memory_info().rss / (1024 ** 2)

    metrics = {
        "time_ms": (end_time - start_time) * 1000,
        "cpu_percent": cpu_after - cpu_before,
        "ram_mb": memo_after - memo_before
    }

    return result, metrics
