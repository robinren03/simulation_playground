import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import time
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def simulate_allreduce_async(data):
    # 模拟一个简单的allreduce操作
    dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=True)
    return False, 0

def compute_kernel(data):
    """A simple computation kernel that performs element-wise operations on the data."""
    for _ in range(40):  # Simulate heavy computation
        # data = data * torch.sin(data) + torch.cos(data)
        data = data * data.T
        data = data / torch.norm(data, dim=1, keepdim=True)
    return False, 0

def mix_operations(data, data1, delay_ms):
    
    # 在默认流中启动模拟allreduce操作
    comm_stream = torch.cuda.Stream()
    with torch.cuda.stream(comm_stream):
        simulate_allreduce_async(data)
    
    st_time = torch.cuda.Event(enable_timing=True)
    en_time = torch.cuda.Event(enable_timing=True)

    # 创造一个新的流进行计算，并在该流中立即引入延迟
    compute_stream = torch.cuda.Stream()
    with torch.cuda.stream(compute_stream):
        # 引入b毫秒的延迟（b * 10^6 纳秒）
        if (delay_ms > 10): torch.cuda._sleep(int(delay_ms * 1e6))
        st_time.record()
        # 在延迟后启动计算任务
        compute_kernel(data1)
        en_time.record()
    
    # 等待计算流完成所有操作
    compute_stream.synchronize()
    pp = st_time.elapsed_time(en_time)
    return True, pp



def measure_time(func, *args, **kwargs):
    """Utility function to measure execution time of a function."""
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_time.record()
    time = 0
    for _ in range(10):
        ex, t = func(*args, **kwargs)
        dist.barrier()
        torch.cuda.synchronize()
        if (ex): time += t

    end_time.record()

    if (time == 0): elapsed_time_ms = start_time.elapsed_time(end_time) / 10
    else: elapsed_time_ms = time / 10
    return elapsed_time_ms

def example(rank, world_size, data_size):
    setup(rank, world_size)

    # 准备数据
    data = torch.ones((int(data_size**0.5), int(data_size**0.5)), device=rank)
    data1 = torch.ones((int(data_size**0.5), int(data_size**0.5)), device=rank)
    
    # Measure allreduce time
    allreduce_time = measure_time(simulate_allreduce_async, data)
    # shared_a[rank] = float(allreduce_time)
    # print(rank, "GOOD 2!")
    # allreduce_time = max(shared_a)
    # print(rank, "GOOD 3!")

    # Measure computation time without overlap
    compute_time = measure_time(compute_kernel, data1)
    # shared_a[rank] = float(compute_time)
    # mp.synchoronize()
    # compute_time = max(shared_a)

    
    print(f"Rank:{rank}, Allreduce time:{allreduce_time}, Compute time:{compute_time}")
    overlaps = np.linspace(0, 1, 5)
    compute_times_normalized = []

    # Experiment with overlapping
    for overlap in overlaps:
        delay = allreduce_time * (1 - overlap)
        d_time = measure_time(mix_operations, data, data1, delay)
        if (rank == 0): compute_times_normalized.append(d_time / compute_time)
        print(f"Rank:{rank}, Overlap rate:{overlap}, D_time: {d_time}")
    
    if (rank == 0): print(compute_times_normalized)
    cleanup()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation and CUDA setup.")

    world_size = torch.cuda.device_count()  # 根据GPU数量设置世界大小
    # shared_a = mp.Array('f', world_size)
    mp.spawn(example, args=(world_size, 1024*1024*64), nprocs=world_size, join=True)