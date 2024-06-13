import os
import time
import argparse
import threading
import torch
import torch.distributed
from torch import nn
from torch.distributed.distributed_c10d import get_world_size
from torch.nn.modules import module
from torch.profiler import profile, ProfilerActivity
torch_profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    with_modules=True,
)

torch.distributed.init_process_group('nccl')
rank = torch.distributed.get_rank()
num_ranks = torch.distributed.get_world_size()
simpleimc.torch.ops.init(rank, num_ranks, -1, 0)
device = torch.device('cuda', rank)
print(f"rank {rank} reporting")
dtype=torch.float
dim = 12288
batch_size = 4
seq_len = 2048
micro_iter = 8
iter = 20
# comm_type = "allreduce"
# comm_type = "reduce_scatter"
comm_type = "allgather"

def get_input() -> torch.Tensor:
    return torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)

class PureMatMut(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=True, dtype=dtype)
    def forward(self, x):
        torch.distributed.barrier()
        for i in range(micro_iter):
            x = self.fc(x)
        return x

class PureNccl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = torch.randn(num_ranks, batch_size, seq_len, dim, device=device, dtype=dtype)
    def forward(self, x):
        assert isinstance(x, list)
        if comm_type in ["allreduce"]:
            x = torch.concat(x)
        torch.distributed.barrier()
        for i in range(micro_iter):
            if comm_type == "allreduce":
                torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.AVG)
            elif comm_type == "allgather":
                torch.distributed._all_gather_base(self.buffer, x[rank], async_op=True)
            else:
                torch.distributed._reduce_scatter_base(x[rank], self.buffer, async_op=True)

class MatMulNccl(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=True, dtype=dtype)
        self.buffer = torch.randn(num_ranks, batch_size, seq_len, dim, device=device, dtype=dtype)
    def forward(self, x, y):
        assert isinstance(y, list)
        if comm_type in ["allreduce"]:
            y = torch.concat(y)
        torch.distributed.barrier()
        for i in range(micro_iter):
            if comm_type == "allreduce":
                handle = torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.AVG, async_op=True)
            elif comm_type == "allgather":
                handle = torch.distributed._all_gather_base(self.buffer, y[rank], async_op=True)
            else:
                handle = torch.distributed._reduce_scatter_base(y[rank], self.buffer, async_op=True)
            x = self.fc(x)
            handle.wait()


def main():
    if rank == 0:
        nsys_filename = f"/opt/tiger/tmp.nsys_trace_{comm_type}_{int(time.time())}.qdrep"
        time.sleep(10)
        # torch_profiler.start()
    for i in range(iter):
        if rank == 0:
            print(f"iter {i}")
            if i == 10:
                os.system(f"nsys start -o {nsys_filename}")
            elif i == 13:
                os.system("nsys stop")
        pure_matmul = PureMatMut(dim).to(device)
        pure_nccl = PureNccl().to(device)
        matmul_nccl = MatMulNccl(dim).to(device)
        x = get_input()
        y = [get_input() for i in range(num_ranks)]
        # pure_matmul(x)
        # pure_nccl(y)
        matmul_nccl(x, y)
        
    # if rank == 0:
        # torch_profiler.stop()
        # filename = f"/opt/tiger/tmp.torch_trace_{time.time()}.json"
        # print(f"==== begin to dump {filename} ====")
        # torch_profiler.export_chrome_trace(filename)
        # print(f"==== succesfully dumped {filename} ====")


main()
