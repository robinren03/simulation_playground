import torch.multiprocessing as mp

def worker(rank, barrier):
    # 示例：每个进程根据其 rank 设置数组中的浮点值
    # shared_array[rank] = rank + 0.5

    # 等待所有进程都执行到这一步
    barrier.wait()

    # 仅在 rank 0 的进程中打印整个数组
    if rank == 0:
        # all_values = [shared_array[i] for i in range(len(shared_array))]
        print(f"Shared array values")

def main(world_size=4):
    # 创建一个共享的浮点数组 ('f' 代表 float)
    shared_array = mp.Array('f', world_size)

    # 创建 Barrier 对象
    barrier = mp.Barrier(world_size)
    
    # 启动多个进程
    mp.spawn(worker,
             args=barrier,
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()