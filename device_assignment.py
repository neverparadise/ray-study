import ray
import torch
import os

ray.init(num_cpus=4, num_gpus=2)

# a = torch.randn(1000, 1000).to(device='cuda:0')
# b = torch.randn(1000, 1000).to(device='cuda:1')

# print(a)
# print(b)

# os.system('nvidia-smi')


@ray.remote(num_cpus=2, num_gpus=1)
def make_tensor():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = 'cuda:0' #CUDA_VISIBLE_DEVICES는 해당 id의 GPU(들)만을 볼 수 있게 한다는 것으로, 넘겨 받은 GPU(들)을 0번부터로 인식한다.
    print(f"current device: {torch.cuda.current_device()}")
    print(f"count: {torch.cuda.device_count()}")
    tensor = torch.randn(10000, 10000).to(device=device, non_blocking=False)
    print(tensor.shape)
    print(tensor.device)

    return tensor

@ray.remote(num_cpus=2, num_gpus=1)
def make_tensor2():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = 'cuda:0'
    print(f"current device: {torch.cuda.current_device()}")
    print(f"count: {torch.cuda.device_count()}")
    tensor = torch.randn(10000, 10000).to(device=device, non_blocking=False)
    tensor_ref = ray.put(tensor) # 이렇게 하면 object store에 저장
    return tensor_ref


os.system('nvidia-smi')
results_ref = [make_tensor.remote() for i in range(2)]
results = ray.get(results_ref) 
print(results[0].device)
print(results[1].device)

# results_ref = [make_tensor2.remote() for i in range(2)]
# refs = ray.get(results_ref)
# tensors = [ray.get(ref) for ref in refs]


os.system('nvidia-smi')

# print(tensors[0])
# print(tensors[1]) 
# print(tensors[0].device)
# print(tensors[1].device)


os.system('nvidia-smi')