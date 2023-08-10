import os


# https://zhuanlan.zhihu.com/p/432698650
local_rank = int(os.environ.get("LOCAL_RANK", -1))

import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
from torch.optim import AdamW
from tqdm import tqdm


class SimpleModel(nn.Module):
    """ 一个简单的模型 """

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5000, 768)
        self.fc_list = nn.Sequential(*[nn.Linear(768, 768) for _ in range(12)])
        self.clf = nn.Linear(768, 2)

    def forward(self, ipt):
        x = self.emb(ipt)
        x = self.fc_list(x)
        x = torch.mean(x, dim=1, keepdim=False)
        # logits: (bsz,2)
        logits = self.clf(x)
        return logits


def collect_fn(batch):
    """

    :param batch:List[data_set[i]]
    :return:
    """
    return torch.LongTensor([i[0] for i in batch]), torch.LongTensor([i[1] for i in batch])


class SimpleDataSet(Dataset):
    def __init__(self, num_data: int = 100000):
        # 构造一个假数据长度是128
        self.data = [[random.randint(1, 4999) for _ in range(128)] for _ in range(num_data)]
        self.labels = [random.randint(0, 1) for _ in range(num_data)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        return self.data[item], self.labels[item]


# Step1 相关全局变量
use_multi_gpu = False
if use_multi_gpu:
    dist.init_process_group(backend="nccl")

# Step2 设置model
if use_multi_gpu:
    device = torch.device(f"cuda:{local_rank}")
    model = torch.nn.parallel.DistributedDataParallel(SimpleModel().to(device), device_ids=[local_rank],
                                                      output_device=local_rank)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)

# Step3 设置数据
train_dataset = SimpleDataSet(200000)
dev_dataset = SimpleDataSet(1000)
if use_multi_gpu:
    train_sampler = DistributedSampler(train_dataset)
else:
    train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=100, sampler=train_sampler, collate_fn=collect_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=100, shuffle=False, collate_fn=collect_fn)

# Step4 设置优化器
optimizer = AdamW(params=model.parameters())

# Step5 开始训练

for step, (ipt, labels) in enumerate(tqdm(train_dataloader), start=1):
    ipt, labels = ipt.to(device), labels.to(device)
    logits = model(ipt)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    optimizer.zero_grad()
    if step % 100 == 0:
        if use_multi_gpu and local_rank == 0:
            # do evaluate
            pass
        elif not use_multi_gpu:
            # do evaluate
            pass

# Step6 执行训练 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2  multi_gpu_single_machine.py