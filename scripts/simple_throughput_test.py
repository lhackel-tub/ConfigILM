from time import time

import torch

from configilm.ConfigILM import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.DataSets import ThroughputTest_DataSet


model_config = ILMConfiguration(
    timm_model_name="resnet18",
    hf_model_name="prajjwal1/bert-tiny",
    image_size=256,
    network_type=ILMType.VQA_CLASSIFICATION,
    max_sequence_length=32,
)

model = ConfigILM(model_config)
ds = ThroughputTest_DataSet.VQAThroughputTestDataset(
    data_dirs={},  # parameter is ignored but required for compatibility with other DataSets in ConfigILM
    seq_length=32,
    num_samples=100,
    split="train",
)

start = time()
for i in range(len(ds)):
    v, q, a = ds[i]
    v_batch = v.unsqueeze(0)
    q_batch = torch.Tensor(q).int().unsqueeze(0)
    model((v_batch, q_batch))

end = time()
print(f"Time taken: {end - start:.2f} seconds")
print(f"Throughput: {1000 / (end - start):.2f} samples per second")
