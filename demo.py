import torch
import numpy as np

captions = torch.randint(0,100, (5,100))
lengths = [5,10,15,20,25]
targets = torch.zeros(len(captions), max(lengths)).long()

for i, cap in enumerate(captions):
    end = lengths[i]
    targets[i, :end] = cap[:end]