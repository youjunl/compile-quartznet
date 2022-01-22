import numpy as np
import torch
torch.set_printoptions(8)
from model import Model

print("Loading torch model")
model = Model()
torch.save(model, "model.pth")