# from onnx_pytorch import code_gen
path = "../Adaptiv/Quartznet/onnx_quartznet.onnx"
# code_gen.gen(path, "./")

import numpy as np
import onnx
import onnxruntime
import torch
torch.set_printoptions(8)

from model import Model

model = Model()
model.eval()
inp = np.random.randn(32, 64, 256).astype(np.float32)
with torch.no_grad():
  torch_outputs = model(torch.from_numpy(inp))

onnx_model = onnx.load(path)
sess_options = onnxruntime.SessionOptions()
session = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                                       sess_options)
inputs = {session.get_inputs()[0].name: inp}
ort_outputs = session.run(None, inputs)
print("torch")
print(torch_outputs.detach().numpy())
print("onnx")
print(ort_outputs)
print(
    "Comparison result:",
    np.allclose(torch_outputs.detach().numpy(),
                ort_outputs[0],
                atol=1e-4,
                rtol=1e-4))