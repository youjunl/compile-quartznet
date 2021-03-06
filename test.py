import numpy as np
#import onnxruntime
import torch
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, to_numpy, ctc_decoder
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
from model import Model
from utils.losses import CTCLossNM
vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
device = torch.device("cpu")
randinput = torch.from_numpy(np.random.randn(1, 64, 256).astype(np.float32))

if __name__ == '__main__':
  # data = 'sample_vai.json'
  predictions = []
  torch_model = Model()
  torch_model.eval()
  calib_data = torch.load('calib.pt')
  
  calib_data = calib_data*1
  calib_data = calib_data.to(torch.int32).to(torch.float32)
  print("calib data")
  print(calib_data)
  torch_outputs = torch_model(calib_data.unsqueeze(-1))
  probs = torch.softmax(torch_outputs, **{'dim': 2})
  logits = torch.log(probs)[0]
  prediction = logits.argmax(dim=-1, keepdim=False)
  predictions.append(prediction.tolist())
  print("model output")
  print(torch_outputs)
  greedy_hypotheses = ctc_decoder(prediction, vocab) 
  print(greedy_hypotheses)
  # test(torch_model, './val/dev_other.json')
  # print(torch_outputs)
  # onnx_model = "../Adaptiv/Quartznet/onnx_quartznet.onnx"
  # ort_outputs = ref(onnx_model, data)
  # print("onnx")
  # print(ort_outputs)

  # print(
  #     "Comparison result:",
  #     np.allclose(torch_outputs[0],
  #                 ort_outputs[0],
  #                 atol=1e-4,
  #                 rtol=1e-4))

  # print(post_process_predictions(torch.Tensor(torch_outputs[0]), vocab))