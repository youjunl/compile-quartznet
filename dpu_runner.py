import numpy as np
import torch
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, ctc_decoder
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
# from model import Model
from ctypes import *
from typing import List

print('Numpy: %s \nPytorch: %s'%(np.__version__, torch.__version__))
import vitis_ai_library
import vart
import xir
# import threading
# import math
import time
print('Finish import')
limit = 800
vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

if __name__ == '__main__':
  print('******************************************************************')
  calib_data = torch.load('calib.pt')
  print("calib data")
  print(calib_data)
  calib_data = calib_data[:,:,:limit].unsqueeze(-1)
  inputData = calib_data.detach().cpu().numpy()
  # ''' get a list of subgraphs from the compiled model file '''
  g = xir.Graph.deserialize('/home/petalinux/notebooks/compile-quartznet/quartznet.xmodel')
  # create the runner
  runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
  predictions = []
  # get input and output tensor buffer, fill input
  input_tensor_buffers = inputData
  output_tensor_buffers = runner.get_outputs()
  time_start = time.time()
  v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
  runner.wait(v)
  print(type(output_tensor_buffers))
  output_data = np.asarray(output_tensor_buffers[0], dtype=np.int8, order="C").astype(np.float32)
  probs = torch.softmax(torch.Tensor(output_data), **{'dim': 2})
  logits = torch.log(probs)[0]
  prediction = logits.argmax(dim=-1, keepdim=False)
  predictions.append(prediction)
  greedy_hypotheses = ctc_decoder(prediction, vocab) # [400, n]
  
  print("model output")
  print(output_data)
  print(type(output_data))
  print(greedy_hypotheses)
  time_end = time.time()
  timetotal = time_end - time_start
  print("DPU Time cost: %ds" % timetotal)