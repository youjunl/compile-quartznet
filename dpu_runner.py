import numpy as np
import torch
from zmq import device
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
device = torch.device("cpu")
def preprocess_frame(input, fixpos):
  fix_scale = 2**fixpos
  output = fix_scale * input
  return output

if __name__ == '__main__':
  print('******************************************************************')
  calib_data = torch.load('calib.pt').to(device)
  print("calib data")
  print(calib_data)
  calib_data = calib_data[:,:,:limit]
  calib_data = torch.movedim(calib_data, 0, 1)
  calib_data = torch.transpose(calib_data, 0, 2)
  inputData = calib_data.detach().cpu().numpy()
  print(inputData.shape)
  # ''' get a list of subgraphs from the compiled model file '''
  g = xir.Graph.deserialize('quartznet.xmodel')
  # create the runner
  runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
  predictions = []
  # get input and output tensor buffer, fill input
  input_tensor_buffers = runner.get_inputs()
  output_tensor_buffers = runner.get_outputs()
  input_ndim = tuple(input_tensor_buffers[0].get_tensor().dims)
  batch = input_ndim[0]
  width = input_ndim[1]
  height = input_ndim[2]
  input_fixpos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")

  inputData = preprocess_frame(inputData, input_fixpos)

  input_Data = np.asarray(input_tensor_buffers[0])
  print(inputData)
  input_Data[0] = inputData
  print(input_Data)

  print('Data format:{} {}'.format(type(input_tensor_buffers), input_tensor_buffers))
  time_start = time.time()
  v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
  runner.wait(v)

  pre_output_size = int(output_tensor_buffers[0].get_tensor().get_element_num() / batch)
  output_data = np.asarray(output_tensor_buffers[0])
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