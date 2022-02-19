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
device = torch.device("cpu")
randinput = torch.from_numpy(np.random.randn(1, 64, 256).astype(np.float32))

@torch.no_grad()
def evaluate(model, val_data):
  model.eval()
  data_layer = AudioToTextDataLayer(
    manifest_filepath=val_data,
    sample_rate=16000,
    labels=vocab,
    batch_size=1,
    shuffle=False,
    drop_last=False)
  preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000) 
  predictions = []
  transcripts = []
  transcripts_len = []
  for i, test_batch in enumerate(data_layer.data_iterator):
      # Get audio [1, n], audio length n, transcript and transcript length
      audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch

      # Get 64d MFCC features and accumulate time
      processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)
      processed_signal = processed_signal[:,:,:limit]
      if a_sig_length_e1/16000 < 8:
        continue
      # Inference and accumulate time. Input shape: [Batch_size, 64, Timesteps]
      t_997 = model(processed_signal.unsqueeze(-1))
      # t_997 = model(processed_signal)
      probs = torch.softmax(t_997, **{'dim': 2})
      print(probs)
      print(probs.size())
      ologits = torch.log(probs)
      alogits = np.asarray(ologits)
      logits = torch.from_numpy(alogits[0])
      predictions_e1 = logits.argmax(dim=-1, keepdim=False)
      transcript_e1 = torch.from_numpy(np.asarray(test_batch[2])) 
      transcript_len_e1 = torch.from_numpy(np.asarray(test_batch[1])) 

      # Save results
      predictions.append(torch.reshape(predictions_e1, (1, -1)))
      transcripts.append(transcript_e1)
      transcripts_len.append(transcript_len_e1)
  greedy_hypotheses = post_process_predictions(predictions, vocab)
  print(greedy_hypotheses)
  references = post_process_transcripts(transcripts, transcripts_len, vocab)
  wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
  return greedy_hypotheses

def run_quartznet(dpu: "Runner", data):
  # Load data
  preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000) 
  predictions = []
  transcripts = []
  transcripts_len = []

  inputTensors = dpu.get_input_tensors()
  outputTensors = dpu.get_output_tensors()
  shapeIn = tuple(inputTensors[0].dims)
  shapeOut = tuple(outputTensors[0].dims)
  pre_output_size = int(outputTensors[0].get_data_size() / shapeIn[0])
  output_fixpos = outputTensors[0].get_attr("fix_point")
  output_scale = 1 / (2**output_fixpos)
  print('input {}'.format(shapeIn))
  print('output {}'.format(shapeOut))

      # Get audio [1, n], audio length n, transcript and transcript length
  audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch

  # Get 64d MFCC features and accumulate time
  processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)
  processed_signal = processed_signal[:,:,:limit]

  # Inference and accumulate time. Input shape: [Batch_size, 64, Timesteps]
  inputData = processed_signal.unsqueeze(-1)
  inputData = inputData.detach().cpu().numpy()
  outputData = [np.empty(shapeOut, dtype=np.int8, order="C")]

  job_id = dpu.execute_async(inputData, outputData)
  dpu.wait(job_id)

  t_997 = torch.Tensor(outputData)
  # t_997 = model(processed_signal)
  probs = torch.softmax(t_997, **{'dim': 2})
  ologits = torch.log(probs)
  alogits = np.asarray(ologits)
  logits = torch.from_numpy(alogits[0])
  predictions_e1 = logits.argmax(dim=-1, keepdim=False)

  # Save results
  predictions.append(torch.reshape(predictions_e1, (1, -1)))
  greedy_hypotheses = post_process_predictions(predictions, vocab)
  print(greedy_hypotheses)
  return greedy_hypotheses  

if __name__ == '__main__':

  

  # global threadnum
  # threadnum = 1
  # threadAll = []
  # data = 'sample.json'  

  # print('Performing torch evaluation')
  # time_start = time.time()  
  # model = Model()
  # model.eval()
  # evaluate(model, data)
  # time_end = time.time()
  # timetotal = time_end - time_start
  # print("TORCH Time cost: %ds" % timetotal)
  print('******************************************************************')
  calib_data = torch.load('calib.pt')
  print("calib data")
  print(calib_data)
  calib_data = calib_data[:,:,:limit].unsqueeze(-1)
  inputData = calib_data.detach().cpu().numpy()
  # ''' get a list of subgraphs from the compiled model file '''
  g = xir.Graph.deserialize('/home/petalinux/notebooks/compile-quartznet/quartznet.xmodel')
  runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
  predictions = []
  input_tensor_buffers = inputData
  output_tensor_buffers = runner.get_outputs()
  time_start = time.time()
  v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
  runner.wait(v)
  output_data = np.asarray(output_tensor_buffers[0]).astype(np.float32)
  probs = torch.softmax(torch.Tensor(output_data), **{'dim': 2})
  logits = torch.log(probs)[0]
  prediction = logits.argmax(dim=-1, keepdim=False)
  predictions.append(prediction)
  greedy_hypotheses = ctc_decoder(prediction, vocab) # [400, n]
  
  print("model output")
  print(output_data)
  print(type(output_data))
  time_end = time.time()
  timetotal = time_end - time_start
  print("DPU Time cost: %ds" % timetotal)