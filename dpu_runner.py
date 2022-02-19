import numpy as np
#import onnxruntime
import torch
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, to_numpy
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
from model import Model
from ctypes import *
from typing import List

print('Numpy: %s \nPytorch: %s'%(np.__version__, torch.__version__))

import vart
import xir
import threading
import math
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

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
  assert graph is not None, "'graph' should not be None."
  root_subgraph = graph.get_root_subgraph()
  assert (
      root_subgraph is not None
  ), "Failed to get root subgraph of input Graph object."
  if root_subgraph.is_leaf:
      return []
  child_subgraphs = root_subgraph.toposort_child_subgraph()
  assert child_subgraphs is not None and len(child_subgraphs) > 0
  return [
      cs
      for cs in child_subgraphs
      if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
  ]

def execute_async(dpu, tensor_buffers_dict):
    input_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()
    ]
    output_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()
    ]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)

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
  # print('******************************************************************')
  calib_data = torch.load('calib.pt')
  calib_data = calib_data[:,:,:limit].unsqueeze(-1)
  inputData = calib_data.detach().cpu().numpy()
  ''' get a list of subgraphs from the compiled model file '''
  g = xir.Graph.deserialize('/home/petalinux/notebooks/compile-quartznet/quartznet.xmodel')
  subgraphs = g.get_root_subgraph().toposort_child_subgraph()
  
  ''' get a list of dpu subgraphs from the compiled model file '''
  dpu_subgraphs = []
  cpu_subgraphs = []
  for subgraph in subgraphs:
    tempDpu = vart.Runner.create_runner(subgraphs, "run")
    inputTensors = tempDpu.get_input_tensors()
    outputTensors = tempDpu.get_output_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    shapeOut = tuple(outputTensors[0].dims)
    
    outputData = [np.empty(shapeOut, dtype=np.int8, order="C")]
    pre_output_size = int(outputTensors[0].get_data_size() / shapeIn[0])
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    job_id = tempDpu.execute_async(inputData, outputData)
    tempDpu.wait(job_id)
    inputData = outputData
    print('input {}'.format(shapeIn))
    print('output {}'.format(shapeOut))
    if subgraph.has_attr("device") and subgraph.get_attr("device").upper() == "DPU":
      dpu_subgraphs.append(subgraph)
      print("DPU")
    else:
      cpu_subgraphs.append(subgraph)
      print("CPU")
  print('Total number of DPU subgraph: {}, DPU: {}, CPU: {}.'.format(len(subgraphs), len(dpu_subgraphs), len(cpu_subgraphs)))
  time_start = time.time()  
  run_quartznet()
  time_end = time.time()
  timetotal = time_end - time_start
  print("DPU Time cost: %ds" % timetotal)