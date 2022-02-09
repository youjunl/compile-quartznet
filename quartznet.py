# Adjusted for Vitis-AI-Quantizer

import glob
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, to_numpy
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
from model import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="val/dev_other.json",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="quartznet.pth",
    help='Trained model file path.'
)
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
args, _ = parser.parse_known_args()

vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

def accuracy(predictions, transcripts, transcripts_len):
  """Computes word error rate"""
  # Map characters
  greedy_hypotheses = post_process_predictions(predictions, vocab)
  references = post_process_transcripts(transcripts, transcripts_len, vocab)
  print(greedy_hypotheses)
  print('ref')
  print(references)
  # Caculate word error rate and time cost
  wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
  return 1 - wer


def evaluate(model, val_data):
  print('Evaluation begin')
  model.eval()
  print('Load data')
  data_layer = AudioToTextDataLayer(
      manifest_filepath=val_data,
      sample_rate=16000,
      labels=vocab,
      batch_size=1,
      shuffle=False,
      drop_last=True)
  print('Load preprocessor')
  preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000) 
  predictions = []
  transcripts = []
  transcripts_len = []
  for i, test_batch in enumerate(data_layer.data_iterator):
      # Get audio [1, n], audio length n, transcript and transcript length
      audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch
      print('audio_signal')
      print(audio_signal_e1)
      # Get 64d MFCC features and accumulate time
      processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)
      print('features')
      print(processed_signal)
      # Inference and accumulate time. Input shape: [Batch_size, 64, Timesteps]
      ologits = model(processed_signal)
      alogits = np.asarray(ologits)
      logits = torch.from_numpy(alogits[0])
      predictions_e1 = logits.argmax(dim=-1, keepdim=False)
      transcript_e1 = torch.from_numpy(np.asarray(test_batch[2])) 
      transcript_len_e1 = torch.from_numpy(np.asarray(test_batch[1])) 

      # Save results
      predictions.append(torch.reshape(predictions_e1, (1, -1)))
      transcripts.append(transcript_e1)
      transcripts_len.append(transcript_len_e1)
  acc = accuracy(predictions, transcripts, transcripts_len)
  return acc, 1 - acc

@torch.no_grad()
def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 

  data_dir = args.data_dir
  quant_mode = args.quant_mode
  finetune = args.fast_finetune
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  model = Model()
  #model.load_state_dict(torch.load(file_path))

  input = torch.randn([batch_size, 64, 256])
  if quant_mode == 'float':
    quant_model = model
  else:
    ## new api
    ####################################################################################
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device)

    quant_model = quantizer.quant_model
    #####################################################################################

  # fast finetune model or load finetuned parameter before test
  if finetune == True:

      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, data_dir))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
   
  # record  modules float model accuracy
  # add modules float model accuracy here

  #register_modification_hooks(model_gen, train=False)
  acc, wer = evaluate(quant_model, data_dir)

  # logging accuracy
  print('wer: %g' % (wer))

  # handle quantization result
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if deploy:
    quantizer.export_xmodel(deploy_check=False)

if __name__ == '__main__':
  model = Model()
  input = torch.randn([args.batch_size, 64, 4000])
  scripted_model = torch.jit.trace(model, input).eval()
  quant_mode = args.quant_mode
  deploy = args.deploy
  quantizer = torch_quantizer(scripted_model, model, (input))
  quant_model = quantizer.quant_model
  acc, wer = evaluate(model, args.data_dir)
  print('wer:', wer)
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if deploy:
    quantizer.export_xmodel()
