from pyexpat import model
import numpy as np
import torch
import time
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, to_numpy
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
from utils.losses import CTCLossNM
from model import Model

from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from pytorch_nndct import QatProcessor
from tqdm import tqdm

limit = 800
vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
device = torch.device("cpu")
randinput = torch.from_numpy(np.random.randn(1, 64, limit, 1).astype(np.float32))

def calibration(quantizer, data, maxNum, quant_mode):
  quant_model = quantizer.quant_model
  num = 0
  data_layer = AudioToTextDataLayer(
      manifest_filepath=data,
      sample_rate=16000,
      labels=vocab,
      batch_size=1,
      shuffle=False,
      drop_last=True,
      num_workers=6)
  preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000) 
  # print("Calibration...")
  # for i, test_batch in tqdm(enumerate(data_layer.data_iterator)):
  #   # Get audio [1, n], audio length n, transcript and transcript length
  #   audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch
  #   if a_sig_length_e1/16000 < 8:
  #     continue
  #   if i>maxNum:
  #     break

  #   # Get 64d MFCC features and accumulate time
  #   processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)
  #   processed_signal = processed_signal[:,:,:limit]
  #   t_997 = quant_model(processed_signal.unsqueeze(-1))
  loss_gen = evaluate(quant_model, data)
  #finetune
  if quant_mode == 'calib':
    print("Fast-fintune...")
    quantizer.fast_finetune(evaluate, (quant_model, data))
    quantizer.export_quant_config()
  elif quant_mode == 'test':
    print("Evaluation...")
    quantizer.load_ft_param()
    loss_gen = evaluate(quant_model, data)
    print('loss: %g' % (loss_gen))
    quantizer.export_xmodel(deploy_check=False)

def evaluate(model, val_data):
  model.eval()
  data_layer = AudioToTextDataLayer(
      manifest_filepath=val_data,
      sample_rate=16000,
      labels=vocab,
      batch_size=1,
      shuffle=False,
      drop_last=True)
  preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000) 
  predictions = []
  transcripts = []
  transcripts_len = []
  Loss = 0
  total = 0
  loss_fn = CTCLossNM(num_classes=len(vocab))
  for i, test_batch in tqdm(enumerate(data_layer.data_iterator)):
    # Get audio [1, n], audio length n, transcript and transcript length
    audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch
    if a_sig_length_e1/16000 < 8:
      continue
    # Get 64d MFCC features and accumulate time
    processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)
    processed_signal = processed_signal[:,:,:limit]
    t_997 = model(processed_signal.unsqueeze(-1))
    # t_997 = model(processed_signal)
    probs = torch.softmax(t_997, **{'dim': 2})
    ologits = torch.log(probs)
    logits = ologits[0]
    predictions_e1 = logits.argmax(dim=-1, keepdim=False)
    transcript_e1 = torch.from_numpy(np.asarray(test_batch[2])) 
    transcript_len_e1 = torch.from_numpy(np.asarray(test_batch[1])) 
    #loss_iter = loss_fn(logits, transcript_e1, a_sig_length_e1, transcript_len_e1)
    #Loss += loss_iter
    total += 1
    # Save results
    predictions.append(torch.reshape(predictions_e1, (1, -1)))
    transcripts.append(transcript_e1)
    transcripts_len.append(transcript_len_e1)
    
  return Loss

if __name__ == '__main__':
    torch_model = Model()
    quantizer = torch_quantizer('calib', torch_model, (randinput), device=device)
    calib = 'val/dev_other.json'
    calibration(quantizer, calib, 200, 'calib')    
    #calib
    quantizer = torch_quantizer('test', torch_model, (randinput), device=device)
    calib = 'val/dev_other.json'
    compile_model = quantizer.quant_model
    compile_model = calibration(compile_model, calib, 200, 'test')

# vai_c_xir -x ./quantize_result/Model_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -n quartznet

# [UNILOG][FATAL][XCOM_DATA_OUTRANGE][Data value is out of range!]