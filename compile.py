from pyexpat import model
import numpy as np
import torch
import time
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, to_numpy
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
from model import Model

from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from pytorch_nndct import QatProcessor

limit = 1000
vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
device = torch.device("cpu")
randinput = torch.from_numpy(np.random.randn(1, 64, limit, 1).astype(np.float32))
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
    print(processed_signal.size())
    if a_sig_length_e1/16000 < 8:
      continue
    # Inference and accumulate time. Input shape: [Batch_size, 64, Timesteps]
    t_997 = model(processed_signal.unsqueeze(-1))
    # t_997 = model(processed_signal)
    probs = torch.softmax(t_997, **{'dim': 2})
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
  return 1-wer

def calibration(quant_model, data, maxNum):
  num = 0
  data_layer = AudioToTextDataLayer(
      manifest_filepath=data,
      sample_rate=16000,
      labels=vocab,
      batch_size=1,
      shuffle=False,
      drop_last=True)
  preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000) 
  predictions = []
  transcripts = []
  transcripts_len = []
  for i, test_batch in enumerate(data_layer.data_iterator):
    # Get audio [1, n], audio length n, transcript and transcript length
    audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch
    if a_sig_length_e1/16000 < 8:
      continue
    if num>maxNum:
      return quant_model
    else:
      num = num+1
    # Get 64d MFCC features and accumulate time
    processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)
    processed_signal = processed_signal[:,:,:limit]
    quant_model(processed_signal.unsqueeze(-1))
  return quant_model

if __name__ == '__main__':
    quant_mode = 'test'
    data = 'sample_vai.json'
    torch_model = Model()
    torch_model.eval()
    print(torch_model)
    t = time.time()
    torch_outputs = evaluate(torch_model, data)

    # calib_data = torch.load('calib.pt')[:,:,:limit].unsqueeze(-1)
    # print('Calib size{}'.format(calib_data.size()))
    input_data = torch.Tensor(1, 64, limit).unsqueeze(-1)
    # scripted_model = torch.jit.trace(torch_model, calib_data).eval()
    quantizer = torch_quantizer('calib', torch_model, (input_data), device=device)
    quant_model = quantizer.quant_model

    #quant_model = calibration(quant_model, data, 200)
    '''Calibration'''
    num = 0
    data_layer = AudioToTextDataLayer(
        manifest_filepath=data,
        sample_rate=16000,
        labels=vocab,
        batch_size=1,
        shuffle=False,
        drop_last=True)
    preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000) 
    predictions = []
    transcripts = []
    transcripts_len = []
    for i, test_batch in enumerate(data_layer.data_iterator):
      # Get audio [1, n], audio length n, transcript and transcript length
      audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch
      if a_sig_length_e1/16000 < 8:
        continue
      if num>200:
        break
      else:
        num = num+1
      # Get 64d MFCC features and accumulate time
      processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)
      processed_signal = processed_signal[:,:,:limit]
      quant_model(processed_signal.unsqueeze(-1))
    quantizer.export_quant_config()
    
    '''Compile'''
    quantizer = torch_quantizer('test', torch_model, (input_data), device=device)
    calib = 'val/dev_other.json'
    compile_model = quantizer.quant_model
    compile_model = calibration(compile_model, data, 200)
    quantizer.export_xmodel(deploy_check=False)

# vai_c_xir -x ./quantize_result/Model_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -n quartznet

# [UNILOG][FATAL][XCOM_DATA_OUTRANGE][Data value is out of range!]