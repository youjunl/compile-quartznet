import numpy as np
import torch
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, to_numpy
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
torch.set_printoptions(8)
from model import Model
vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
@torch.no_grad()
def evaluate(model, val_data):
  model = model.to(device)
  data_layer = AudioToTextDataLayer(
      manifest_filepath=val_data,
      sample_rate=16000,
      labels=vocab,
      batch_size=32,
      shuffle=False,
      drop_last=True)
  preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000) 
  predictions = []
  transcripts = []
  transcripts_len = []
  for i, test_batch in enumerate(data_layer.data_iterator):
      # Get audio [1, n], audio length n, transcript and transcript length
      audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch

      # Get 64d MFCC features and accumulate time
      processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)

      # Inference and accumulate time. Input shape: [Batch_size, 64, Timesteps]
      prob = model(processed_signal)
      ologits = torch.log(prob)
      alogits = np.asarray(ologits)
      logits = torch.from_numpy(alogits[0])
      predictions_e1 = logits.argmax(dim=-1, keepdim=False)
      transcript_e1 = torch.from_numpy(np.asarray(test_batch[2])) 
      transcript_len_e1 = torch.from_numpy(np.asarray(test_batch[1])) 

      # Save results
      predictions.append(predictions_e1)
      transcripts.append(transcript_e1)
      transcripts_len.append(transcript_len_e1)
  acc = accuracy(predictions, transcripts, transcripts_len)
  return acc, 1 - acc

@torch.no_grad()
def accuracy(predictions, transcripts, transcripts_len):
  """Computes word error rate"""
  # Map characters
  greedy_hypotheses = post_process_predictions(predictions, vocab)
  references = post_process_transcripts(transcripts, transcripts_len, vocab)
  # Caculate word error rate and time cost
  wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
  return 1 - wer

print("Loading torch model")
model = Model()
model = model.eval()

input_shape = [1, 64, 256]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

#Load a data
data = 'sample.json'
acc, wer = evaluate(scripted_model, data)

print('wer: %2f'%wer)
import tvm
from tvm import relay

#Load a test sample