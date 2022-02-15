import numpy as np
#import onnxruntime
import torch
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, to_numpy
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
from model import Model
vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
device = torch.device("cpu")
randinput = torch.from_numpy(np.random.randn(1, 64, 256).astype(np.float32))
@torch.no_grad()
def test(model, val_data):
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
  for i, test_batch in enumerate(data_layer.data_iterator):
    # Get audio [1, n], audio length n, transcript and transcript length
    audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch

    # Get 64d MFCC features and accumulate time
    processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)
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
  greedy_hypotheses = post_process_predictions(predictions, vocab)
  return greedy_hypotheses

def ref(model_path, val_data):
  session = onnxruntime.InferenceSession(model_path)
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
  for i, test_batch in enumerate(data_layer.data_iterator):
    # Get audio [1, n], audio length n, transcript and transcript length
    audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = test_batch

    # Get 64d MFCC features and accumulate time
    processed_signal = preprocessor.get_features(audio_signal_e1, a_sig_length_e1)

    # Inference and accumulate time. Input shape: [Batch_size, 64, Timesteps]
    inputs = {session.get_inputs()[0].name: to_numpy(processed_signal),}
    ologits = session.run(None, inputs)
    alogits = np.asarray(ologits)
    logits = torch.from_numpy(alogits[0])
    predictions_e1 = logits.argmax(dim=-1, keepdim=False)
    transcript_e1 = torch.from_numpy(np.asarray(test_batch[2])) 
    transcript_len_e1 = torch.from_numpy(np.asarray(test_batch[1])) 
    
    # Save results
    predictions.append(predictions_e1)
    transcripts.append(transcript_e1)
    transcripts_len.append(transcript_len_e1)
  greedy_hypotheses = post_process_predictions(predictions, vocab)
  return greedy_hypotheses

if __name__ == '__main__':
  data = 'sample.json'
  torch_model = Model()
  torch_outputs = test(torch_model, data)

  # onnx_model = "../Adaptiv/Quartznet/onnx_quartznet.onnx"
  # ort_outputs = ref(onnx_model, data)
  print(torch_model(torch.from_numpy(np.random.randn(1, 64, 256).astype(np.float32))))
  print("torch")
  print(torch_outputs)
  # print("onnx")
  # print(ort_outputs)

  # print(
  #     "Comparison result:",
  #     np.allclose(torch_outputs[0],
  #                 ort_outputs[0],
  #                 atol=1e-4,
  #                 rtol=1e-4))

  # print(post_process_predictions(torch.Tensor(torch_outputs[0]), vocab))