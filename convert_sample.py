import numpy as np
import torch
import time
from utils.common import post_process_predictions, post_process_transcripts, word_error_rate, to_numpy
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from utils.data_layer import AudioToTextDataLayer
vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
path = 'sample.json'
num = 0
data_layer = AudioToTextDataLayer(
    manifest_filepath=path,
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

torch.save(processed_signal, 'calib.pt')