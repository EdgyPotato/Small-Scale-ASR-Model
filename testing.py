import torch
import torchaudio
import pandas as pd
import os
import torch.nn as nn
from torch.nn import functional as F
import argparse

# =============================================================================
# Re-define components to match the training script
# =============================================================================
class Config:
    BEST_MODEL_PATH = "./models/best.pt"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    NUM_CLASSES = 30
    N_FEATS_IN = 32
    N_FEATS_CNN = 32
    N_FEATS_DENSE = 128
    TRANSFORMER_D_MODEL = 128
    TRANSFORMER_NHEAD = 2
    TRANSFORMER_ENCODER_LAYERS = 3
    TRANSFORMER_DECODER_LAYERS = 3
    TRANSFORMER_DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    
    SAMPLE_RATE = 16000
    N_MFCC = 16
    N_FFT = 400
    N_MELS = 81

class TextProcess:
    def __init__(self):
        char_map_str = """
		<PAD> 0
		<SPACE> 1
		a 2
		b 3
		c 4
		d 5
		e 6
		f 7
		g 8
		h 9
		i 10
		j 11
		k 12
		l 13
		m 14
		n 15
		o 16
		p 17
		q 18
		r 19
		s 20
		t 21
		u 22
		v 23
		w 24
		x 25
		y 26
		z 27
        ' 28
        <UNK> 29
		"""
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            ch = ch.strip()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def int_to_text_sequence(self, labels):
        return ''.join([self.index_map.get(i, '?') for i in labels]).replace('<SPACE>', ' ')

class AudioFeatures(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spectrogram = torchaudio.transforms.Spectrogram(power=None, n_fft=config.N_FFT)
        # Fix: Create separate mel scale transform
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=config.N_MELS, 
            sample_rate=config.SAMPLE_RATE, 
            n_stft=config.N_FFT // 2 + 1
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Create DCT matrix for MFCC computation
        self.register_buffer('dct_mat', torchaudio.functional.create_dct(config.N_MFCC, config.N_MELS, 'ortho'))
        self.deltas = torchaudio.transforms.ComputeDeltas(win_length=4)

    def forward(self, waveform):
        spec = self.spectrogram(waveform)
        
        # Convert to power spectrogram
        power_spec = spec.abs().pow(2)
        
        # Apply mel scale
        mel_spec = self.mel_scale(power_spec)
        
        # Convert to dB
        db_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Apply DCT to get MFCC
        mfcc = torch.matmul(db_mel_spec.transpose(-2, -1), self.dct_mat).transpose(-2, -1)
        
        # Take only the first N_MFCC coefficients
        mfcc = mfcc[:, :self.config.N_MFCC, :]

        delta = self.deltas(mfcc)
        return torch.cat((mfcc, delta), 1)

class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.gelu(self.norm(x)))
        return x

class SpeechRecognitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(config.N_FEATS_IN, config.N_FEATS_CNN, kernel_size=10, stride=2, padding=4),
            ActDropNormCNN1D(config.N_FEATS_CNN, config.DROPOUT)
        )
        self.dense = nn.Sequential(
            nn.Linear(config.N_FEATS_CNN, config.N_FEATS_DENSE), nn.LayerNorm(config.N_FEATS_DENSE),
            nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(config.N_FEATS_DENSE, config.N_FEATS_DENSE), nn.LayerNorm(config.N_FEATS_DENSE),
            nn.GELU(), nn.Dropout(config.DROPOUT)
        )
        # CHANGE: Ensure testing model matches training model with batch_first=True
        self.transformer = nn.Transformer(
            d_model=config.TRANSFORMER_D_MODEL, 
            nhead=config.TRANSFORMER_NHEAD,
            num_encoder_layers=config.TRANSFORMER_ENCODER_LAYERS, 
            num_decoder_layers=config.TRANSFORMER_DECODER_LAYERS,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD, 
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(config.TRANSFORMER_D_MODEL)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.final_fc = nn.Linear(config.TRANSFORMER_D_MODEL, config.NUM_CLASSES)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.cnn(x)
        x = self.dense(x)
        x = self.transformer(x, x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.final_fc(x)
        return F.log_softmax(x.transpose(0, 1), dim=2)

# =============================================================================
# Greedy CTC Decoder
# =============================================================================
def greedy_decoder(output, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2).squeeze(0) # Squeeze the time dimension
    decode = []
    for i, index in enumerate(arg_maxes):
        if index != blank_label:
            if collapse_repeated and i != 0 and index == arg_maxes[i - 1]:
                continue
            decode.append(index.item())
    return decode

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate (CER)"""
    ref_chars = list(reference.replace(' ', ''))
    hyp_chars = list(hypothesis.replace(' ', ''))
    
    # Simple Levenshtein distance calculation
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 100.0
    
    edit_distance = levenshtein_distance(ref_chars, hyp_chars)
    cer = (edit_distance / len(ref_chars)) * 100
    return cer

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Simple Levenshtein distance calculation for words
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 100.0
    
    edit_distance = levenshtein_distance(ref_words, hyp_words)
    wer = (edit_distance / len(ref_words)) * 100
    return wer

# =============================================================================
# Main Testing Function
# =============================================================================
def test_model():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test speech recognition model')
    parser.add_argument('-a', '--audio', type=str, help='Path to audio file (.mp3, .m4a, .wav, .flac, etc.)')
    parser.add_argument('-t', '--transcript', type=str, help='Ground truth transcript text')
    
    args = parser.parse_args()
    
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Model file not found at '{Config.BEST_MODEL_PATH}'.")
        print("Please run 'main.py' first to train and save the model.")
        return

    print(f"Loading model from {Config.BEST_MODEL_PATH}...")
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    model = SpeechRecognitionModel(Config).to(Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
    print(f"Model was saved at epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('loss', 'unknown')}")

    featurizer = AudioFeatures(Config)
    text_processor = TextProcess()
    
    # Check if custom audio file is specified
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found at '{args.audio}'.")
            return
        
        # Use custom audio file and transcript
        print(f"\nTesting on custom audio file:")
        audio_path = args.audio
        ground_truth_text = args.transcript if args.transcript else "No ground truth provided"
        
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert stereo to mono if necessary
            if waveform.shape[0] > 1:
                print(f"Converting stereo audio to mono (original shape: {waveform.shape})")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                print(f"Converted to mono (new shape: {waveform.shape})")
            
            if sr != Config.SAMPLE_RATE:
                print(f"Resampling from {sr}Hz to {Config.SAMPLE_RATE}Hz")
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=Config.SAMPLE_RATE)
            
            features = featurizer(waveform).unsqueeze(0).to(Config.DEVICE)
            
            print(f"\n{'='*60}")
            print(f"Custom Audio Test")
            print(f"Audio File: {os.path.basename(audio_path)}")
            print(f"Audio duration: {waveform.shape[1] / Config.SAMPLE_RATE:.2f} seconds")
            print(f"Features shape: {features.shape}")

            print("\nTranscribing audio file...")
            with torch.no_grad():
                output = model(features)
            
            print(f"Model output shape: {output.shape}")
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
            
            # Show raw predictions for debugging
            raw_predictions = torch.argmax(output, dim=2).squeeze().cpu().numpy()
            print(f"Raw predictions (first 20): {raw_predictions[:20]}")
            
            decoded_indices = greedy_decoder(output.cpu())
            transcribed_text = text_processor.int_to_text_sequence(decoded_indices)
            
            # Strip leading and trailing whitespace from transcribed text
            transcribed_text = transcribed_text.strip()

            print("-" * 50)
            print(f"Ground Truth Transcript: '{ground_truth_text}'")
            print(f"Model Prediction:        '{transcribed_text}'")
            print(f"Decoded indices: {decoded_indices[:20]}...")
            print("-" * 50)
            
            # Calculate basic metrics
            if transcribed_text.strip():
                print(f"Transcript length - Ground Truth: {len(ground_truth_text)}, Predicted: {len(transcribed_text)}")
                
                # Calculate CER and WER if ground truth is provided
                if args.transcript:
                    cer = calculate_cer(ground_truth_text, transcribed_text)
                    wer = calculate_wer(ground_truth_text, transcribed_text)
                    print(f"Character Error Rate (CER): {cer:.2f}%")
                    print(f"Word Error Rate (WER): {wer:.2f}%")
            else:
                print("WARNING: Empty transcription!")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            import traceback
            traceback.print_exc()
            return
    
    else:
        # Default behavior: test on random samples from test dataset
        test_manifest_path = os.path.join("./data", "test-clean_manifest.json")
        if not os.path.exists(test_manifest_path):
            print(f"Error: Test manifest not found at '{test_manifest_path}'.")
            return
            
        test_data = pd.read_json(test_manifest_path, lines=True)
        
        # Test multiple samples for better debugging
        num_samples = min(3, len(test_data))
        print(f"\nTesting on {num_samples} random samples from LibriSpeech:")
        
        for sample_idx in range(num_samples):
            random_sample = test_data.sample(1).iloc[0]
            audio_path = random_sample['key']
            ground_truth_text = random_sample['text']
            
            try:
                waveform, sr = torchaudio.load(audio_path)
                
                # Convert stereo to mono if necessary (though LibriSpeech should be mono)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if sr != Config.SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=Config.SAMPLE_RATE)
                
                features = featurizer(waveform).unsqueeze(0).to(Config.DEVICE)
                
                print(f"\n{'='*60}")
                print(f"Sample {sample_idx + 1}")
                print(f"Audio File: {os.path.basename(audio_path)}")
                print(f"Audio duration: {waveform.shape[1] / Config.SAMPLE_RATE:.2f} seconds")
                print(f"Features shape: {features.shape}")

                print("\nTranscribing audio file...")
                with torch.no_grad():
                    output = model(features)
                
                print(f"Model output shape: {output.shape}")
                print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
                
                # Show raw predictions for debugging
                raw_predictions = torch.argmax(output, dim=2).squeeze().cpu().numpy()
                print(f"Raw predictions (first 20): {raw_predictions[:20]}")
                
                decoded_indices = greedy_decoder(output.cpu())
                transcribed_text = text_processor.int_to_text_sequence(decoded_indices)
                
                # Strip leading and trailing whitespace from transcribed text
                transcribed_text = transcribed_text.strip()

                print("-" * 50)
                print(f"Original Transcript:    '{ground_truth_text}'")
                print(f"Transcribed Transcript: '{transcribed_text}'")
                print(f"Decoded indices: {decoded_indices[:20]}...")
                print("-" * 50)
                
                # Calculate basic metrics
                if transcribed_text.strip():
                    print(f"Transcript length - Original: {len(ground_truth_text)}, Predicted: {len(transcribed_text)}")
                    
                    # Calculate CER and WER
                    cer = calculate_cer(ground_truth_text, transcribed_text)
                    wer = calculate_wer(ground_truth_text, transcribed_text)
                    print(f"Character Error Rate (CER): {cer:.2f}%")
                    print(f"Word Error Rate (WER): {wer:.2f}%")
                else:
                    print("WARNING: Empty transcription!")
                    
            except Exception as e:
                print(f"Error processing sample {sample_idx + 1}: {e}")
                continue

if __name__ == "__main__":
    test_model()