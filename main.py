import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import os
import tarfile
import requests
from tqdm import tqdm
import pandas as pd
import unicodedata
import string
from testing import test_model
import multiprocessing as mp

# =============================================================================
# 1. Configuration & Hyperparameters
# =============================================================================
class Config:
    DATASET_URLS = {
        "train-clean-360": "https://openslr.magicdatatech.com/resources/12/train-clean-360.tar.gz",
        "dev-clean": "https://openslr.magicdatatech.com/resources/12/dev-clean.tar.gz",
        "test-clean": "https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz",
    }
    DATA_PATH = "./data"
    LIBRISPEECH_PATH = os.path.join(DATA_PATH, "LibriSpeech")
    BEST_MODEL_PATH = "./models/best.pt"
    
    EPOCHS = 30
    BATCH_SIZE = 8  # Reduced for gradient accumulation
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8 * 2 = 16
    LR = 1e-4
    
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
    SPEC_AUG_RATE = 0.5
    TIME_STRETCH_MAX = 1.1
    TIME_STRETCH_MIN = 0.9
    FREQ_MASK = 8   # Reduced from 15 to be less aggressive
    TIME_MASK = 15  # Reduced from 35 to be less aggressive

# =============================================================================
# 2. Data Preparation
# =============================================================================
def download_and_unpack(url, download_path, librispeech_path):
    split_name = os.path.basename(url).replace('.tar.gz', '')
    
    if not os.path.exists(os.path.join(librispeech_path, split_name)):
        filename = os.path.basename(url)
        file_path = os.path.join(download_path, filename)
        
        os.makedirs(download_path, exist_ok=True)
        print(f"Split '{split_name}' not found. Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=filename, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            
            print(f"Extracting {filename}...")
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=download_path)
            os.remove(file_path)
            print(f"Extracted to {download_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
    else:
        print(f"Dataset split '{split_name}' found at {librispeech_path}. Skipping download.")

def create_manifest(librispeech_path, split_name, manifest_path):
    if os.path.exists(manifest_path):
        print(f"Manifest {manifest_path} already exists. Skipping creation.")
        return
        
    print(f"Creating manifest for {split_name}...")
    data = []
    walk_path = os.path.join(librispeech_path, split_name)
    
    if not os.path.exists(walk_path):
        print(f"Error: Directory not found to create manifest: {walk_path}")
        return

    for root, _, files in tqdm(os.walk(walk_path), desc=f"Scanning {split_name}"):
        for file in files:
            if file.endswith('.trans.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            audio_id, transcript = parts
                            audio_path = os.path.join(root, f"{audio_id}.flac")
                            if os.path.exists(audio_path):
                                data.append({"key": audio_path, "text": transcript.lower()})
    
    pd.DataFrame(data).to_json(manifest_path, orient='records', lines=True)
    print(f"Manifest created at {manifest_path} with {len(data)} entries.")

# =============================================================================
# 3. Text & Audio Processing
# =============================================================================
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

    def unicode_to_ascii(self, s):
        ALL_LETTERS = string.ascii_letters + " '"
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)

    def text_to_int_sequence(self, text):
        text = self.unicode_to_ascii(text)
        return [self.char_map.get(c, self.char_map['<UNK>']) if c != ' ' else self.char_map['<SPACE>'] for c in text.lower()]

    def int_to_text_sequence(self, labels):
        return ''.join([self.index_map.get(i, '?') for i in labels]).replace('<SPACE>', ' ')

class SpecStretch(nn.Module):
    def __init__(self, rate, stretch_max, stretch_min):
        super().__init__()
        self.rate = rate
        self.stretch_max = stretch_max
        self.stretch_min = stretch_min

    def forward(self, spec):
        if torch.rand(1).item() < self.rate:
            rate = torch.empty(1).uniform_(self.stretch_min, self.stretch_max).item()
            stretch_transform = torchaudio.transforms.TimeStretch(fixed_rate=rate, n_freq=spec.size(1))
            return stretch_transform(spec.permute(0, 2, 1)).permute(0, 2, 1)
        return spec

class AudioFeatures(nn.Module):
    def __init__(self, config, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.config = config
        self.spectrogram = torchaudio.transforms.Spectrogram(power=None, n_fft=config.N_FFT)
        self.spec_stretch = SpecStretch(config.SPEC_AUG_RATE, config.TIME_STRETCH_MAX, config.TIME_STRETCH_MIN)
        
        # Add lighter SpecAugment transforms for training (only apply 50% of the time)
        if is_train:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=config.FREQ_MASK)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=config.TIME_MASK)
            self.augment_prob = 0.3  # Only apply SpecAugment 30% of the time
        
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
        if self.is_train:
            spec = self.spec_stretch(spec)
        
        # Convert to power spectrogram
        power_spec = spec.abs().pow(2)
        
        # Apply mel scale
        mel_spec = self.mel_scale(power_spec)
        
        # Apply lighter SpecAugment during training (with probability control)
        if self.is_train and torch.rand(1).item() < self.augment_prob:
            # Apply frequency masking
            mel_spec = self.freq_mask(mel_spec)
            # Apply time masking
            mel_spec = self.time_mask(mel_spec)
        
        # Convert to dB
        db_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Apply DCT to get MFCC
        mfcc = torch.matmul(db_mel_spec.transpose(-2, -1), self.dct_mat).transpose(-2, -1)
        
        # Take only the first N_MFCC coefficients
        mfcc = mfcc[:, :self.config.N_MFCC, :]

        delta = self.deltas(mfcc)
        return torch.cat((mfcc, delta), 1)

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, config, is_train=True):
        self.data = pd.read_json(manifest_path, lines=True)
        self.text_process = TextProcess()
        self.audio_transforms = AudioFeatures(config, is_train=is_train)
        self.sample_rate = config.SAMPLE_RATE
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.item()
        try:
            item = self.data.iloc[idx]
            waveform, sr = torchaudio.load(item['key'])
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            
            features = self.audio_transforms(waveform)
            label = self.text_process.text_to_int_sequence(item['text'])
            
            l_in = features.shape[2]
            spec_len = (l_in + 2*4 - 10) // 2 + 1

            if features.shape[0] > 1: raise ValueError("Dual channel")
            if l_in > 3000: raise ValueError("Too long")
            if len(label) == 0: raise ValueError("Empty transcript")
            if spec_len < len(label): raise ValueError("Output shorter than label")

            return features, torch.tensor(label, dtype=torch.long), spec_len, len(label)
        except Exception:
            return None
        
def collate_fn(data):
    data = [item for item in data if item is not None]
    if not data:
        return None, None, None, None

    spectrograms, labels, spec_lengths, label_lengths = zip(*data)
    
    spectrograms = [s.squeeze(0).transpose(0, 1) for s in spectrograms]
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence([l for l in labels], batch_first=True)
    
    return spectrograms, labels, torch.tensor(spec_lengths, dtype=torch.long), torch.tensor(label_lengths, dtype=torch.long)

# =============================================================================
# 4. Model Architecture
# =============================================================================
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
            nn.Conv1d(config.N_FEATS_IN, config.N_FEATS_CNN, kernel_size=10, stride=2, padding=5),
            ActDropNormCNN1D(config.N_FEATS_CNN, config.DROPOUT)
        )
        self.dense = nn.Sequential(
            nn.Linear(config.N_FEATS_CNN, config.N_FEATS_DENSE), nn.LayerNorm(config.N_FEATS_DENSE),
            nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(config.N_FEATS_DENSE, config.N_FEATS_DENSE), nn.LayerNorm(config.N_FEATS_DENSE),
            nn.GELU(), nn.Dropout(config.DROPOUT)
        )
        # CHANGE: Re-added batch_first=True to resolve the warning
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
        # Input x: [Batch, 1, Features, Time]
        x = x.squeeze(1)      # -> [Batch, Features, Time]
        x = self.cnn(x)       # -> [Batch, Time, Features]
        x = self.dense(x)     # -> [Batch, Time, Features]
        
        # Transformer with batch_first=True expects [Batch, Time, Features]
        x = self.transformer(x, x) # -> [Batch, Time, Features]
        
        # Post-transformer normalization
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        x = self.final_fc(x)  # -> [Batch, Time, Classes]
        
        # CTCLoss expects [Time, Batch, Classes]
        return F.log_softmax(x.transpose(0, 1), dim=2)

# =============================================================================
# 5. Training and Validation Loop
# =============================================================================
def test_saved_model():
    """Test the saved model on a sample audio file"""
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("No saved model found for testing.")
        return
    
    print("\n" + "="*60)
    print("TESTING SAVED MODEL")
    print("="*60)
    
    # Load model
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    model = SpeechRecognitionModel(Config).to(Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load a test sample
    test_manifest_path = os.path.join(Config.DATA_PATH, "test-clean_manifest.json")
    if not os.path.exists(test_manifest_path):
        print("Test manifest not found.")
        return
    
    test_data = pd.read_json(test_manifest_path, lines=True)
    sample = test_data.iloc[0]  # Take first sample
    
    print(f"Audio File: {os.path.basename(sample['key'])}")
    print(f"Ground Truth: '{sample['text']}'")
    
    # Process audio
    waveform, sr = torchaudio.load(sample['key'])
    if sr != Config.SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=Config.SAMPLE_RATE)
    
    audio_features = AudioFeatures(Config, is_train=False)
    features = audio_features(waveform).unsqueeze(0).to(Config.DEVICE)
    
    # Get model output
    with torch.no_grad():
        output = model(features)
    
    print(f"Model Output Shape: {output.shape}")
    print(f"Output Range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Simple greedy decoding
    predicted_indices = torch.argmax(output, dim=2).squeeze().cpu().numpy()
    print(f"Raw Predictions (first 20): {predicted_indices[:20]}")
    
    # Remove blanks and consecutive duplicates
    text_processor = TextProcess()
    decoded = []
    prev = None
    for idx in predicted_indices:
        if idx != 0 and idx != prev:  # 0 is blank token
            decoded.append(idx)
        prev = idx
    
    predicted_text = text_processor.int_to_text_sequence(decoded)
    print(f"Predicted Text: '{predicted_text}'")
    print("="*60)

def train_one_epoch(model, data_loader, criterion, optimizer, scheduler, epoch, config):
    model.train()
    total_loss = 0.0
    items_processed = 0
    
    # Initialize gradient scaler for mixed precision - Updated API
    scaler = torch.amp.GradScaler("cuda")
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    
    # Initialize gradient accumulation
    optimizer.zero_grad()
    
    for i, (spectrograms, labels, spec_lengths, label_lengths) in enumerate(progress_bar):
        if spectrograms is None:
            continue

        spectrograms, labels = spectrograms.to(config.DEVICE), labels.to(config.DEVICE)
        spec_lengths = spec_lengths.to(config.DEVICE)
        label_lengths = label_lengths.to(config.DEVICE)
        
        # DEBUG: Check for empty batches or invalid data
        if spectrograms.size(0) == 0:
            print(f"\nWarning: Empty batch {i}")
            continue
        
        # DEBUG: Print shapes and values for first batch of first epoch
        if i == 0 and epoch == 1:
            print(f"\nDEBUG - First Training Batch:")
            print(f"  Input shape: {spectrograms.shape}")
            print(f"  Spec lengths: {spec_lengths}")
            print(f"  Label lengths: {label_lengths}")
        
        # Check for valid inputs to CTC loss
        if torch.any(spec_lengths <= 0) or torch.any(label_lengths <= 0):
            print(f"\nWarning: Invalid lengths in batch {i}")
            continue
        
        # Use automatic mixed precision - Updated API
        with torch.amp.autocast("cuda"):
            outputs = model(spectrograms)
            loss = criterion(outputs, labels, spec_lengths, label_lengths)
            
            # Scale loss for gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # DEBUG: Print loss value only for first batch of first epoch
        if i == 0 and epoch == 1:
            print(f"  Raw loss: {loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.8f}")
            print(f"  Is loss zero? {loss.item() == 0.0}")
        
        if torch.isinf(loss) or torch.isnan(loss): 
            print(f"\nWarning: Skipping batch {i} due to invalid loss value: {loss.item()}")
            continue

        # Check if loss is exactly zero (which indicates a problem)
        if loss.item() == 0.0:
            print(f"\nWarning: Loss is exactly zero for batch {i}")
            if i < 3:  # Only for first few batches
                print(f"  CTC inputs - outputs: {outputs.shape}, labels: {labels.shape}")
                print(f"  CTC inputs - input_lengths: {spec_lengths}, target_lengths: {label_lengths}")

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation: only step optimizer every N batches
        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping and optimizer step with scaler
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Step the OneCycleLR scheduler after each optimizer step
            scheduler.step()
            
            optimizer.zero_grad()
        
        # Track loss (scale back up for reporting)
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        items_processed += 1
        if items_processed > 0:
            progress_bar.set_postfix(loss=total_loss/items_processed)
    
    # Handle remaining gradients if batch count doesn't divide evenly
    remaining_steps = items_processed % config.GRADIENT_ACCUMULATION_STEPS
    if remaining_steps != 0:
        # Only step if we actually have accumulated some gradients
        try:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Step scheduler here too
        except AssertionError:
            # If no gradients were scaled, just do a regular step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad()
        
    return total_loss / items_processed if items_processed > 0 else 0.0

def validate_one_epoch(model, data_loader, criterion, epoch, config):
    model.eval()
    total_loss = 0.0
    items_processed = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Valid]")
    
    with torch.no_grad():
        for i, (spectrograms, labels, spec_lengths, label_lengths) in enumerate(progress_bar):
            if spectrograms is None:
                continue

            spectrograms, labels = spectrograms.to(config.DEVICE), labels.to(config.DEVICE)
            spec_lengths = spec_lengths.to(config.DEVICE)
            label_lengths = label_lengths.to(config.DEVICE)
            
            # Use automatic mixed precision for validation too - Updated API
            with torch.amp.autocast("cuda"):
                outputs = model(spectrograms)
                loss = criterion(outputs, labels, spec_lengths, label_lengths)
            
            # DEBUG: Print first validation loss only
            if i == 0 and epoch == 1:
                print(f"\nDEBUG - Validation batch {i} loss: {loss.item():.6f}")
            
            if torch.isinf(loss) or torch.isnan(loss): continue
            
            total_loss += loss.item()
            items_processed += 1
            if items_processed > 0:
                progress_bar.set_postfix(loss=total_loss/items_processed)
            
    return total_loss / items_processed if items_processed > 0 else 0.0

def debug_model_output():
    """Debug function to test model output on a sample"""
    print("\n" + "="*60)
    print("DEBUG: Testing model output on a sample")
    print("="*60)
    
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("No saved model found for debugging.")
        return
    
    # Load model
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    model = SpeechRecognitionModel(Config).to(Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load a test sample
    test_manifest_path = os.path.join(Config.DATA_PATH, "test-clean_manifest.json")
    if not os.path.exists(test_manifest_path):
        print("Test manifest not found.")
        return
    
    test_data = pd.read_json(test_manifest_path, lines=True)
    sample = test_data.iloc[0]  # Take first sample
    
    print(f"Testing on: {os.path.basename(sample['key'])}")
    print(f"Ground truth: '{sample['text']}'")
    
    # Process audio
    waveform, sr = torchaudio.load(sample['key'])
    if sr != Config.SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=Config.SAMPLE_RATE)
    
    audio_features = AudioFeatures(Config, is_train=False)
    features = audio_features(waveform).unsqueeze(0).to(Config.DEVICE)
    
    # Get model output
    with torch.no_grad():
        output = model(features)
    
    print(f"Model output shape: {output.shape}")
    print(f"Output min/max: {output.min():.4f} / {output.max():.4f}")
    
    # Simple greedy decoding
    predicted_indices = torch.argmax(output, dim=2).squeeze().cpu().numpy()
    print(f"Raw predictions (first 20): {predicted_indices[:20]}")
    
    # Remove blanks and consecutive duplicates
    text_processor = TextProcess()
    decoded = []
    prev = None
    for idx in predicted_indices:
        if idx != 0 and idx != prev:  # 0 is blank token
            decoded.append(idx)
        prev = idx
    
    predicted_text = text_processor.int_to_text_sequence(decoded)
    print(f"Decoded text: '{predicted_text}'")
    print("="*60)

def main():
    # Enable cuDNN benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Enable optimized attention for memory efficiency (only if available)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except AttributeError:
        pass  # Not available in older PyTorch versions
    
    for name, url in Config.DATASET_URLS.items():
        download_and_unpack(url, Config.DATA_PATH, Config.LIBRISPEECH_PATH)
        manifest_path = os.path.join(Config.DATA_PATH, f"{name}_manifest.json")
        create_manifest(Config.LIBRISPEECH_PATH, name, manifest_path)

    train_dataset = LibriSpeechDataset(os.path.join(Config.DATA_PATH, "train-clean-360_manifest.json"), Config, is_train=True)
    val_dataset = LibriSpeechDataset(os.path.join(Config.DATA_PATH, "dev-clean_manifest.json"), Config, is_train=False)
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=6,
        pin_memory=True, 
        collate_fn=collate_fn,
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=6,
        pin_memory=True, 
        collate_fn=collate_fn,
        persistent_workers=True
    )
    
    # Create model without JIT compilation initially
    model = SpeechRecognitionModel(Config).to(Config.DEVICE)
    
    
    print(f"Using device: {Config.DEVICE}")
    print(f"Model created. Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    
    # OneCycleLR scheduler for better convergence
    total_steps = len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS * Config.EPOCHS  
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR * 10,  # Peak learning rate (10x base LR)
        total_steps=total_steps,
        pct_start=0.1,  # Percentage of cycle spent increasing LR
        anneal_strategy='cos',  # Cosine annealing
        div_factor=25.0,  # Initial LR = max_lr / div_factor
        final_div_factor=1e4,  # Final LR = initial_lr / final_div_factor
    )
    
    best_val_loss = float('inf')
    
    print("Starting training with conservative optimizations:")
    print(f"- Mixed Precision Training: Enabled")
    print(f"- Gradient Accumulation Steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"- Data Loader Workers: 6")
    print(f"- CuDNN Benchmark: Enabled")
    print(f"- Learning Rate: {Config.LR} (OneCycleLR scheduler)")
    print(f"- SpecAugment: Light (30% prob, freq_mask={Config.FREQ_MASK}, time_mask={Config.TIME_MASK})")
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, Config)
        val_loss = validate_one_epoch(model, val_loader, criterion, epoch, Config)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {val_loss:.6f}. Saving model to {Config.BEST_MODEL_PATH}")
            
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(Config.BEST_MODEL_PATH), exist_ok=True)
            
            # Save model state dict
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss
            }, Config.BEST_MODEL_PATH)
            
            # Test the newly saved model
            test_saved_model()
        
        # Memory management: clear cache periodically
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
        
        # Stop early if loss is still 0 after a few epochs
        if epoch >= 3 and train_loss == 0.0 and val_loss == 0.0:
            print("Warning: Loss is still 0.0 after 3 epochs. There may be an issue with the model or data.")
            break

    print("\nTraining finished. Running final test on a sample audio file...")
    
    # Clean up memory
    del train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()
    
    test_model()

if __name__ == "__main__":
    mp.freeze_support()
    main()