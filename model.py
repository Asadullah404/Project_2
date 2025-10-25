import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import numpy as np
import torch.nn.functional as F
import math
import librosa
from pesq import pesq
from pystoi import stoi
from torch.optim.lr_scheduler import ExponentialLR
import traceback

# --- Global Configuration (Must match Colab) ---
HOP_SIZE = 160 # 10ms frame (160 samples / 16000 Hz = 0.01s)
LATENT_DIM = 128 

VQ_EMBEDDINGS = 256
NUM_VQ_STAGES = 2 
VQ_INDICES_PER_STAGE = 10 
NUM_QUANTIZERS = VQ_INDICES_PER_STAGE # 16 kbps target

# --- DAC INTEGRATION ---
try:
    import dac
    DAC_AVAILABLE = True
except ImportError:
    DAC_AVAILABLE = False
    print("Warning: DAC not available. Install with: pip install descript-audio-codec")

class DACCodec:
    """Wrapper for Descript Audio Codec (DAC) at 16kHz with 20ms latency"""
    def __init__(self, model_path=None, model_type="16khz"):
        if not DAC_AVAILABLE:
            raise ImportError("DAC is not installed. Install with: pip install descript-audio-codec")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # Load DAC model
        if model_path and os.path.exists(model_path):
            self.model = dac.DAC.load(model_path)
        else:
            try:
                # Try to download the standard 16kHz model if no path is provided
                model_path = dac.utils.download(model_type="16khz")
                self.model = dac.DAC.load(model_path)
            except:
                # Fallback in case of multiple download issues
                self.model = dac.DAC.load(dac.utils.download(model_type="16khz"))
        
        self.model.to(self.device)
        self.model.eval()
        
        # DAC 16kHz specs - optimized for low latency
        self.sample_rate = 16000
        self.hop_size = 320  # DAC's internal hop size (20ms)
        self.chunk_size = 320  # 20ms at 16kHz (matches hop size for minimal latency)
        
    def encode(self, audio_tensor):
        """
        Encode audio to DAC latent codes
        audio_tensor: (B, 1, T) or (B, T)
        Returns: codes tensor and original length
        """
        with torch.no_grad():
            # Ensure shape is (B, 1, T)
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(1)
            elif audio_tensor.dim() == 3 and audio_tensor.shape[1] == 1:
                pass # Correct shape
            elif audio_tensor.dim() == 3 and audio_tensor.shape[1] > 1:
                # Handle multi-channel by taking first channel
                audio_tensor = audio_tensor[:, :1, :]
            
            audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
            
            # Store original length
            original_length = audio_tensor.shape[-1]
            
            # Pad to ensure length is compatible with DAC (must be multiple of hop_size)
            if original_length < self.hop_size:
                pad_length = self.hop_size - original_length
                audio_tensor = F.pad(audio_tensor, (0, pad_length))
            elif original_length % self.hop_size != 0:
                pad_length = self.hop_size - (original_length % self.hop_size)
                audio_tensor = F.pad(audio_tensor, (0, pad_length))
            
            # DAC encode: returns (z, codes, latents, commitment_loss, codebook_loss)
            # We only need 'codes' for discrete representation
            _, codes, _, _, _ = self.model.encode(audio_tensor)
            
            # codes shape: (batch, n_codebooks, sequence_length)
            return codes, original_length
    
    def decode(self, codes, original_length=None):
        """
        Decode DAC codes to audio
        codes: (batch, n_codebooks, sequence_length) discrete codes from encode
        original_length: original audio length to trim to
        Returns: (B, 1, T) audio tensor
        """
        with torch.no_grad():
            # Ensure codes is on the right device
            if not isinstance(codes, torch.Tensor):
                # DAC codes should typically be torch.LongTensor
                codes = torch.tensor(codes, dtype=torch.long)
            codes = codes.to(self.device)
            
            # Convert codes back to continuous latent representation
            z = self.model.quantizer.from_codes(codes)[0]
            
            # Decode from latent
            audio_recon = self.model.decode(z)
            
            # Trim to original length if provided
            # NOTE: DAC.decode might handle trimming if z was derived from a padded tensor, 
            # but explicit trimming here ensures consistency, especially for chunked data.
            if original_length is not None and audio_recon.shape[-1] > original_length:
                audio_recon = audio_recon[..., :original_length]
            
            return audio_recon
    
    def compress(self, audio_tensor):
        """Full compression pipeline - returns codes and metadata"""
        codes, original_length = self.encode(audio_tensor)
        return codes, original_length
    
    def decompress(self, codes, original_length):
        """Full decompression pipeline"""
        return self.decode(codes, original_length)

# --- Loss Components ---
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = torch.hann_window
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

    def forward(self, y_hat, y):
        sc_loss, mag_loss = 0.0, 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = self.window(win, device=y.device)
            y_hat_float = y_hat.squeeze(1).to(torch.float32)
            y_float = y.squeeze(1).to(torch.float32)
            spec_hat = torch.stft(y_hat_float, n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            spec = torch.stft(y_float, n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            
            sc_loss += torch.norm(torch.abs(spec) - torch.abs(spec_hat), p='fro') / (torch.norm(torch.abs(spec), p='fro') + 1e-6)
            mag_loss += F.l1_loss(torch.log(torch.abs(spec).clamp(min=1e-9)), torch.log(torch.abs(spec_hat).clamp(min=1e-9)))
            
        return (sc_loss / len(self.fft_sizes)) + (mag_loss / len(self.fft_sizes))

# --- Causal Components ---
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.kernel_size[0] - 1
    def forward(self, x):
        return super().forward(F.pad(x, (self.causal_padding, 0)))

class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.kernel_size[0] - self.stride[0]
    def forward(self, x):
        x = super().forward(x)
        if self.causal_padding != 0:
            return x[..., :-self.causal_padding]
        return x

# --- Residual Vector Quantizer (RVQ) ---
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_stages, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_stages = num_stages
        self.commitment_cost = commitment_cost
        self.vqs = nn.ModuleList([
            VectorQuantizerSingle(num_embeddings, embedding_dim, 0.0) 
            for _ in range(num_stages)
        ])

    def forward(self, z_e):
        quantized_output = 0.0 
        residual = z_e
        total_vq_loss = 0.0
        all_indices = []

        for vq in self.vqs:
            z_q_i, vq_loss_i, indices_i = vq(residual)
            residual = residual - z_q_i.detach()
            quantized_output = quantized_output + z_q_i
            total_vq_loss += vq_loss_i
            all_indices.append(indices_i)
        
        total_vq_loss = total_vq_loss * self.commitment_cost / self.num_stages
        return quantized_output, total_vq_loss, torch.stack(all_indices, dim=1) 

class VectorQuantizerSingle(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z_e):
        z_e_float = z_e.to(torch.float32)
        z_e_flat = z_e_float.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z_e_float.shape[0], -1, self.embedding_dim)
        z_q = z_q.permute(0, 2, 1).contiguous()
        e_loss = F.mse_loss(z_q.detach(), z_e_float)
        z_q = z_e + (z_q - z_e_float).detach()
        return z_q, e_loss, encoding_indices.view(z_e.shape[0], -1) 


# --- Encoder/Decoder (16x Downsampling) ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        ResBlock = lambda c: nn.Sequential(CausalConv1d(c, c, 3), nn.ELU(), CausalConv1d(c, c, 1))
        self.net = nn.Sequential(
            CausalConv1d(1, 64, 5), nn.ELU(), ResBlock(64),
            CausalConv1d(64, 128, 3, stride=2), nn.ELU(), ResBlock(128),
            CausalConv1d(128, 256, 3, stride=2), nn.ELU(), ResBlock(256),
            CausalConv1d(256, 512, 3, stride=2), nn.ELU(), ResBlock(512),
            CausalConv1d(512, 512, 3, stride=2), nn.ELU(), ResBlock(512),
            CausalConv1d(512, LATENT_DIM, 3, stride=1), nn.ELU(),
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ResBlock = lambda c: nn.Sequential(CausalConv1d(c, c, 3), nn.ELU(), CausalConv1d(c, c, 1))
        self.net = nn.Sequential(
            CausalConvTranspose1d(LATENT_DIM, 512, 3, stride=1), nn.ELU(), ResBlock(512),
            CausalConvTranspose1d(512, 512, 3, stride=2), nn.ELU(), ResBlock(512),
            CausalConvTranspose1d(512, 256, 3, stride=2), nn.ELU(), ResBlock(256),
            CausalConvTranspose1d(256, 128, 3, stride=2), nn.ELU(), ResBlock(128),
            CausalConvTranspose1d(128, 64, 3, stride=2), nn.ELU(), ResBlock(64),
            CausalConv1d(64, 1, 3), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# --- Causal Transformer ---
class CausalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1, history_chunks=0): 
        super().__init__()
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation=F.gelu) 
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.state_len = history_chunks * VQ_INDICES_PER_STAGE

    def get_causal_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).to(torch.bool)

    def forward(self, x, state):
        if self.state_len > 0 and state is not None:
            state_history = state[:, -self.state_len:, :]
            inp = torch.cat([state_history, x], dim=1)
        else:
            inp = x
        
        new_state = inp.detach()
        mask = self.get_causal_mask(inp.shape[1], x.device)
        norm_inp = self.norm(inp)
        out = self.transformer(norm_inp, mask=mask, src_key_padding_mask=None)
        out = out[:, -x.shape[1]:, :] 
        return out, new_state

# --- Generator: TS3 Codec (RVQ Version) ---
class TS3_Codec(nn.Module):
    """The Generator model (GACodec) for streaming."""
    def __init__(self, history_chunks=0):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = ResidualVectorQuantizer(NUM_VQ_STAGES, VQ_EMBEDDINGS, LATENT_DIM, 0.25)
        self.encoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=1, history_chunks=history_chunks)
        self.decoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=1, history_chunks=history_chunks)
        self.decoder = Decoder()
        
    def forward(self, x, h_enc=None, h_dec=None):
        x = x.to(torch.float32) 
        z_e = self.encoder(x)
        z_e_tfm_in = z_e.permute(0, 2, 1)
        z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
        z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1)
        z_q, vq_loss, indices = self.quantizer(z_e_tfm_out) 
        z_q_tfm_in = z_q.permute(0, 2, 1)
        z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q_tfm_in, h_dec)
        z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1)
        x_hat = self.decoder(z_q_tfm_out)
        return x_hat, vq_loss, (h_enc_new, h_dec_new)

    def encode(self, x, h_enc=None):
        """Streaming encode path: output flattened indices (B, 20)."""
        x = x.to(torch.float32) 
        z_e = self.encoder(x)
        z_e_tfm_in = z_e.permute(0, 2, 1)
        z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
        z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1)
        with torch.no_grad():
            _, _, indices = self.quantizer(z_e_tfm_out)
        return indices.view(indices.shape[0], -1), h_enc_new

    def decode(self, indices, h_dec=None):
        """Streaming decode path: input flattened indices (B, 20)."""
        indices = indices.view(indices.shape[0], NUM_VQ_STAGES, VQ_INDICES_PER_STAGE)
        z_q = 0.0
        for stage in range(NUM_VQ_STAGES):
            vq_layer = self.quantizer.vqs[stage]
            indices_i = indices[:, stage, :]
            # Ensure indices are long/int type for embedding lookup
            indices_i = indices_i.to(torch.long)
            z_q_i = vq_layer.embedding(indices_i)
            z_q = z_q + z_q_i.permute(0, 2, 1)
            
        z_q_tfm_in = z_q.permute(0, 2, 1)
        z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q_tfm_in, h_dec)
        z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1)
        x_hat = self.decoder(z_q_tfm_out)
        return x_hat, h_dec_new


# --- Discriminator: Multi-Scale Adversarial Network ---
class Discriminator(nn.Module):
    def __init__(self, start_channels=16):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, start_channels, 15, stride=1, padding=7),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(nn.Conv1d(start_channels, start_channels * 2, 41, stride=4, padding=20, groups=4), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Conv1d(start_channels * 2, start_channels * 4, 41, stride=4, padding=20, groups=16), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Conv1d(start_channels * 4, start_channels * 8, 41, stride=4, padding=20, groups=64), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Conv1d(start_channels * 8, start_channels * 8, 5, stride=1, padding=2), nn.LeakyReLU(0.2)),
        ])
        self.final_conv = nn.Conv1d(start_channels * 8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        feature_maps = []
        for layer in self.net:
            x = layer(x)
            feature_maps.append(x)
        output = self.final_conv(x)
        feature_maps.append(output)
        return output, feature_maps

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(start_channels=16), 
            Discriminator(start_channels=32),
            Discriminator(start_channels=64),
        ])
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        feature_maps = []
        
        out_d1, fm_d1 = self.discriminators[0](x)
        outputs.append(out_d1); feature_maps.append(fm_d1)

        x_2x = self.downsample(x) 
        out_d2, fm_d2 = self.discriminators[1](x_2x)
        outputs.append(out_d2); feature_maps.append(fm_d2)

        x_4x = self.downsample(x_2x)
        out_d3, fm_d3 = self.discriminators[2](x_4x)
        outputs.append(out_d3); feature_maps.append(fm_d3)

        return outputs, feature_maps


# --- TRADITIONAL CODECS (Baseline Comparison) ---
class MuLawCodec:
    def __init__(self, quantization_channels=256): self.mu = float(quantization_channels - 1)
    def encode(self, x):
        mu_t = torch.tensor(self.mu, dtype=torch.float32, device=x.device)
        encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
        return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
    def decode(self, z):
        z_float = z.to(torch.float32)
        mu_t = torch.tensor(self.mu, dtype=torch.float32, device=z.device)
        y = (z_float / self.mu) * 2.0 - 1.0
        return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

class ALawCodec:
    def __init__(self): self.A = 87.6
    def encode(self, x):
        a_t = torch.tensor(self.A, dtype=torch.float32, device=x.device)
        abs_x = torch.abs(x)
        encoded = torch.zeros_like(x)
        cond = abs_x < (1 / self.A)
        encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
        encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
        return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
    def decode(self, z):
        z_float = z.to(torch.float32)
        a_t = torch.tensor(self.A, dtype=torch.float32, device=z.device)
        y = (z_float / 127.5) - 1.0
        abs_y = torch.abs(y)
        decoded = torch.zeros_like(y)
        cond = abs_y < (1 / (1 + torch.log(a_t)))
        decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
        decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
        return decoded.unsqueeze(1)


# --- TRAINING HELPER FUNCTIONS ---

def generator_loss(disc_fake_features, disc_real_features_for_fm, fm_weight, gan_weight):
    """Calculates Generator Loss (Adversarial + Feature Matching)."""
    adv_loss = 0.0
    fm_loss = 0.0
    
    for disc_fake in disc_fake_features:
        adv_loss += F.mse_loss(disc_fake[-1], torch.ones_like(disc_fake[-1]))

    for (fake_features, real_features) in zip(disc_fake_features, disc_real_features_for_fm):
        for (fake_feature, real_feature) in zip(fake_features[:-1], real_features[:-1]): 
            fm_loss += F.l1_loss(fake_feature, real_feature.detach())
            
    num_disc = len(disc_fake_features)
    num_fm_layers = sum(len(f) - 1 for f in disc_fake_features)
    fm_loss = fm_loss / (num_fm_layers if num_fm_layers > 0 else 1.0)
    adv_loss = adv_loss / num_disc
    
    return adv_loss * gan_weight, fm_loss * fm_weight

def discriminator_loss(disc_real_features, disc_fake_features):
    """Calculates Discriminator Loss (Real vs Fake)."""
    loss = 0.0
    for real_out, fake_out in zip(disc_real_features, disc_fake_features):
        real_loss = F.mse_loss(real_out[-1], torch.ones_like(real_out[-1]))
        fake_loss = F.mse_loss(fake_out[-1], torch.zeros_like(fake_out[-1]))
        loss += (real_loss + fake_loss)
    return loss / len(disc_real_features)

TRAIN_CHUNK_SIZE = 16000

class AudioChunkDataset(Dataset):
    def __init__(self, directory, chunk_size=TRAIN_CHUNK_SIZE, sample_rate=16000):
        self.chunk_size, self.sample_rate = chunk_size, sample_rate
        self.file_paths = [os.path.join(r, f) for r, _, fs in os.walk(directory) for f in fs if f.lower().endswith(('.wav', '.flac'))]
        if not self.file_paths: raise ValueError("No audio files found.")
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        try:
            waveform, sr = torchaudio.load(self.file_paths[idx])
            if sr != self.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.to(torch.float32)
            if waveform.shape[1] > self.chunk_size:
                start = np.random.randint(0, waveform.shape[1] - self.chunk_size)
                waveform = waveform[:, start:start + self.chunk_size]
            else:
                waveform = F.pad(waveform, (0, self.chunk_size - waveform.shape[1]))
            if waveform.shape[1] != self.chunk_size:
                waveform = F.pad(waveform, (0, self.chunk_size - waveform.shape[1]))
            return waveform
        except Exception as e:
            print(f"Warning: Skipping file {self.file_paths[idx]}. Error: {e}")
            return torch.zeros((1, self.chunk_size), dtype=torch.float32)

def train_model(dataset_path, epochs, learning_rate, batch_size, model_save_path, progress_callback, stop_event, model_type, tfm_history_chunks, disc_warmup_steps, lr_decay_rate, lambda_stft, lambda_fm, d_g_ratio):
    """Main GAN Training function for the desktop app."""
    if model_type != 'transformer':
        progress_callback.emit("ERROR: Only TS3_Codec ('transformer') GAN training is supported."); return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        progress_callback.emit(f"Using device: {device}")
        
        train_dataset = AudioChunkDataset(directory=dataset_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        progress_callback.emit(f"Train Dataset loaded with {len(train_dataset)} files.")

        generator = TS3_Codec(history_chunks=tfm_history_chunks).to(device)
        discriminator = MultiScaleDiscriminator().to(device)

        optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        
        scheduler_g = ExponentialLR(optimizer_g, gamma=lr_decay_rate)
        scheduler_d = ExponentialLR(optimizer_d, gamma=lr_decay_rate)

        stft_criterion = MultiResolutionSTFTLoss().to(device)
        l1_criterion = nn.L1Loss().to(device)
        
        progress_callback.emit(f"Starting GAN training for TS3_Codec...")
        
        # Hyperparameters for Loss (L1, VQ are defaults)
        lambda_l1, lambda_vq = 1.0, 0.1
        gan_weight, fm_weight = 1.0, lambda_fm
        global_step = 0
        d_step_counter = 0

        for epoch in range(epochs):
            if stop_event.is_set():
                progress_callback.emit("Training stopped by user."); break
            
            generator.train(); discriminator.train()
            
            for i, data in enumerate(train_dataloader):
                if stop_event.is_set(): break
                inputs = data.to(device)
                global_step += 1
                d_step_counter += 1

                if d_step_counter % d_g_ratio == 0:
                    # --- Train Discriminator ---
                    optimizer_d.zero_grad()
                    x_hat, _, _ = generator(inputs)
                    x_hat_detached = x_hat.detach()
                    
                    _, disc_real_features = discriminator(inputs)
                    _, disc_fake_features = discriminator(x_hat_detached)
                    loss_d_total = discriminator_loss(disc_real_features, disc_fake_features)

                    loss_d_total.backward()
                    optimizer_d.step()
                    scheduler_d.step()
                
                if global_step > disc_warmup_steps:
                    # --- Train Generator ---
                    optimizer_g.zero_grad()
                    
                    x_hat, vq_loss, _ = generator(inputs)
                    stft_loss = stft_criterion(x_hat, inputs)
                    l1_loss = l1_criterion(x_hat, inputs)

                    disc_fake_outputs_g, disc_fake_features_g = discriminator(x_hat)
                    with torch.no_grad():
                        _, disc_real_features_for_fm = discriminator(inputs)
                    
                    adv_loss, fm_loss = generator_loss(disc_fake_features_g, disc_real_features_for_fm, fm_weight=fm_weight, gan_weight=gan_weight)
                    
                    loss_g_total = ((adv_loss + fm_loss) + 
                                    lambda_stft * stft_loss + 
                                    lambda_l1 * l1_loss + 
                                    lambda_vq * vq_loss)

                    loss_g_total.backward()
                    optimizer_g.step()
                    scheduler_g.step()

                    if global_step % 20 == 19:
                        # Ensure d_step_counter has run at least once for a meaningful D loss
                        d_loss_item = loss_d_total.item() if 'loss_d_total' in locals() else float('nan')
                        progress_callback.emit(
                            f"[E {epoch + 1}, B {i + 1}] G Loss: {loss_g_total.item():.4f} (Adv: {adv_loss.item():.4f}, FM: {fm_loss.item():.4f}, STFT: {stft_loss.item():.4f}) | D Loss: {d_loss_item:.4f}"
                        )
            
            progress_callback.emit(f"--- Epoch {epoch + 1} finished ---")

        if not stop_event.is_set():
            progress_callback.emit("Training finished. Saving Generator...")
            torch.save(generator.state_dict(), model_save_path)
            progress_callback.emit(f"Generator saved to {model_save_path}")
    except Exception as e:
        progress_callback.emit(f"ERROR in training: {e}")
        progress_callback.emit(traceback.format_exc())
