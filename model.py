# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.optim as optim
# # # # from torch.utils.data import Dataset, DataLoader
# # # # import torchaudio
# # # # import os
# # # # import numpy as np
# # # # import torch.nn.functional as F
# # # # import math

# # # # # --- Perceptual Loss Function ---
# # # # class MultiResolutionSTFTLoss(nn.Module):
# # # #     """
# # # #     Multi-resolution STFT loss, common in audio generation models.
# # # #     This is a key part of achieving high quality (PESQ/STOI).
# # # #     """
# # # #     def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
# # # #         super(MultiResolutionSTFTLoss, self).__init__()
# # # #         self.fft_sizes = fft_sizes
# # # #         self.hop_sizes = hop_sizes
# # # #         self.win_lengths = win_lengths
# # # #         self.window = torch.hann_window
# # # #         assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

# # # #     def forward(self, y_hat, y):
# # # #         sc_loss, mag_loss = 0.0, 0.0
# # # #         for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
# # # #             window = self.window(win, device=y.device)
# # # #             spec_hat = torch.stft(y_hat.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
# # # #             spec = torch.stft(y.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            
# # # #             sc_loss += torch.norm(torch.abs(spec) - torch.abs(spec_hat), p='fro') / torch.norm(torch.abs(spec), p='fro')
# # # #             mag_loss += F.l1_loss(torch.log(torch.abs(spec).clamp(min=1e-9)), torch.log(torch.abs(spec_hat).clamp(min=1e-9)))
            
# # # #         return (sc_loss / len(self.fft_sizes)) + (mag_loss / len(self.fft_sizes))

# # # # # --- Causal Convolution ---
# # # # class CausalConv1d(nn.Conv1d):
# # # #     """
# # # #     A 1D convolution that is causal (cannot see the future).
# # # #     This is critical for the < 20ms latency goal.
# # # #     """
# # # #     def __init__(self, *args, **kwargs):
# # # #         super().__init__(*args, **kwargs)
# # # #         # Calculate the padding needed to make it causal
# # # #         self.causal_padding = self.kernel_size[0] - 1

# # # #     def forward(self, x):
# # # #         # Pad on the left (past) only
# # # #         return super().forward(F.pad(x, (self.causal_padding, 0)))

# # # # class CausalConvTranspose1d(nn.ConvTranspose1d):
# # # #     """
# # # #     A 1D *transpose* convolution that is causal.
# # # #     It removes output samples that would "see the future".
# # # #     """
# # # #     def __init__(self, *args, **kwargs):
# # # #         super().__init__(*args, **kwargs)
# # # #         self.causal_padding = self.kernel_size[0] - self.stride[0]

# # # #     def forward(self, x):
# # # #         x = super().forward(x)
# # # #         # Remove the invalid, "future-seeing" samples from the end
# # # #         if self.causal_padding != 0:
# # # #             return x[..., :-self.causal_padding]
# # # #         return x

# # # # # --- Vector Quantizer (The heart of the *COMPRESSION*) ---
# # # # class VectorQuantizer(nn.Module):
# # # #     """
# # # #     The Vector Quantizer (VQ) module. This is what enables low-bitrate compression.
# # # #     It maps continuous latent vectors to a discrete set of "codes" from a codebook.
# # # #     """
# # # #     def __init__(self, num_embeddings, embedding_dim, commitment_cost):
# # # #         super(VectorQuantizer, self).__init__()
# # # #         self.num_embeddings = num_embeddings # Codebook size (e.g., 256)
# # # #         self.embedding_dim = embedding_dim # Dimension of each code
# # # #         self.commitment_cost = commitment_cost # 'beta' in VQ-VAE
        
# # # #         # The codebook
# # # #         self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
# # # #         self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

# # # #     def forward(self, z_e):
# # # #         # z_e shape: (B, C, T) -> (B*T, C)
# # # #         z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        
# # # #         # Find the closest codebook vector (L2 distance)
# # # #         distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True) 
# # # #                      + torch.sum(self.embedding.weight**2, dim=1)
# # # #                      - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))
        
# # # #         # Get the indices of the closest vectors
# # # #         encoding_indices = torch.argmin(distances, dim=1)
        
# # # #         # Quantize: Map indices back to codebook vectors
# # # #         z_q = self.embedding(encoding_indices).view(z_e.shape[0], -1, self.embedding_dim)
# # # #         z_q = z_q.permute(0, 2, 1).contiguous() # (B, C, T)

# # # #         # VQ-VAE Loss (Commitment Loss)
# # # #         e_loss = F.mse_loss(z_q.detach(), z_e) * self.commitment_cost
# # # #         q_loss = F.mse_loss(z_q, z_e.detach())
# # # #         vq_loss = q_loss + e_loss
        
# # # #         # Straight-Through Estimator (STE)
# # # #         # This copies the gradient from z_q to z_e
# # # #         z_q = z_e + (z_q - z_e).detach()
        
# # # #         return z_q, vq_loss, encoding_indices.view(z_e.shape[0], -1) # (B, T)

# # # # # --- The New Codec Architecture Components ---
# # # # # These are based on the SoundStream model, but simplified.

# # # # HOP_SIZE = 320 # 20ms frame (320 samples / 16000 Hz = 0.02s)
# # # # LATENT_DIM = 64
# # # # VQ_EMBEDDINGS = 256 # 8 bits per code
# # # # # 16000 bits/sec / 50 frames/sec = 320 bits/frame
# # # # # 320 bits / 8 bits/index = 40 indices per frame
# # # # NUM_QUANTIZERS = 40 # This is our 16kbps target (40 bytes * 50 fps = 2000 B/s = 16 kbps)

# # # # class Encoder(nn.Module):
# # # #     """
# # # #     Causal encoder. Takes raw audio and produces latent vectors.
# # # #     Takes a 320-sample chunk and produces 40 latent vectors.
# # # #     Total stride must be 320 / 40 = 8.
# # # #     """
# # # #     def __init__(self):
# # # #         super().__init__()
# # # #         self.net = nn.Sequential(
# # # #             CausalConv1d(1, 32, 7), nn.ELU(),
# # # #             CausalConv1d(32, 64, 5, stride=2), nn.ELU(), # 320 -> 160
# # # #             CausalConv1d(64, 64, 5, stride=2), nn.ELU(), # 160 -> 80
# # # #             CausalConv1d(64, LATENT_DIM, 5, stride=2), nn.ELU() # 80 -> 40
# # # #         )
# # # #         # Output shape: (B, LATENT_DIM, 40)
# # # #         # This is exactly what we need.

# # # #     def forward(self, x):
# # # #         return self.net(x)

# # # # class Decoder(nn.Module):
# # # #     """
# # # #     Causal decoder. Takes quantized latents and reconstructs audio.
# # # #     Must be the inverse of the Encoder.
# # # #     """
# # # #     def __init__(self):
# # # #         super().__init__()
# # # #         self.net = nn.Sequential(
# # # #             CausalConvTranspose1d(LATENT_DIM, 64, 5, stride=2), nn.ELU(), # 40 -> 80
# # # #             CausalConvTranspose1d(64, 64, 5, stride=2), nn.ELU(), # 80 -> 160
# # # #             CausalConvTranspose1d(64, 32, 5, stride=2), nn.ELU(), # 160 -> 320
# # # #             CausalConv1d(32, 1, 7), nn.Tanh() # Final output
# # # #         )
    
# # # #     def forward(self, x):
# # # #         return self.net(x)

# # # # # --- Causal Transformer (for the Transformer-based Codec) ---
# # # # class CausalTransformerEncoder(nn.Module):
# # # #     def __init__(self, d_model, nhead, num_layers):
# # # #         super().__init__()
# # # #         self.d_model = d_model
# # # #         layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
# # # #         self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
    
# # # #     def get_causal_mask(self, sz):
# # # #         # Returns a mask of shape (sz, sz)
# # # #         return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# # # #     def forward(self, x, state):
# # # #         # x shape: (B, T, C) e.g. (1, 40, 64)
# # # #         # state shape: (B, S, C) e.g. (1, 100, 64)
        
# # # #         if state is None:
# # # #             # First frame, no state
# # # #             inp = x
# # # #         else:
# # # #             # Append new frame to old state
# # # #             inp = torch.cat([state, x], dim=1)
        
# # # #         # We must limit the state size to avoid OOM
# # # #         # Let's say, 10 frames of history (10 * 40 = 400 steps)
# # # #         STATE_LEN = 400
# # # #         if inp.shape[1] > STATE_LEN:
# # # #             inp = inp[:, -STATE_LEN:, :]
        
# # # #         new_state = inp.detach() # The new state is the full input
        
# # # #         # Create a causal mask for the *full input sequence*
# # # #         mask = self.get_causal_mask(inp.shape[1]).to(x.device)
        
# # # #         # Process the full sequence
# # # #         out = self.transformer(inp, mask=mask)
        
# # # #         # Only return the *new* frames, corresponding to x
# # # #         # This is how we make it stateful
# # # #         out = out[:, -x.shape[1]:, :] # (B, T, C)
        
# # # #         return out, new_state


# # # # # --- MODEL 1: GRU Codec (Fast) ---
# # # # class GRU_Codec(nn.Module):
# # # #     """
# # # #     A stateful, causal codec using a GRU (RNN) as the core.
# # # #     This is the "simpler neural approach" and is very fast.
# # # #     """
# # # #     def __init__(self):
# # # #         super().__init__()
# # # #         self.encoder = Encoder()
# # # #         self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
# # # #         self.decoder = Decoder()
        
# # # #         self.encoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
# # # #         self.decoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
    
# # # #     def forward(self, x, h_enc=None, h_dec=None):
# # # #         # x: (B, 1, T)
# # # #         z_e = self.encoder(x) # (B, C, T_latent)
        
# # # #         # --- Stateful RNN ---
# # # #         # (B, C, T_latent) -> (B, T_latent, C)
# # # #         z_e_rnn_in = z_e.permute(0, 2, 1)
# # # #         z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
# # # #         # (B, T_latent, C) -> (B, C, T_latent)
# # # #         z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
# # # #         # ---
        
# # # #         z_q, vq_loss, indices = self.quantizer(z_e_rnn_out)

# # # #         # --- Stateful RNN ---
# # # #         # (B, C, T_latent) -> (B, T_latent, C)
# # # #         z_q_rnn_in = z_q.permute(0, 2, 1)
# # # #         z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
# # # #         # (B, T_latent, C) -> (B, C, T_latent)
# # # #         z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
# # # #         # ---
        
# # # #         x_hat = self.decoder(z_q_rnn_out)
        
# # # #         return x_hat, vq_loss, (h_enc_new, h_dec_new)

# # # #     def encode(self, x, h_enc):
# # # #         """For streaming: encode audio to indices."""
# # # #         z_e = self.encoder(x) # (B, C, 40)
# # # #         z_e_rnn_in = z_e.permute(0, 2, 1)
# # # #         z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
# # # #         z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
        
# # # #         # We don't need vq_loss here, just indices
# # # #         _, _, indices = self.quantizer(z_e_rnn_out) # (B, 40)
# # # #         return indices, h_enc_new

# # # #     def decode(self, indices, h_dec):
# # # #         """For streaming: decode indices to audio."""
# # # #         # Convert indices (B, 40) to codebook vectors (B, C, 40)
# # # #         z_q = self.quantizer.embedding(indices) # (B, 40, C)
# # # #         z_q = z_q.permute(0, 2, 1) # (B, C, 40)
        
# # # #         z_q_rnn_in = z_q.permute(0, 2, 1)
# # # #         z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
# # # #         z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
        
# # # #         x_hat = self.decoder(z_q_rnn_out) # (B, 1, 320)
# # # #         return x_hat, h_dec_new


# # # # # --- MODEL 2: TS3 Codec (Transformer) ---
# # # # class TS3_Codec(nn.Module):
# # # #     """
# # # #     A stateful, causal codec using a Causal Transformer as the core.
# # # #     This directly addresses your "Transformer" requirement.
# # # #     """
# # # #     def __init__(self):
# # # #         super().__init__()
# # # #         self.encoder = Encoder()
# # # #         self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
# # # #         self.decoder = Decoder()
        
# # # #         self.encoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
# # # #         self.decoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
    
# # # #     def forward(self, x, h_enc=None, h_dec=None):
# # # #         # x: (B, 1, T)
# # # #         z_e = self.encoder(x) # (B, C, T_latent)
        
# # # #         # --- Stateful TFM ---
# # # #         z_e_tfm_in = z_e.permute(0, 2, 1) # (B, T_latent, C)
# # # #         z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
# # # #         z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
# # # #         # ---
        
# # # #         z_q, vq_loss, indices = self.quantizer(z_e_tfm_out)

# # # #         # --- Stateful TFM ---
# # # #         z_q_tfm_in = z_q.permute(0, 2, 1) # (B, T_latent, C)
# # # #         z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q_tfm_in, h_dec)
# # # #         z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
# # # #         # ---
        
# # # #         x_hat = self.decoder(z_q_tfm_out)
        
# # # #         return x_hat, vq_loss, (h_enc_new, h_dec_new)

# # # #     def encode(self, x, h_enc):
# # # #         """For streaming: encode audio to indices."""
# # # #         z_e = self.encoder(x) # (B, C, 40)
# # # #         z_e_tfm_in = z_e.permute(0, 2, 1)
# # # #         z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
# # # #         z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1)
        
# # # #         _, _, indices = self.quantizer(z_e_tfm_out) # (B, 40)
# # # #         return indices, h_enc_new

# # # #     def decode(self, indices, h_dec):
# # # #         """For streaming: decode indices to audio."""
# # # #         z_q = self.quantizer.embedding(indices) # (B, 40, C)
        
# # # #         # TFM model input is (B, 40, C)
# # # #         z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q, h_dec)
# # # #         z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, 40)
        
# # # #         x_hat = self.decoder(z_q_tfm_out) # (B, 1, 320)
# # # #         return x_hat, h_dec_new

# # # # # --- MODEL 3: ScoreDec (Diffusion Post-Filter) ---

# # # # class SinusoidalPosEmb(nn.Module):
# # # #     def __init__(self, dim):
# # # #         super().__init__()
# # # #         self.dim = dim

# # # #     def forward(self, x):
# # # #         half_dim = self.dim // 2
# # # #         emb = math.log(10000) / (half_dim - 1)
# # # #         emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
# # # #         emb = x[:, None] * emb[None, :]
# # # #         return torch.cat((emb.sin(), emb.cos()), dim=-1)

# # # # class DiffusionBlock(nn.Module):
# # # #     """A single U-Net block for the diffusion model."""
# # # #     def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
# # # #         super().__init__()
# # # #         self.time_mlp = nn.Linear(time_emb_dim, out_channels)
# # # #         self.cond_mlp = nn.Linear(cond_dim, out_channels)
        
# # # #         self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, padding=1)
# # # #         self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, padding=1)
# # # #         self.bn1 = nn.BatchNorm1d(out_channels)
# # # #         self.bn2 = nn.BatchNorm1d(out_channels)
# # # #         self.relu = nn.ReLU()

# # # #     def forward(self, x, t_emb, cond_emb):
# # # #         h = self.relu(self.bn1(self.conv1(x)))
        
# # # #         # Add time and condition embeddings
# # # #         time_emb = self.relu(self.time_mlp(t_emb))
# # # #         cond_emb = self.relu(self.cond_mlp(cond_emb))
# # # #         h = h + time_emb.unsqueeze(-1) + cond_emb.unsqueeze(-1)
        
# # # #         h = self.relu(self.bn2(self.conv2(h)))
# # # #         return h

# # # # class DiffusionUNet1D(nn.Module):
# # # #     """A 1D U-Net for diffusion, conditioned on the low-quality codec output."""
# # # #     def __init__(self, in_channels=1, model_channels=64, time_emb_dim=256, cond_dim=1):
# # # #         super().__init__()
# # # #         self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU())
        
# # # #         # Conditioning projector
# # # #         self.cond_proj = nn.Conv1d(cond_dim, 64, 1)
        
# # # #         self.down1 = DiffusionBlock(in_channels, model_channels, time_emb_dim, 64)
# # # #         self.down2 = DiffusionBlock(model_channels, model_channels * 2, time_emb_dim, 64)
# # # #         self.bot = DiffusionBlock(model_channels * 2, model_channels * 2, time_emb_dim, 64)
# # # #         self.up1 = DiffusionBlock(model_channels * 3, model_channels, time_emb_dim, 64)
# # # #         self.out = CausalConv1d(model_channels, in_channels, kernel_size=1)
        
# # # #         self.pool = nn.MaxPool1d(2)
# # # #         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
# # # #         # --- Simpler ScoreDec Model ---
# # # #         # This part was incorrectly indented inside a 'forward' method.
# # # #         # It is now correctly placed inside '__init__'.
        
# # # #         self.time_mlp_simple = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, model_channels), nn.ReLU())
# # # #         self.cond_proj_simple = CausalConv1d(cond_dim, model_channels, 1)
# # # #         self.in_conv = CausalConv1d(in_channels, model_channels, 1)
        
# # # #         self.blocks = nn.ModuleList([
# # # #             CausalConv1d(model_channels, model_channels, 3, padding=1) for _ in range(4)
# # # #         ])
# # # #         self.out_conv = CausalConv1d(model_channels, in_channels, 1)

# # # #     # This buggy U-Net forward pass is unused and has been removed.
# # # #     # The comments inside it were moved to '__init__'.

# # # #     def forward_simple(self, x, time, cond):
# # # #         t_emb = self.time_mlp_simple(time).unsqueeze(-1) # (B, C, 1)
# # # #         c_emb = self.cond_proj_simple(cond) # (B, C, T)
# # # #         x = self.in_conv(x) # (B, C, T)
        
# # # #         h = x + t_emb + c_emb
# # # #         for block in self.blocks:
# # # #             h = block(h) + h # Residual
# # # #         return self.out_conv(h)

# # # #     # Let's stick to the simple forward
# # # #     def forward(self, x, time, cond):
# # # #         return self.forward_simple(x, time, cond)

# # # # class ScoreDecPostFilter(nn.Module):
# # # #     """
# # # #     Wraps the diffusion U-Net and provides the enhancement logic.
# # # #     This is what is called by the streaming/evaluation tabs.
# # # #     """
# # # #     def __init__(self, timesteps=50, model_channels=64):
# # # #         super().__init__()
# # # #         self.timesteps = timesteps
# # # #         self.model = DiffusionUNet1D(model_channels=model_channels)
        
# # # #         betas = torch.linspace(1e-4, 0.02, timesteps)
# # # #         alphas = 1. - betas
# # # #         alphas_cumprod = torch.cumprod(alphas, axis=0)

# # # #         self.register_buffer('betas', betas)
# # # #         self.register_buffer('alphas_cumprod', alphas_cumprod)
# # # #         self.register_buffer('alphas', alphas)
        
# # # #     def q_sample(self, x_start, t, noise=None):
# # # #         """Forward diffusion: noise the clean signal."""
# # # #         if noise is None: noise = torch.randn_like(x_start)
# # # #         sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(x_start.shape[0], 1, 1)
# # # #         sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().view(x_start.shape[0], 1, 1)
# # # #         return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# # # #     @torch.no_grad()
# # # #     def p_sample(self, x_t, t, cond):
# # # #         """One step of the reverse diffusion process."""
# # # #         t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
# # # #         alpha_t = self.alphas[t].view(-1, 1, 1)
# # # #         alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        
# # # #         predicted_noise = self.model(x_t, t_tensor.float(), cond)
        
# # # #         # DDPM sampling step
# # # #         x_prev = (x_t - ((1-alpha_t) / (1-alpha_cumprod_t).sqrt()) * predicted_noise) / alpha_t.sqrt()
        
# # # #         if t > 0:
# # # #             noise = torch.randn_like(x_t)
# # # #             alpha_cumprod_prev_t = self.alphas_cumprod[t-1]
# # # #             posterior_variance = (1-alpha_cumprod_prev_t) / (1-alpha_cumprod_t) * self.betas[t]
# # # #             x_prev += torch.sqrt(posterior_variance.view(-1, 1, 1)) * noise
# # # #         return x_prev

# # # #     @torch.no_grad()
# # # #     def enhance(self, x_low_quality, timesteps=10):
# # # #         """
# # # #         The main enhancement function.
# # # #         x_low_quality is the output from the GRU_Codec.
# # # #         This is SLOW and NOT real-time.
# # # #         """
# # # #         # Start from the low-quality audio, but add noise
# # # #         t_start = timesteps - 1
# # # #         x_t = self.q_sample(x_low_quality, torch.tensor([t_start], device=x_low_quality.device))
        
# # # #         for i in reversed(range(timesteps)):
# # # #             x_t = self.p_sample(x_t, i, cond=x_low_quality)
            
# # # #         return torch.tanh(x_t)


# # # # # --- TRADITIONAL CODECS (For Baseline Comparison) ---
# # # # class MuLawCodec:
# # # #     def __init__(self, quantization_channels=256): self.mu = float(quantization_channels - 1)
# # # #     def encode(self, x):
# # # #         mu_t = torch.tensor(self.mu, device=x.device, dtype=torch.float32)
# # # #         encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
# # # #         return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
# # # #     def decode(self, z):
# # # #         z_float = z.to(torch.float32)
# # # #         mu_t = torch.tensor(self.mu, device=z.device, dtype=torch.float32)
# # # #         y = (z_float / self.mu) * 2.0 - 1.0
# # # #         return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

# # # # class ALawCodec:
# # # #     def __init__(self): self.A = 87.6
# # # #     def encode(self, x):
# # # #         a_t = torch.tensor(self.A, device=x.device, dtype=torch.float32)
# # # #         abs_x = torch.abs(x)
# # # #         encoded = torch.zeros_like(x)
# # # #         cond = abs_x < (1 / self.A)
# # # #         encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
# # # #         encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
# # # #         return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
# # # #     def decode(self, z):
# # # #         z_float = z.to(torch.float32)
# # # #         a_t = torch.tensor(self.A, device=z.device, dtype=torch.float32)
# # # #         y = (z_float / 127.5) - 1.0
# # # #         abs_y = torch.abs(y)
# # # #         decoded = torch.zeros_like(y)
# # # #         cond = abs_y < (1 / (1 + torch.log(a_t)))
# # # #         decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
# # # #         decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
# # # #         return decoded.unsqueeze(1)

# # # # # --- DATASET & TRAINING ---
# # # # TRAIN_CHUNK_SIZE = 16000 # 1 second

# # # # class AudioChunkDataset(Dataset):
# # # #     def __init__(self, directory, chunk_size=TRAIN_CHUNK_SIZE, sample_rate=16000):
# # # #         self.chunk_size, self.sample_rate = chunk_size, sample_rate
# # # #         self.file_paths = [os.path.join(r, f) for r, _, fs in os.walk(directory) for f in fs if f.lower().endswith(('.wav', '.flac'))]
# # # #         if not self.file_paths: raise ValueError("No audio files found.")
# # # #     def __len__(self): return len(self.file_paths)
# # # #     def __getitem__(self, idx):
# # # #         try:
# # # #             waveform, sr = torchaudio.load(self.file_paths[idx])
# # # #             if sr != self.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
# # # #             if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
# # # #             if waveform.shape[1] > self.chunk_size:
# # # #                 start = np.random.randint(0, waveform.shape[1] - self.chunk_size)
# # # #                 waveform = waveform[:, start:start + self.chunk_size]
# # # #             else:
# # # #                 waveform = F.pad(waveform, (0, self.chunk_size - waveform.shape[1]))
# # # #             return waveform
# # # #         except Exception as e:
# # # #             print(f"Warning: Skipping file {self.file_paths[idx]}. Error: {e}")
# # # #             return torch.zeros((1, self.chunk_size))

# # # # def train_model(dataset_path, epochs, learning_rate, batch_size, model_save_path, progress_callback, stop_event, model_type):
# # # #     """
# # # #     Main training function. Now handles 'gru', 'transformer', and 'scoredec' types.
# # # #     """
# # # #     try:
# # # #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #         progress_callback.emit(f"Using device: {device}")
        
# # # #         dataset = AudioChunkDataset(directory=dataset_path)
# # # #         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# # # #         progress_callback.emit(f"Dataset loaded with {len(dataset)} files.")

# # # #         if model_type == 'gru':
# # # #             model = GRU_Codec().to(device)
# # # #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# # # #             stft_criterion = MultiResolutionSTFTLoss().to(device)
# # # #             l1_criterion = nn.L1Loss().to(device)
# # # #             progress_callback.emit(f"Starting training for GRU_Codec model...")
        
# # # #         elif model_type == 'transformer':
# # # #             model = TS3_Codec().to(device)
# # # #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# # # #             stft_criterion = MultiResolutionSTFTLoss().to(device)
# # # #             l1_criterion = nn.L1Loss().to(device)
# # # #             progress_callback.emit(f"Starting training for TS3_Codec model...")
        
# # # #         elif model_type == 'scoredec':
# # # #             progress_callback.emit("--- Starting ScoreDec Post-Filter Training ---")
# # # #             progress_callback.emit("Loading pre-trained GRU_Codec...")
# # # #             try:
# # # #                 gru_codec = GRU_Codec().to(device)
# # # #                 gru_codec.load_state_dict(torch.load("low_latency_codec_gru.pth", map_location=device))
# # # #                 gru_codec.eval()
# # # #                 for param in gru_codec.parameters():
# # # #                     param.requires_grad = False
# # # #                 progress_callback.emit("GRU_Codec loaded and frozen.")
# # # #             except FileNotFoundError:
# # # #                 progress_callback.emit("ERROR: 'low_latency_codec_gru.pth' not found. You must train the GRU_Codec first.")
# # # #                 return
            
# # # #             model = ScoreDecPostFilter().to(device)
# # # #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# # # #             l1_criterion = nn.L1Loss().to(device)
# # # #             progress_callback.emit("Starting training for ScoreDec model...")
            
# # # #         else:
# # # #             raise ValueError(f"Unknown model type for training: {model_type}")

# # # #         # --- Main Training Loop ---
# # # #         for epoch in range(epochs):
# # # #             if stop_event.is_set():
# # # #                 progress_callback.emit("Training stopped by user."); break
            
# # # #             for i, data in enumerate(dataloader):
# # # #                 inputs = data.to(device)
# # # #                 optimizer.zero_grad()
                
# # # #                 if model_type in ['gru', 'transformer']:
# # # #                     h_enc, h_dec = None, None # Reset state per batch
# # # #                     x_hat, vq_loss, (h_enc, h_dec) = model(inputs, h_enc, h_dec)
# # # #                     x_hat = x_hat[..., :inputs.shape[-1]]
                    
# # # #                     stft_loss = stft_criterion(x_hat, inputs)
# # # #                     l1_loss = l1_criterion(x_hat, inputs)
# # # #                     loss = stft_loss + 0.1 * l1_loss + vq_loss
                    
# # # #                     if i % 20 == 19:
# # # #                         progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss.item():.5f} (STFT: {stft_loss.item():.4f}, VQ: {vq_loss.item():.4f})")

# # # #                 elif model_type == 'scoredec':
# # # #                     with torch.no_grad():
# # # #                         x_hat_low_quality, _, _ = gru_codec(inputs)
# # # #                         x_hat_low_quality = x_hat_low_quality.detach()
                    
# # # #                     # Train diffusion model
# # # #                     t = torch.randint(0, model.timesteps, (inputs.shape[0],), device=device).long()
# # # #                     x_t, noise = model.q_sample(x_start=inputs, t=t)
                    
# # # #                     predicted_noise = model.model(x_t, t.float(), cond=x_hat_low_quality)
# # # #                     loss = l1_criterion(predicted_noise, noise)
                    
# # # #                     if i % 20 == 19:
# # # #                         progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Denoising Loss: {loss.item():.5f}")

# # # #                 loss.backward()
# # # #                 optimizer.step()

# # # #             progress_callback.emit(f"--- Epoch {epoch + 1} finished ---")

# # # #         if not stop_event.is_set():
# # # #             progress_callback.emit("Training finished. Saving model...")
# # # #             torch.save(model.state_dict(), model_save_path)
# # # #             progress_callback.emit(f"Model saved to {model_save_path}")
# # # #     except Exception as e:
# # # #         progress_callback.emit(f"ERROR in training: {e}")

# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.data import Dataset, DataLoader
# # # import torchaudio
# # # import os
# # # import numpy as np
# # # import torch.nn.functional as F
# # # import math

# # # # --- Perceptual Loss Function ---
# # # class MultiResolutionSTFTLoss(nn.Module):
# # #     """
# # #     Multi-resolution STFT loss, common in audio generation models.
# # #     This is a key part of achieving high quality (PESQ/STOI).
# # #     """
# # #     def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
# # #         super(MultiResolutionSTFTLoss, self).__init__()
# # #         self.fft_sizes = fft_sizes
# # #         self.hop_sizes = hop_sizes
# # #         self.win_lengths = win_lengths
# # #         self.window = torch.hann_window
# # #         assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

# # #     def forward(self, y_hat, y):
# # #         sc_loss, mag_loss = 0.0, 0.0
# # #         for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
# # #             window = self.window(win, device=y.device)
# # #             spec_hat = torch.stft(y_hat.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
# # #             spec = torch.stft(y.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            
# # #             sc_loss += torch.norm(torch.abs(spec) - torch.abs(spec_hat), p='fro') / torch.norm(torch.abs(spec), p='fro')
# # #             mag_loss += F.l1_loss(torch.log(torch.abs(spec).clamp(min=1e-9)), torch.log(torch.abs(spec_hat).clamp(min=1e-9)))
            
# # #         return (sc_loss / len(self.fft_sizes)) + (mag_loss / len(self.fft_sizes))

# # # # --- Causal Convolution ---
# # # class CausalConv1d(nn.Conv1d):
# # #     """
# # #     A 1D convolution that is causal (cannot see the future).
# # #     This is critical for the < 20ms latency goal.
# # #     """
# # #     def __init__(self, *args, **kwargs):
# # #         super().__init__(*args, **kwargs)
# # #         # Calculate the padding needed to make it causal
# # #         self.causal_padding = self.kernel_size[0] - 1

# # #     def forward(self, x):
# # #         # Pad on the left (past) only
# # #         return super().forward(F.pad(x, (self.causal_padding, 0)))

# # # class CausalConvTranspose1d(nn.ConvTranspose1d):
# # #     """
# # #     A 1D *transpose* convolution that is causal.
# # #     It removes output samples that would "see the future".
# # #     """
# # #     def __init__(self, *args, **kwargs):
# # #         super().__init__(*args, **kwargs)
# # #         self.causal_padding = self.kernel_size[0] - self.stride[0]

# # #     def forward(self, x):
# # #         x = super().forward(x)
# # #         # Remove the invalid, "future-seeing" samples from the end
# # #         if self.causal_padding != 0:
# # #             return x[..., :-self.causal_padding]
# # #         return x

# # # # --- Vector Quantizer (The heart of the *COMPRESSION*) ---
# # # class VectorQuantizer(nn.Module):
# # #     """
# # #     The Vector Quantizer (VQ) module. This is what enables low-bitrate compression.
# # #     It maps continuous latent vectors to a discrete set of "codes" from a codebook.
# # #     """
# # #     def __init__(self, num_embeddings, embedding_dim, commitment_cost):
# # #         super(VectorQuantizer, self).__init__()
# # #         self.num_embeddings = num_embeddings # Codebook size (e.g., 256)
# # #         self.embedding_dim = embedding_dim # Dimension of each code
# # #         self.commitment_cost = commitment_cost # 'beta' in VQ-VAE
        
# # #         # The codebook
# # #         self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
# # #         self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

# # #     def forward(self, z_e):
# # #         # z_e shape: (B, C, T) -> (B*T, C)
# # #         z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        
# # #         # Find the closest codebook vector (L2 distance)
# # #         distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True) 
# # #                      + torch.sum(self.embedding.weight**2, dim=1)
# # #                      - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))
        
# # #         # Get the indices of the closest vectors
# # #         encoding_indices = torch.argmin(distances, dim=1)
        
# # #         # Quantize: Map indices back to codebook vectors
# # #         z_q = self.embedding(encoding_indices).view(z_e.shape[0], -1, self.embedding_dim)
# # #         z_q = z_q.permute(0, 2, 1).contiguous() # (B, C, T)

# # #         # VQ-VAE Loss (Commitment Loss)
# # #         e_loss = F.mse_loss(z_q.detach(), z_e) * self.commitment_cost
# # #         q_loss = F.mse_loss(z_q, z_e.detach())
# # #         vq_loss = q_loss + e_loss
        
# # #         # Straight-Through Estimator (STE)
# # #         # This copies the gradient from z_q to z_e
# # #         z_q = z_e + (z_q - z_e).detach()
        
# # #         return z_q, vq_loss, encoding_indices.view(z_e.shape[0], -1) # (B, T)

# # # # --- The New Codec Architecture Components ---
# # # # These are based on the SoundStream model, but simplified.

# # # HOP_SIZE = 320 # 20ms frame (320 samples / 16000 Hz = 0.02s)
# # # LATENT_DIM = 64
# # # VQ_EMBEDDINGS = 256 # 8 bits per code
# # # # 16000 bits/sec / 50 frames/sec = 320 bits/frame
# # # # 320 bits / 8 bits/index = 40 indices per frame
# # # NUM_QUANTIZERS = 40 # This is our 16kbps target (40 bytes * 50 fps = 2000 B/s = 16 kbps)

# # # class Encoder(nn.Module):
# # #     """
# # #     Causal encoder. Takes raw audio and produces latent vectors.
# # #     Takes a 320-sample chunk and produces 40 latent vectors.
# # #     Total stride must be 320 / 40 = 8.
# # #     """
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.net = nn.Sequential(
# # #             CausalConv1d(1, 32, 7), nn.ELU(),
# # #             CausalConv1d(32, 64, 5, stride=2), nn.ELU(), # 320 -> 160
# # #             CausalConv1d(64, 64, 5, stride=2), nn.ELU(), # 160 -> 80
# # #             CausalConv1d(64, LATENT_DIM, 5, stride=2), nn.ELU() # 80 -> 40
# # #         )
# # #         # Output shape: (B, LATENT_DIM, 40)
# # #         # This is exactly what we need.

# # #     def forward(self, x):
# # #         return self.net(x)

# # # class Decoder(nn.Module):
# # #     """
# # #     Causal decoder. Takes quantized latents and reconstructs audio.
# # #     Must be the inverse of the Encoder.
# # #     """
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.net = nn.Sequential(
# # #             CausalConvTranspose1d(LATENT_DIM, 64, 5, stride=2), nn.ELU(), # 40 -> 80
# # #             CausalConvTranspose1d(64, 64, 5, stride=2), nn.ELU(), # 80 -> 160
# # #             CausalConvTranspose1d(64, 32, 5, stride=2), nn.ELU(), # 160 -> 320
# # #             CausalConv1d(32, 1, 7), nn.Tanh() # Final output
# # #         )
    
# # #     def forward(self, x):
# # #         return self.net(x)

# # # # --- Causal Transformer (for the Transformer-based Codec) ---
# # # class CausalTransformerEncoder(nn.Module):
# # #     def __init__(self, d_model, nhead, num_layers):
# # #         super().__init__()
# # #         self.d_model = d_model
# # #         layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
# # #         self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
    
# # #     def get_causal_mask(self, sz):
# # #         # Returns a mask of shape (sz, sz)
# # #         return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# # #     def forward(self, x, state):
# # #         # x shape: (B, T, C) e.g. (1, 40, 64)
# # #         # state shape: (B, S, C) e.g. (1, 100, 64)
        
# # #         if state is None:
# # #             # First frame, no state
# # #             inp = x
# # #         else:
# # #             # Append new frame to old state
# # #             inp = torch.cat([state, x], dim=1)
        
# # #         # We must limit the state size to avoid OOM
# # #         # Let's say, 10 frames of history (10 * 40 = 400 steps)
# # #         STATE_LEN = 400
# # #         if inp.shape[1] > STATE_LEN:
# # #             inp = inp[:, -STATE_LEN:, :]
        
# # #         new_state = inp.detach() # The new state is the full input
        
# # #         # Create a causal mask for the *full input sequence*
# # #         mask = self.get_causal_mask(inp.shape[1]).to(x.device)
        
# # #         # Process the full sequence
# # #         out = self.transformer(inp, mask=mask)
        
# # #         # Only return the *new* frames, corresponding to x
# # #         # This is how we make it stateful
# # #         out = out[:, -x.shape[1]:, :] # (B, T, C)
        
# # #         return out, new_state


# # # # --- MODEL 1: GRU Codec (Fast) ---
# # # class GRU_Codec(nn.Module):
# # #     """
# # #     A stateful, causal codec using a GRU (RNN) as the core.
# # #     This is the "simpler neural approach" and is very fast.
# # #     """
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.encoder = Encoder()
# # #         self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
# # #         self.decoder = Decoder()
        
# # #         self.encoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
# # #         self.decoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
    
# # #     def forward(self, x, h_enc=None, h_dec=None):
# # #         # x: (B, 1, T)
# # #         z_e = self.encoder(x) # (B, C, T_latent)
        
# # #         # --- Stateful RNN ---
# # #         # (B, C, T_latent) -> (B, T_latent, C)
# # #         z_e_rnn_in = z_e.permute(0, 2, 1)
# # #         z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
# # #         # (B, T_latent, C) -> (B, C, T_latent)
# # #         z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
# # #         # ---
        
# # #         z_q, vq_loss, indices = self.quantizer(z_e_rnn_out)

# # #         # --- Stateful RNN ---
# # #         # (B, C, T_latent) -> (B, T_latent, C)
# # #         z_q_rnn_in = z_q.permute(0, 2, 1)
# # #         z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
# # #         # (B, T_latent, C) -> (B, C, T_latent)
# # #         z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
# # #         # ---
        
# # #         x_hat = self.decoder(z_q_rnn_out)
        
# # #         return x_hat, vq_loss, (h_enc_new, h_dec_new)

# # #     def encode(self, x, h_enc):
# # #         """For streaming: encode audio to indices."""
# # #         z_e = self.encoder(x) # (B, C, 40)
# # #         z_e_rnn_in = z_e.permute(0, 2, 1)
# # #         z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
# # #         z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
        
# # #         # We don't need vq_loss here, just indices
# # #         _, _, indices = self.quantizer(z_e_rnn_out) # (B, 40)
# # #         return indices, h_enc_new

# # #     def decode(self, indices, h_dec):
# # #         """For streaming: decode indices to audio."""
# # #         # Convert indices (B, 40) to codebook vectors (B, C, 40)
# # #         z_q = self.quantizer.embedding(indices) # (B, 40, C)
# # #         z_q = z_q.permute(0, 2, 1) # (B, C, 40)
        
# # #         z_q_rnn_in = z_q.permute(0, 2, 1)
# # #         z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
# # #         z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
        
# # #         x_hat = self.decoder(z_q_rnn_out) # (B, 1, 320)
# # #         return x_hat, h_dec_new


# # # # --- MODEL 2: TS3 Codec (Transformer) ---
# # # class TS3_Codec(nn.Module):
# # #     """
# # #     A stateful, causal codec using a Causal Transformer as the core.
# # #     This directly addresses your "Transformer" requirement.
# # #     """
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.encoder = Encoder()
# # #         self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
# # #         self.decoder = Decoder()
        
# # #         self.encoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
# # #         self.decoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
    
# # #     def forward(self, x, h_enc=None, h_dec=None):
# # #         # x: (B, 1, T)
# # #         z_e = self.encoder(x) # (B, C, T_latent)
        
# # #         # --- Stateful TFM ---
# # #         z_e_tfm_in = z_e.permute(0, 2, 1) # (B, T_latent, C)
# # #         z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
# # #         z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
# # #         # ---
        
# # #         z_q, vq_loss, indices = self.quantizer(z_e_tfm_out)

# # #         # --- Stateful TFM ---
# # #         z_q_tfm_in = z_q.permute(0, 2, 1) # (B, T_latent, C)
# # #         z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q_tfm_in, h_dec)
# # #         z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
# # #         # ---
        
# # #         x_hat = self.decoder(z_q_tfm_out)
        
# # #         return x_hat, vq_loss, (h_enc_new, h_dec_new)

# # #     def encode(self, x, h_enc):
# # #         """For streaming: encode audio to indices."""
# # #         z_e = self.encoder(x) # (B, C, 40)
# # #         z_e_tfm_in = z_e.permute(0, 2, 1)
# # #         z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
# # #         z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1)
        
# # #         _, _, indices = self.quantizer(z_e_tfm_out) # (B, 40)
# # #         return indices, h_enc_new

# # #     def decode(self, indices, h_dec):
# # #         """For streaming: decode indices to audio."""
# # #         z_q = self.quantizer.embedding(indices) # (B, 40, C)
        
# # #         # TFM model input is (B, 40, C)
# # #         z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q, h_dec)
# # #         z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, 40)
        
# # #         x_hat = self.decoder(z_q_tfm_out) # (B, 1, 320)
# # #         return x_hat, h_dec_new

# # # # --- MODEL 3: ScoreDec (Diffusion Post-Filter) ---

# # # class SinusoidalPosEmb(nn.Module):
# # #     def __init__(self, dim):
# # #         super().__init__()
# # #         self.dim = dim

# # #     def forward(self, x):
# # #         half_dim = self.dim // 2
# # #         emb = math.log(10000) / (half_dim - 1)
# # #         emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
# # #         emb = x[:, None] * emb[None, :]
# # #         return torch.cat((emb.sin(), emb.cos()), dim=-1)

# # # class DiffusionBlock(nn.Module):
# # #     """A single U-Net block for the diffusion model."""
# # #     def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
# # #         super().__init__()
# # #         self.time_mlp = nn.Linear(time_emb_dim, out_channels)
# # #         self.cond_mlp = nn.Linear(cond_dim, out_channels)
        
# # #         self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, padding=1)
# # #         self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, padding=1)
# # #         self.bn1 = nn.BatchNorm1d(out_channels)
# # #         self.bn2 = nn.BatchNorm1d(out_channels)
# # #         self.relu = nn.ReLU()

# # #     def forward(self, x, t_emb, cond_emb):
# # #         h = self.relu(self.bn1(self.conv1(x)))
        
# # #         # Add time and condition embeddings
# # #         time_emb = self.relu(self.time_mlp(t_emb))
# # #         cond_emb = self.relu(self.cond_mlp(cond_emb))
# # #         h = h + time_emb.unsqueeze(-1) + cond_emb.unsqueeze(-1)
        
# # #         h = self.relu(self.bn2(self.conv2(h)))
# # #         return h

# # # class DiffusionUNet1D(nn.Module):
# # #     """
# # #     A 1D *CAUSAL* Denoising model, conditioned on the low-quality codec output.
# # #     This is a simple "WaveNet" style stack, not a U-Net, to maintain causality.
# # #     """
# # #     def __init__(self, in_channels=1, model_channels=64, time_emb_dim=256, cond_dim=1):
# # #         super().__init__()
        
# # #         # --- Simple Causal Stack ---
# # #         self.time_mlp_simple = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, model_channels), nn.ReLU())
# # #         self.cond_proj_simple = CausalConv1d(cond_dim, model_channels, 1)
# # #         self.in_conv = CausalConv1d(in_channels, model_channels, 1)
        
# # #         self.blocks = nn.ModuleList([
# # #             CausalConv1d(model_channels, model_channels, 3, padding=1) for _ in range(4) # 4 residual blocks
# # #         ])
# # #         self.out_conv = CausalConv1d(model_channels, in_channels, 1)
# # #         # --- End Simple Causal Stack ---

# # #     def forward(self, x, time, cond):
# # #         # t_emb shape: (B, C, 1)
# # #         t_emb = self.time_mlp_simple(time).unsqueeze(-1) 
# # #         # c_emb shape: (B, C, T)
# # #         c_emb = self.cond_proj_simple(cond) 
# # #         # x_in shape: (B, C, T)
# # #         x_in = self.in_conv(x) 
        
# # #         # Add time and condition embeddings
# # #         h = x_in + t_emb + c_emb 
# # #         for block in self.blocks:
# # #             h = block(h) + h # Residual connection
# # #         return self.out_conv(h)

# # # class ScoreDecPostFilter(nn.Module):
# # #     """
# # #     Wraps the diffusion U-Net and provides the enhancement logic.
# # #     This is what is called by the streaming/evaluation tabs.
# # #     """
# # #     def __init__(self, timesteps=50, model_channels=64):
# # #         super().__init__()
# # #         self.timesteps = timesteps
# # #         self.model = DiffusionUNet1D(model_channels=model_channels)
        
# # #         betas = torch.linspace(1e-4, 0.02, timesteps)
# # #         alphas = 1. - betas
# # #         alphas_cumprod = torch.cumprod(alphas, axis=0)

# # #         self.register_buffer('betas', betas)
# # #         self.register_buffer('alphas_cumprod', alphas_cumprod)
# # #         self.register_buffer('alphas', alphas)
        
# # #     def q_sample(self, x_start, t, noise=None):
# # #         """Forward diffusion: noise the clean signal."""
# # #         if noise is None: noise = torch.randn_like(x_start)
# # #         sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(x_start.shape[0], 1, 1)
# # #         sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().view(x_start.shape[0], 1, 1)
# # #         return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# # #     @torch.no_grad()
# # #     def p_sample(self, x_t, t, cond):
# # #         """One step of the reverse diffusion process."""
# # #         t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
# # #         alpha_t = self.alphas[t].view(-1, 1, 1)
# # #         alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        
# # #         predicted_noise = self.model(x_t, t_tensor.float(), cond)
        
# # #         # DDPM sampling step
# # #         x_prev = (x_t - ((1-alpha_t) / (1-alpha_cumprod_t).sqrt()) * predicted_noise) / alpha_t.sqrt()
        
# # #         if t > 0:
# # #             noise = torch.randn_like(x_t)
# # #             alpha_cumprod_prev_t = self.alphas_cumprod[t-1]
# # #             posterior_variance = (1-alpha_cumprod_prev_t) / (1-alpha_cumprod_t) * self.betas[t]
# # #             x_prev += torch.sqrt(posterior_variance.view(-1, 1, 1)) * noise
# # #         return x_prev

# # #     @torch.no_grad()
# # #     def enhance(self, x_low_quality, timesteps=10):
# # #         """
# # #         The main enhancement function.
# # #         x_low_quality is the output from the GRU_Codec.
# # #         This is SLOW and NOT real-time.
# # #         """
# # #         # Start from the low-quality audio, but add noise
# # #         t_start = timesteps - 1
# # #         x_t = self.q_sample(x_low_quality, torch.tensor([t_start], device=x_low_quality.device))
        
# # #         for i in reversed(range(timesteps)):
# # #             x_t = self.p_sample(x_t, i, cond=x_low_quality)
            
# # #         return torch.tanh(x_t)


# # # # --- TRADITIONAL CODECS (For Baseline Comparison) ---
# # # class MuLawCodec:
# # #     def __init__(self, quantization_channels=256): self.mu = float(quantization_channels - 1)
# # #     def encode(self, x):
# # #         mu_t = torch.tensor(self.mu, device=x.device, dtype=torch.float32)
# # #         encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
# # #         return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
# # #     def decode(self, z):
# # #         z_float = z.to(torch.float32)
# # #         mu_t = torch.tensor(self.mu, device=z.device, dtype=torch.float32)
# # #         y = (z_float / self.mu) * 2.0 - 1.0
# # #         return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

# # # class ALawCodec:
# # #     def __init__(self): self.A = 87.6
# # #     def encode(self, x):
# # #         a_t = torch.tensor(self.A, device=x.device, dtype=torch.float32)
# # #         abs_x = torch.abs(x)
# # #         encoded = torch.zeros_like(x)
# # #         cond = abs_x < (1 / self.A)
# # #         encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
# # #         encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
# # #         return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
# # #     def decode(self, z):
# # #         z_float = z.to(torch.float32)
# # #         a_t = torch.tensor(self.A, device=z.device, dtype=torch.float32)
# # #         y = (z_float / 127.5) - 1.0
# # #         abs_y = torch.abs(y)
# # #         decoded = torch.zeros_like(y)
# # #         cond = abs_y < (1 / (1 + torch.log(a_t)))
# # #         decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
# # #         decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
# # #         return decoded.unsqueeze(1)

# # # # --- DATASET & TRAINING ---
# # # TRAIN_CHUNK_SIZE = 16000 # 1 second

# # # class AudioChunkDataset(Dataset):
# # #     def __init__(self, directory, chunk_size=TRAIN_CHUNK_SIZE, sample_rate=16000):
# # #         self.chunk_size, self.sample_rate = chunk_size, sample_rate
# # #         self.file_paths = [os.path.join(r, f) for r, _, fs in os.walk(directory) for f in fs if f.lower().endswith(('.wav', '.flac'))]
# # #         if not self.file_paths: raise ValueError("No audio files found.")
# # #     def __len__(self): return len(self.file_paths)
# # #     def __getitem__(self, idx):
# # #         try:
# # #             waveform, sr = torchaudio.load(self.file_paths[idx])
# # #             if sr != self.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
# # #             if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
# # #             if waveform.shape[1] > self.chunk_size:
# # #                 start = np.random.randint(0, waveform.shape[1] - self.chunk_size)
# # #                 waveform = waveform[:, start:start + self.chunk_size]
# # #             else:
# # #                 waveform = F.pad(waveform, (0, self.chunk_size - waveform.shape[1]))
# # #             return waveform
# # #         except Exception as e:
# # #             print(f"Warning: Skipping file {self.file_paths[idx]}. Error: {e}")
# # #             return torch.zeros((1, self.chunk_size))

# # # def train_model(dataset_path, epochs, learning_rate, batch_size, model_save_path, progress_callback, stop_event, model_type):
# # #     """
# # #     Main training function. Now handles 'gru', 'transformer', and 'scoredec' types.
# # #     """
# # #     try:
# # #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #         progress_callback.emit(f"Using device: {device}")
        
# # #         dataset = AudioChunkDataset(directory=dataset_path)
# # #         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# # #         progress_callback.emit(f"Dataset loaded with {len(dataset)} files.")

# # #         if model_type == 'gru':
# # #             model = GRU_Codec().to(device)
# # #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# # #             stft_criterion = MultiResolutionSTFTLoss().to(device)
# # #             l1_criterion = nn.L1Loss().to(device)
# # #             progress_callback.emit(f"Starting training for GRU_Codec model...")
        
# # #         elif model_type == 'transformer':
# # #             model = TS3_Codec().to(device)
# # #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# # #             stft_criterion = MultiResolutionSTFTLoss().to(device)
# # #             l1_criterion = nn.L1Loss().to(device)
# # #             progress_callback.emit(f"Starting training for TS3_Codec model...")
        
# # #         elif model_type == 'scoredec':
# # #             progress_callback.emit("--- Starting ScoreDec Post-Filter Training ---")
# # #             progress_callback.emit("Loading pre-trained GRU_Codec...")
# # #             try:
# # #                 gru_codec = GRU_Codec().to(device)
# # #                 gru_codec.load_state_dict(torch.load("low_latency_codec_gru.pth", map_location=device))
# # #                 gru_codec.eval()
# # #                 for param in gru_codec.parameters():
# # #                     param.requires_grad = False
# # #                 progress_callback.emit("GRU_Codec loaded and frozen.")
# # #             except FileNotFoundError:
# # #                 progress_callback.emit("ERROR: 'low_latency_codec_gru.pth' not found. You must train the GRU_Codec first.")
# # #                 return
            
# # #             model = ScoreDecPostFilter().to(device)
# # #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# # #             l1_criterion = nn.L1Loss().to(device)
# # #             progress_callback.emit("Starting training for ScoreDec model...")
            
# # #         else:
# # #             raise ValueError(f"Unknown model type for training: {model_type}")

# # #         # --- Main Training Loop ---
# # #         for epoch in range(epochs):
# # #             if stop_event.is_set():
# # #                 progress_callback.emit("Training stopped by user."); break
            
# # #             for i, data in enumerate(dataloader):
# # #                 inputs = data.to(device)
# # #                 optimizer.zero_grad()
                
# # #                 if model_type in ['gru', 'transformer']:
# # #                     h_enc, h_dec = None, None # Reset state per batch
# # #                     x_hat, vq_loss, (h_enc, h_dec) = model(inputs, h_enc, h_dec)
                    
# # #                     # --- FIX: Defensively pad output to match input length ---
# # #                     # This is the most likely source of the STFT error
# # #                     input_len = inputs.shape[-1]
# # #                     output_len = x_hat.shape[-1]
                    
# # #                     if output_len < input_len:
# # #                         # Pad x_hat on the right if it's shorter
# # #                         padding = input_len - output_len
# # #                         x_hat = F.pad(x_hat, (0, padding))
# # #                     elif output_len > input_len:
# # #                         # Trim x_hat if it's longer
# # #                         x_hat = x_hat[..., :input_len]
# # #                     # --- End Fix ---
                    
# # #                     stft_loss = stft_criterion(x_hat, inputs)
# # #                     l1_loss = l1_criterion(x_hat, inputs)
# # #                     loss = stft_loss + 0.1 * l1_loss + vq_loss
                    
# # #                     if i % 20 == 19:
# # #                         progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss.item():.5f} (STFT: {stft_loss.item():.4f}, VQ: {vq_loss.item():.4f})")

# # #                 elif model_type == 'scoredec':
# # #                     with torch.no_grad():
# # #                         x_hat_low_quality, _, _ = gru_codec(inputs)
                        
# # #                         # --- FIX: Ensure low-quality output matches input ---
# # #                         input_len = inputs.shape[-1]
# # #                         output_len = x_hat_low_quality.shape[-1]
# # #                         if output_len < input_len:
# # #                             padding = input_len - output_len
# # #                             x_hat_low_quality = F.pad(x_hat_low_quality, (0, padding))
# # #                         elif output_len > input_len:
# # #                             x_hat_low_quality = x_hat_low_quality[..., :input_len]
# # #                         # --- End Fix ---
                        
# # #                         x_hat_low_quality = x_hat_low_quality.detach()
                    
# # #                     # Train diffusion model
# # #                     t = torch.randint(0, model.timesteps, (inputs.shape[0],), device=device).long()
# # #                     x_t, noise = model.q_sample(x_start=inputs, t=t)
                    
# # #                     predicted_noise = model.model(x_t, t.float(), cond=x_hat_low_quality)
# # #                     loss = l1_criterion(predicted_noise, noise)
                    
# # #                     if i % 20 == 19:
# # #                         progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Denoising Loss: {loss.item():.5f}")

# # #                 loss.backward()
# # #                 optimizer.step()

# # #             progress_callback.emit(f"--- Epoch {epoch + 1} finished ---")

# # #         if not stop_event.is_set():
# # #             progress_callback.emit("Training finished. Saving model...")
# # #             torch.save(model.state_dict(), model_save_path)
# # #             progress_callback.emit(f"Model saved to {model_save_path}")
# # #     except Exception as e:
# # #         progress_callback.emit(f"ERROR in training: {e}")


# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import Dataset, DataLoader
# # import torchaudio
# # import os
# # import numpy as np
# # import torch.nn.functional as F
# # import math

# # # --- Perceptual Loss Function ---
# # class MultiResolutionSTFTLoss(nn.Module):
# #     """
# #     Multi-resolution STFT loss, common in audio generation models.
# #     This is a key part of achieving high quality (PESQ/STOI).
# #     """
# #     def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
# #         super(MultiResolutionSTFTLoss, self).__init__()
# #         self.fft_sizes = fft_sizes
# #         self.hop_sizes = hop_sizes
# #         self.win_lengths = win_lengths
# #         self.window = torch.hann_window
# #         assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

# #     def forward(self, y_hat, y):
# #         sc_loss, mag_loss = 0.0, 0.0
# #         for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
# #             window = self.window(win, device=y.device)
# #             spec_hat = torch.stft(y_hat.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
# #             spec = torch.stft(y.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            
# #             sc_loss += torch.norm(torch.abs(spec) - torch.abs(spec_hat), p='fro') / torch.norm(torch.abs(spec), p='fro')
# #             mag_loss += F.l1_loss(torch.log(torch.abs(spec).clamp(min=1e-9)), torch.log(torch.abs(spec_hat).clamp(min=1e-9)))
            
# #         return (sc_loss / len(self.fft_sizes)) + (mag_loss / len(self.fft_sizes))

# # # --- Causal Convolution ---
# # class CausalConv1d(nn.Conv1d):
# #     """
# #     A 1D convolution that is causal (cannot see the future).
# #     This is critical for the < 20ms latency goal.
# #     """
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         # Calculate the padding needed to make it causal
# #         self.causal_padding = self.kernel_size[0] - 1

# #     def forward(self, x):
# #         # Pad on the left (past) only
# #         return super().forward(F.pad(x, (self.causal_padding, 0)))

# # class CausalConvTranspose1d(nn.ConvTranspose1d):
# #     """
# #     A 1D *transpose* convolution that is causal.
# #     It removes output samples that would "see the future".
# #     """
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.causal_padding = self.kernel_size[0] - self.stride[0]

# #     def forward(self, x):
# #         x = super().forward(x)
# #         # Remove the invalid, "future-seeing" samples from the end
# #         if self.causal_padding != 0:
# #             return x[..., :-self.causal_padding]
# #         return x

# # # --- Vector Quantizer (The heart of the *COMPRESSION*) ---
# # class VectorQuantizer(nn.Module):
# #     """
# #     The Vector Quantizer (VQ) module. This is what enables low-bitrate compression.
# #     It maps continuous latent vectors to a discrete set of "codes" from a codebook.
# #     """
# #     def __init__(self, num_embeddings, embedding_dim, commitment_cost):
# #         super(VectorQuantizer, self).__init__()
# #         self.num_embeddings = num_embeddings # Codebook size (e.g., 256)
# #         self.embedding_dim = embedding_dim # Dimension of each code
# #         self.commitment_cost = commitment_cost # 'beta' in VQ-VAE
        
# #         # The codebook
# #         self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
# #         self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

# #     def forward(self, z_e):
# #         # z_e shape: (B, C, T) -> (B*T, C)
# #         z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        
# #         # Find the closest codebook vector (L2 distance)
# #         distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True) 
# #                      + torch.sum(self.embedding.weight**2, dim=1)
# #                      - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))
        
# #         # Get the indices of the closest vectors
# #         encoding_indices = torch.argmin(distances, dim=1)
        
# #         # Quantize: Map indices back to codebook vectors
# #         z_q = self.embedding(encoding_indices).view(z_e.shape[0], -1, self.embedding_dim)
# #         z_q = z_q.permute(0, 2, 1).contiguous() # (B, C, T)

# #         # VQ-VAE Loss (Commitment Loss)
# #         e_loss = F.mse_loss(z_q.detach(), z_e) * self.commitment_cost
# #         q_loss = F.mse_loss(z_q, z_e.detach())
# #         vq_loss = q_loss + e_loss
        
# #         # Straight-Through Estimator (STE)
# #         # This copies the gradient from z_q to z_e
# #         z_q = z_e + (z_q - z_e).detach()
        
# #         return z_q, vq_loss, encoding_indices.view(z_e.shape[0], -1) # (B, T)

# # # --- The New Codec Architecture Components ---
# # # These are based on the SoundStream model, but simplified.

# # HOP_SIZE = 320 # 20ms frame (320 samples / 16000 Hz = 0.02s)
# # LATENT_DIM = 64
# # VQ_EMBEDDINGS = 256 # 8 bits per code
# # # 16000 bits/sec / 50 frames/sec = 320 bits/frame
# # # 320 bits / 8 bits/index = 40 indices per frame
# # NUM_QUANTIZERS = 40 # This is our 16kbps target (40 bytes * 50 fps = 2000 B/s = 16 kbps)

# # class Encoder(nn.Module):
# #     """
# #     Causal encoder. Takes raw audio and produces latent vectors.
# #     Takes a 320-sample chunk and produces 40 latent vectors.
# #     Total stride must be 320 / 40 = 8.
# #     """
# #     def __init__(self):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             CausalConv1d(1, 32, 7), nn.ELU(),
# #             CausalConv1d(32, 64, 5, stride=2), nn.ELU(), # 320 -> 160
# #             CausalConv1d(64, 64, 5, stride=2), nn.ELU(), # 160 -> 80
# #             CausalConv1d(64, LATENT_DIM, 5, stride=2), nn.ELU() # 80 -> 40
# #         )
# #         # Output shape: (B, LATENT_DIM, 40)
# #         # This is exactly what we need.

# #     def forward(self, x):
# #         return self.net(x)

# # class Decoder(nn.Module):
# #     """
# #     Causal decoder. Takes quantized latents and reconstructs audio.
# #     Must be the inverse of the Encoder.
# #     """
# #     def __init__(self):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             CausalConvTranspose1d(LATENT_DIM, 64, 5, stride=2), nn.ELU(), # 40 -> 80
# #             CausalConvTranspose1d(64, 64, 5, stride=2), nn.ELU(), # 80 -> 160
# #             CausalConvTranspose1d(64, 32, 5, stride=2), nn.ELU(), # 160 -> 320
# #             CausalConv1d(32, 1, 7), nn.Tanh() # Final output
# #         )
    
# #     def forward(self, x):
# #         return self.net(x)

# # # --- Causal Transformer (for the Transformer-based Codec) ---
# # class CausalTransformerEncoder(nn.Module):
# #     def __init__(self, d_model, nhead, num_layers):
# #         super().__init__()
# #         self.d_model = d_model
# #         layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
# #         self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
    
# #     def get_causal_mask(self, sz):
# #         # Returns a mask of shape (sz, sz)
# #         # FIX: Create a boolean mask where True means "masked out"
# #         mask = torch.triu(torch.ones(sz, sz), diagonal=1) # (sz, sz) with 1s in upper triangle
# #         return mask.to(torch.bool) # Convert to boolean

# #     def forward(self, x, state):
# #         # x shape: (B, T, C) e.g. (1, 40, 64)
# #         # state shape: (B, S, C) e.g. (1, 100, 64)
        
# #         if state is None:
# #             # First frame, no state
# #             inp = x
# #         else:
# #             # Append new frame to old state
# #             inp = torch.cat([state, x], dim=1)
        
# #         # We must limit the state size to avoid OOM
# #         # Let's say, 10 frames of history (10 * 40 = 400 steps)
# #         STATE_LEN = 400
# #         if inp.shape[1] > STATE_LEN:
# #             inp = inp[:, -STATE_LEN:, :]
        
# #         new_state = inp.detach() # The new state is the full input
        
# #         # Create a causal mask for the *full input sequence*
# #         mask = self.get_causal_mask(inp.shape[1]).to(x.device)
        
# #         # Process the full sequence
# #         out = self.transformer(inp, mask=mask)
        
# #         # Only return the *new* frames, corresponding to x
# #         # This is how we make it stateful
# #         out = out[:, -x.shape[1]:, :] # (B, T, C)
        
# #         return out, new_state


# # # --- MODEL 1: GRU Codec (Fast) ---
# # class GRU_Codec(nn.Module):
# #     """
# #     A stateful, causal codec using a GRU (RNN) as the core.
# #     This is the "simpler neural approach" and is very fast.
# #     """
# #     def __init__(self):
# #         super().__init__()
# #         self.encoder = Encoder()
# #         self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
# #         self.decoder = Decoder()
        
# #         self.encoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
# #         self.decoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
    
# #     def forward(self, x, h_enc=None, h_dec=None):
# #         # x: (B, 1, T)
# #         z_e = self.encoder(x) # (B, C, T_latent)
        
# #         # --- Stateful RNN ---
# #         # (B, C, T_latent) -> (B, T_latent, C)
# #         z_e_rnn_in = z_e.permute(0, 2, 1)
# #         z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
# #         # (B, T_latent, C) -> (B, C, T_latent)
# #         z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
# #         # ---
        
# #         z_q, vq_loss, indices = self.quantizer(z_e_rnn_out)

# #         # --- Stateful RNN ---
# #         # (B, C, T_latent) -> (B, T_latent, C)
# #         z_q_rnn_in = z_q.permute(0, 2, 1)
# #         z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
# #         # (B, T_latent, C) -> (B, C, T_latent)
# #         z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
# #         # ---
        
# #         x_hat = self.decoder(z_q_rnn_out)
        
# #         return x_hat, vq_loss, (h_enc_new, h_dec_new)

# #     def encode(self, x, h_enc):
# #         """For streaming: encode audio to indices."""
# #         z_e = self.encoder(x) # (B, C, 40)
# #         z_e_rnn_in = z_e.permute(0, 2, 1)
# #         z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
# #         z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
        
# #         # We don't need vq_loss here, just indices
# #         _, _, indices = self.quantizer(z_e_rnn_out) # (B, 40)
# #         return indices, h_enc_new

# #     def decode(self, indices, h_dec):
# #         """For streaming: decode indices to audio."""
# #         # Convert indices (B, 40) to codebook vectors (B, C, 40)
# #         z_q = self.quantizer.embedding(indices) # (B, 40, C)
# #         z_q = z_q.permute(0, 2, 1) # (B, C, 40)
        
# #         z_q_rnn_in = z_q.permute(0, 2, 1)
# #         z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
# #         z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
        
# #         x_hat = self.decoder(z_q_rnn_out) # (B, 1, 320)
# #         return x_hat, h_dec_new


# # # --- MODEL 2: TS3 Codec (Transformer) ---
# # class TS3_Codec(nn.Module):
# #     """
# #     A stateful, causal codec using a Causal Transformer as the core.
# #     This directly addresses your "Transformer" requirement.
# #     """
# #     def __init__(self):
# #         super().__init__()
# #         self.encoder = Encoder()
# #         self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
# #         self.decoder = Decoder()
        
# #         self.encoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
# #         self.decoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
    
# #     def forward(self, x, h_enc=None, h_dec=None):
# #         # x: (B, 1, T)
# #         z_e = self.encoder(x) # (B, C, T_latent)
        
# #         # --- Stateful TFM ---
# #         z_e_tfm_in = z_e.permute(0, 2, 1) # (B, T_latent, C)
# #         z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
# #         z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
# #         # ---
        
# #         z_q, vq_loss, indices = self.quantizer(z_e_tfm_out)

# #         # --- Stateful TFM ---
# #         z_q_tfm_in = z_q.permute(0, 2, 1) # (B, T_latent, C)
# #         z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q_tfm_in, h_dec)
# #         z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
# #         # ---
        
# #         x_hat = self.decoder(z_q_tfm_out)
        
# #         return x_hat, vq_loss, (h_enc_new, h_dec_new)

# #     def encode(self, x, h_enc):
# #         """For streaming: encode audio to indices."""
# #         z_e = self.encoder(x) # (B, C, 40)
# #         z_e_tfm_in = z_e.permute(0, 2, 1)
# #         z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
# #         z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1)
        
# #         _, _, indices = self.quantizer(z_e_tfm_out) # (B, 40)
# #         return indices, h_enc_new

# #     def decode(self, indices, h_dec):
# #         """For streaming: decode indices to audio."""
# #         z_q = self.quantizer.embedding(indices) # (B, 40, C)
        
# #         # TFM model input is (B, 40, C)
# #         z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q, h_dec)
# #         z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, 40)
        
# #         x_hat = self.decoder(z_q_tfm_out) # (B, 1, 320)
# #         return x_hat, h_dec_new

# # # --- MODEL 3: ScoreDec (Diffusion Post-Filter) ---

# # class SinusoidalPosEmb(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.dim = dim

# #     def forward(self, x):
# #         half_dim = self.dim // 2
# #         emb = math.log(10000) / (half_dim - 1)
# #         emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
# #         emb = x[:, None] * emb[None, :]
# #         return torch.cat((emb.sin(), emb.cos()), dim=-1)

# # class DiffusionBlock(nn.Module):
# #     """A single U-Net block for the diffusion model."""
# #     def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
# #         super().__init__()
# #         self.time_mlp = nn.Linear(time_emb_dim, out_channels)
# #         self.cond_mlp = nn.Linear(cond_dim, out_channels)
        
# #         self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, padding=1)
# #         self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, padding=1)
# #         self.bn1 = nn.BatchNorm1d(out_channels)
# #         self.bn2 = nn.BatchNorm1d(out_channels)
# #         self.relu = nn.ReLU()

# #     def forward(self, x, t_emb, cond_emb):
# #         h = self.relu(self.bn1(self.conv1(x)))
        
# #         # Add time and condition embeddings
# #         time_emb = self.relu(self.time_mlp(t_emb))
# #         cond_emb = self.relu(self.cond_mlp(cond_emb))
# #         h = h + time_emb.unsqueeze(-1) + cond_emb.unsqueeze(-1)
        
# #         h = self.relu(self.bn2(self.conv2(h)))
# #         return h

# # class DiffusionUNet1D(nn.Module):
# #     """
# #     A 1D *CAUSAL* Denoising model, conditioned on the low-quality codec output.
# #     This is a simple "WaveNet" style stack, not a U-Net, to maintain causality.
# #     """
# #     def __init__(self, in_channels=1, model_channels=64, time_emb_dim=256, cond_dim=1):
# #         super().__init__()
        
# #         # --- Simple Causal Stack ---
# #         self.time_mlp_simple = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, model_channels), nn.ReLU())
# #         self.cond_proj_simple = CausalConv1d(cond_dim, model_channels, 1)
# #         self.in_conv = CausalConv1d(in_channels, model_channels, 1)
        
# #         self.blocks = nn.ModuleList([
# #             CausalConv1d(model_channels, model_channels, 3, padding=1) for _ in range(4) # 4 residual blocks
# #         ])
# #         self.out_conv = CausalConv1d(model_channels, in_channels, 1)
# #         # --- End Simple Causal Stack ---

# #     def forward(self, x, time, cond):
# #         # t_emb shape: (B, C, 1)
# #         t_emb = self.time_mlp_simple(time).unsqueeze(-1) 
# #         # c_emb shape: (B, C, T)
# #         c_emb = self.cond_proj_simple(cond) 
# #         # x_in shape: (B, C, T)
# #         x_in = self.in_conv(x) 
        
# #         # Add time and condition embeddings
# #         h = x_in + t_emb + c_emb 
# #         for block in self.blocks:
# #             h = block(h) + h # Residual connection
# #         return self.out_conv(h)

# # class ScoreDecPostFilter(nn.Module):
# #     """
# #     Wraps the diffusion U-Net and provides the enhancement logic.
# #     This is what is called by the streaming/evaluation tabs.
# #     """
# #     def __init__(self, timesteps=50, model_channels=64):
# #         super().__init__()
# #         self.timesteps = timesteps
# #         self.model = DiffusionUNet1D(model_channels=model_channels)
        
# #         betas = torch.linspace(1e-4, 0.02, timesteps)
# #         alphas = 1. - betas
# #         alphas_cumprod = torch.cumprod(alphas, axis=0)

# #         self.register_buffer('betas', betas)
# #         self.register_buffer('alphas_cumprod', alphas_cumprod)
# #         self.register_buffer('alphas', alphas)
        
# #     def q_sample(self, x_start, t, noise=None):
# #         """Forward diffusion: noise the clean signal."""
# #         if noise is None: noise = torch.randn_like(x_start)
# #         sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(x_start.shape[0], 1, 1)
# #         sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().view(x_start.shape[0], 1, 1)
        
# #         # FIX: Return both the noised tensor and the noise itself
# #         noised_tensor = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
# #         return noised_tensor, noise

# #     @torch.no_grad()
# #     def p_sample(self, x_t, t, cond):
# #         """One step of the reverse diffusion process."""
# #         t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
# #         alpha_t = self.alphas[t].view(-1, 1, 1)
# #         alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        
# #         predicted_noise = self.model(x_t, t_tensor.float(), cond)
        
# #         # DDPM sampling step
# #         x_prev = (x_t - ((1-alpha_t) / (1-alpha_cumprod_t).sqrt()) * predicted_noise) / alpha_t.sqrt()
        
# #         if t > 0:
# #             noise = torch.randn_like(x_t)
# #             alpha_cumprod_prev_t = self.alphas_cumprod[t-1]
# #             posterior_variance = (1-alpha_cumprod_prev_t) / (1-alpha_cumprod_t) * self.betas[t]
# #             x_prev += torch.sqrt(posterior_variance.view(-1, 1, 1)) * noise
# #         return x_prev

# #     @torch.no_grad()
# #     def enhance(self, x_low_quality, timesteps=10):
# #         """
# #         The main enhancement function.
# #         x_low_quality is the output from the GRU_Codec.
# #         This is SLOW and NOT real-time.
# #         """
# #         # Start from the low-quality audio, but add noise
# #         t_start = timesteps - 1
# #         x_t = self.q_sample(x_low_quality, torch.tensor([t_start], device=x_low_quality.device))
        
# #         for i in reversed(range(timesteps)):
# #             x_t = self.p_sample(x_t, i, cond=x_low_quality)
            
# #         return torch.tanh(x_t)


# # # --- TRADITIONAL CODECS (For Baseline Comparison) ---
# # class MuLawCodec:
# #     def __init__(self, quantization_channels=256): self.mu = float(quantization_channels - 1)
# #     def encode(self, x):
# #         mu_t = torch.tensor(self.mu, device=x.device, dtype=torch.float32)
# #         encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
# #         return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
# #     def decode(self, z):
# #         z_float = z.to(torch.float32)
# #         mu_t = torch.tensor(self.mu, device=z.device, dtype=torch.float32)
# #         y = (z_float / self.mu) * 2.0 - 1.0
# #         return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

# # class ALawCodec:
# #     def __init__(self): self.A = 87.6
# #     def encode(self, x):
# #         a_t = torch.tensor(self.A, device=x.device, dtype=torch.float32)
# #         abs_x = torch.abs(x)
# #         encoded = torch.zeros_like(x)
# #         cond = abs_x < (1 / self.A)
# #         encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
# #         encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
# #         return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
# #     def decode(self, z):
# #         z_float = z.to(torch.float32)
# #         a_t = torch.tensor(self.A, device=z.device, dtype=torch.float32)
# #         y = (z_float / 127.5) - 1.0
# #         abs_y = torch.abs(y)
# #         decoded = torch.zeros_like(y)
# #         cond = abs_y < (1 / (1 + torch.log(a_t)))
# #         decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
# #         decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
# #         return decoded.unsqueeze(1)

# # # --- DATASET & TRAINING ---
# # TRAIN_CHUNK_SIZE = 16000 # 1 second

# # class AudioChunkDataset(Dataset):
# #     def __init__(self, directory, chunk_size=TRAIN_CHUNK_SIZE, sample_rate=16000):
# #         self.chunk_size, self.sample_rate = chunk_size, sample_rate
# #         self.file_paths = [os.path.join(r, f) for r, _, fs in os.walk(directory) for f in fs if f.lower().endswith(('.wav', '.flac'))]
# #         if not self.file_paths: raise ValueError("No audio files found.")
# #     def __len__(self): return len(self.file_paths)
# #     def __getitem__(self, idx):
# #         try:
# #             waveform, sr = torchaudio.load(self.file_paths[idx])
# #             if sr != self.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
# #             if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
# #             if waveform.shape[1] > self.chunk_size:
# #                 start = np.random.randint(0, waveform.shape[1] - self.chunk_size)
# #                 waveform = waveform[:, start:start + self.chunk_size]
# #             else:
# #                 waveform = F.pad(waveform, (0, self.chunk_size - waveform.shape[1]))
# #             return waveform
# #         except Exception as e:
# #             print(f"Warning: Skipping file {self.file_paths[idx]}. Error: {e}")
# #             return torch.zeros((1, self.chunk_size))

# # def train_model(dataset_path, epochs, learning_rate, batch_size, model_save_path, progress_callback, stop_event, model_type):
# #     """
# #     Main training function. Now handles 'gru', 'transformer', and 'scoredec' types.
# #     """
# #     try:
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         progress_callback.emit(f"Using device: {device}")
        
# #         dataset = AudioChunkDataset(directory=dataset_path)
# #         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# #         progress_callback.emit(f"Dataset loaded with {len(dataset)} files.")

# #         if model_type == 'gru':
# #             model = GRU_Codec().to(device)
# #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# #             stft_criterion = MultiResolutionSTFTLoss().to(device)
# #             l1_criterion = nn.L1Loss().to(device)
# #             progress_callback.emit(f"Starting training for GRU_Codec model...")
        
# #         elif model_type == 'transformer':
# #             model = TS3_Codec().to(device)
# #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# #             stft_criterion = MultiResolutionSTFTLoss().to(device)
# #             l1_criterion = nn.L1Loss().to(device)
# #             progress_callback.emit(f"Starting training for TS3_Codec model...")
        
# #         elif model_type == 'scoredec':
# #             progress_callback.emit("--- Starting ScoreDec Post-Filter Training ---")
# #             progress_callback.emit("Loading pre-trained GRU_Codec...")
# #             try:
# #                 gru_codec = GRU_Codec().to(device)
# #                 gru_codec.load_state_dict(torch.load("low_latency_codec_gru.pth", map_location=device))
# #                 gru_codec.eval()
# #                 for param in gru_codec.parameters():
# #                     param.requires_grad = False
# #                 progress_callback.emit("GRU_Codec loaded and frozen.")
# #             except FileNotFoundError:
# #                 progress_callback.emit("ERROR: 'low_latency_codec_gru.pth' not found. You must train the GRU_Codec first.")
# #                 return
            
# #             model = ScoreDecPostFilter().to(device)
# #             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# #             l1_criterion = nn.L1Loss().to(device)
# #             progress_callback.emit("Starting training for ScoreDec model...")
            
# #         else:
# #             raise ValueError(f"Unknown model type for training: {model_type}")

# #         # --- Main Training Loop ---
# #         for epoch in range(epochs):
# #             if stop_event.is_set():
# #                 progress_callback.emit("Training stopped by user."); break
            
# #             for i, data in enumerate(dataloader):
# #                 inputs = data.to(device)
# #                 optimizer.zero_grad()
                
# #                 if model_type in ['gru', 'transformer']:
# #                     h_enc, h_dec = None, None # Reset state per batch
# #                     x_hat, vq_loss, (h_enc, h_dec) = model(inputs, h_enc, h_dec)
                    
# #                     # --- FIX: Defensively pad output to match input length ---
# #                     # This is the most likely source of the STFT error
# #                     input_len = inputs.shape[-1]
# #                     output_len = x_hat.shape[-1]
                    
# #                     if output_len < input_len:
# #                         # Pad x_hat on the right if it's shorter
# #                         padding = input_len - output_len
# #                         x_hat = F.pad(x_hat, (0, padding))
# #                     elif output_len > input_len:
# #                         # Trim x_hat if it's longer
# #                         x_hat = x_hat[..., :input_len]
# #                     # --- End Fix ---
                    
# #                     stft_loss = stft_criterion(x_hat, inputs)
# #                     l1_loss = l1_criterion(x_hat, inputs)
# #                     loss = stft_loss + 0.1 * l1_loss + vq_loss
                    
# #                     if i % 20 == 19:
# #                         progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss.item():.5f} (STFT: {stft_loss.item():.4f}, VQ: {vq_loss.item():.4f})")

# #                 elif model_type == 'scoredec':
# #                     with torch.no_grad():
# #                         x_hat_low_quality, _, _ = gru_codec(inputs)
                        
# #                         # --- FIX: Ensure low-quality output matches input ---
# #                         input_len = inputs.shape[-1]
# #                         output_len = x_hat_low_quality.shape[-1]
# #                         if output_len < input_len:
# #                             padding = input_len - output_len
# #                             x_hat_low_quality = F.pad(x_hat_low_quality, (0, padding))
# #                         elif output_len > input_len:
# #                             x_hat_low_quality = x_hat_low_quality[..., :input_len]
# #                         # --- End Fix ---
                        
# #                         x_hat_low_quality = x_hat_low_quality.detach()
                    
# #                     # Train diffusion model
# #                     t = torch.randint(0, model.timesteps, (inputs.shape[0],), device=device).long()
                    
# #                     # This line was the source of the "unpack" error
# #                     # It is now fixed because q_sample returns (x_t, noise)
# #                     x_t, noise = model.q_sample(x_start=inputs, t=t)
                    
# #                     predicted_noise = model.model(x_t, t.float(), cond=x_hat_low_quality)
# #                     loss = l1_criterion(predicted_noise, noise)
                    
# #                     if i % 20 == 19:
# #                         progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Denoising Loss: {loss.item():.5f}")

# #                 loss.backward()
# #                 optimizer.step()

# #             progress_callback.emit(f"--- Epoch {epoch + 1} finished ---")

# #         if not stop_event.is_set():
# #             progress_callback.emit("Training finished. Saving model...")
# #             torch.save(model.state_dict(), model_save_path)
# #             progress_callback.emit(f"Model saved to {model_save_path}")
# #     except Exception as e:
# #         progress_callback.emit(f"ERROR in training: {e}")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchaudio
# import os
# import numpy as np
# import torch.nn.functional as F
# import math

# # --- Perceptual Loss Function ---
# class MultiResolutionSTFTLoss(nn.Module):
#     """
#     Multi-resolution STFT loss, common in audio generation models.
#     This is a key part of achieving high quality (PESQ/STOI).
#     """
#     def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
#         super(MultiResolutionSTFTLoss, self).__init__()
#         self.fft_sizes = fft_sizes
#         self.hop_sizes = hop_sizes
#         self.win_lengths = win_lengths
#         self.window = torch.hann_window
#         assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

#     def forward(self, y_hat, y):
#         sc_loss, mag_loss = 0.0, 0.0
#         for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
#             window = self.window(win, device=y.device)
#             spec_hat = torch.stft(y_hat.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
#             spec = torch.stft(y.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            
#             sc_loss += torch.norm(torch.abs(spec) - torch.abs(spec_hat), p='fro') / torch.norm(torch.abs(spec), p='fro')
#             mag_loss += F.l1_loss(torch.log(torch.abs(spec).clamp(min=1e-9)), torch.log(torch.abs(spec_hat).clamp(min=1e-9)))
            
#         return (sc_loss / len(self.fft_sizes)) + (mag_loss / len(self.fft_sizes))

# # --- Causal Convolution ---
# class CausalConv1d(nn.Conv1d):
#     """
#     A 1D convolution that is causal (cannot see the future).
#     This is critical for the < 20ms latency goal.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Calculate the padding needed to make it causal
#         self.causal_padding = self.kernel_size[0] - 1

#     def forward(self, x):
#         # Pad on the left (past) only
#         return super().forward(F.pad(x, (self.causal_padding, 0)))

# class CausalConvTranspose1d(nn.ConvTranspose1d):
#     """
#     A 1D *transpose* convolution that is causal.
#     It removes output samples that would "see the future".
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.causal_padding = self.kernel_size[0] - self.stride[0]

#     def forward(self, x):
#         x = super().forward(x)
#         # Remove the invalid, "future-seeing" samples from the end
#         if self.causal_padding != 0:
#             return x[..., :-self.causal_padding]
#         return x

# # --- Vector Quantizer (The heart of the *COMPRESSION*) ---
# class VectorQuantizer(nn.Module):
#     """
#     The Vector Quantizer (VQ) module. This is what enables low-bitrate compression.
#     It maps continuous latent vectors to a discrete set of "codes" from a codebook.
#     """
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost):
#         super(VectorQuantizer, self).__init__()
#         self.num_embeddings = num_embeddings # Codebook size (e.g., 256)
#         self.embedding_dim = embedding_dim # Dimension of each code
#         self.commitment_cost = commitment_cost # 'beta' in VQ-VAE
        
#         # The codebook
#         self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
#         self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

#     def forward(self, z_e):
#         # z_e shape: (B, C, T) -> (B*T, C)
#         z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        
#         # Find the closest codebook vector (L2 distance)
#         distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True) 
#                      + torch.sum(self.embedding.weight**2, dim=1)
#                      - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))
        
#         # Get the indices of the closest vectors
#         encoding_indices = torch.argmin(distances, dim=1)
        
#         # Quantize: Map indices back to codebook vectors
#         z_q = self.embedding(encoding_indices).view(z_e.shape[0], -1, self.embedding_dim)
#         z_q = z_q.permute(0, 2, 1).contiguous() # (B, C, T)

#         # VQ-VAE Loss (Commitment Loss)
#         e_loss = F.mse_loss(z_q.detach(), z_e) * self.commitment_cost
#         q_loss = F.mse_loss(z_q, z_e.detach())
#         vq_loss = q_loss + e_loss
        
#         # Straight-Through Estimator (STE)
#         # This copies the gradient from z_q to z_e
#         z_q = z_e + (z_q - z_e).detach()
        
#         return z_q, vq_loss, encoding_indices.view(z_e.shape[0], -1) # (B, T)

# # --- The New Codec Architecture Components ---
# # These are based on the SoundStream model, but simplified.

# HOP_SIZE = 320 # 20ms frame (320 samples / 16000 Hz = 0.02s)
# LATENT_DIM = 64
# VQ_EMBEDDINGS = 256 # 8 bits per code
# # 16000 bits/sec / 50 frames/sec = 320 bits/frame
# # 320 bits / 8 bits/index = 40 indices per frame
# NUM_QUANTIZERS = 40 # This is our 16kbps target (40 bytes * 50 fps = 2000 B/s = 16 kbps)

# class Encoder(nn.Module):
#     """
#     Causal encoder. Takes raw audio and produces latent vectors.
#     Takes a 320-sample chunk and produces 40 latent vectors.
#     Total stride must be 320 / 40 = 8.
#     """
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             CausalConv1d(1, 32, 7), nn.ELU(),
#             CausalConv1d(32, 64, 5, stride=2), nn.ELU(), # 320 -> 160
#             CausalConv1d(64, 64, 5, stride=2), nn.ELU(), # 160 -> 80
#             CausalConv1d(64, LATENT_DIM, 5, stride=2), nn.ELU() # 80 -> 40
#         )
#         # Output shape: (B, LATENT_DIM, 40)
#         # This is exactly what we need.

#     def forward(self, x):
#         return self.net(x)

# class Decoder(nn.Module):
#     """
#     Causal decoder. Takes quantized latents and reconstructs audio.
#     Must be the inverse of the Encoder.
#     """
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             CausalConvTranspose1d(LATENT_DIM, 64, 5, stride=2), nn.ELU(), # 40 -> 80
#             CausalConvTranspose1d(64, 64, 5, stride=2), nn.ELU(), # 80 -> 160
#             CausalConvTranspose1d(64, 32, 5, stride=2), nn.ELU(), # 160 -> 320
#             CausalConv1d(32, 1, 7), nn.Tanh() # Final output
#         )
    
#     def forward(self, x):
#         return self.net(x)

# # --- Causal Transformer (for the Transformer-based Codec) ---
# class CausalTransformerEncoder(nn.Module):
#     def __init__(self, d_model, nhead, num_layers):
#         super().__init__()
#         self.d_model = d_model
#         layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#         self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
    
#     def get_causal_mask(self, sz):
#         # Returns a mask of shape (sz, sz)
#         # FIX: Create a boolean mask where True means "masked out"
#         # Create directly as bool to avoid the UserWarning
#         return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

#     def forward(self, x, state):
#         # x shape: (B, T, C) e.g. (1, 40, 64)
#         # state shape: (B, S, C) e.g. (1, 100, 64)
        
#         if state is None:
#             # First frame, no state
#             inp = x
#         else:
#             # Append new frame to old state
#             inp = torch.cat([state, x], dim=1)
        
#         # We must limit the state size to avoid OOM
#         # Let's say, 10 frames of history (10 * 40 = 400 steps)
#         STATE_LEN = 400
#         if inp.shape[1] > STATE_LEN:
#             inp = inp[:, -STATE_LEN:, :]
        
#         new_state = inp.detach() # The new state is the full input
        
#         # Create a causal mask for the *full input sequence*
#         mask = self.get_causal_mask(inp.shape[1]).to(x.device)
        
#         # Process the full sequence
#         out = self.transformer(inp, mask=mask)
        
#         # Only return the *new* frames, corresponding to x
#         # This is how we make it stateful
#         out = out[:, -x.shape[1]:, :] # (B, T, C)
        
#         return out, new_state


# # --- MODEL 1: GRU Codec (Fast) ---
# class GRU_Codec(nn.Module):
#     """
#     A stateful, causal codec using a GRU (RNN) as the core.
#     This is the "simpler neural approach" and is very fast.
#     """
#     def __init__(self):
#         super().__init__()
#         self.encoder = Encoder()
#         self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
#         self.decoder = Decoder()
        
#         self.encoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
#         self.decoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
    
#     def forward(self, x, h_enc=None, h_dec=None):
#         # x: (B, 1, T)
#         z_e = self.encoder(x) # (B, C, T_latent)
        
#         # --- Stateful RNN ---
#         # (B, C, T_latent) -> (B, T_latent, C)
#         z_e_rnn_in = z_e.permute(0, 2, 1)
#         z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
#         # (B, T_latent, C) -> (B, C, T_latent)
#         z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
#         # ---
        
#         z_q, vq_loss, indices = self.quantizer(z_e_rnn_out)

#         # --- Stateful RNN ---
#         # (B, C, T_latent) -> (B, T_latent, C)
#         z_q_rnn_in = z_q.permute(0, 2, 1)
#         z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
#         # (B, T_latent, C) -> (B, C, T_latent)
#         z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
#         # ---
        
#         x_hat = self.decoder(z_q_rnn_out)
        
#         return x_hat, vq_loss, (h_enc_new, h_dec_new)

#     def encode(self, x, h_enc):
#         """For streaming: encode audio to indices."""
#         z_e = self.encoder(x) # (B, C, 40)
#         z_e_rnn_in = z_e.permute(0, 2, 1)
#         z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
#         z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
        
#         # We don't need vq_loss here, just indices
#         _, _, indices = self.quantizer(z_e_rnn_out) # (B, 40)
#         return indices, h_enc_new

#     def decode(self, indices, h_dec):
#         """For streaming: decode indices to audio."""
#         # Convert indices (B, 40) to codebook vectors (B, C, 40)
#         z_q = self.quantizer.embedding(indices) # (B, 40, C)
#         z_q = z_q.permute(0, 2, 1) # (B, C, 40)
        
#         z_q_rnn_in = z_q.permute(0, 2, 1)
#         z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
#         z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
        
#         x_hat = self.decoder(z_q_rnn_out) # (B, 1, 320)
#         return x_hat, h_dec_new


# # --- MODEL 2: TS3 Codec (Transformer) ---
# class TS3_Codec(nn.Module):
#     """
#     A stateful, causal codec using a Causal Transformer as the core.
#     This directly addresses your "Transformer" requirement.
#     """
#     def __init__(self):
#         super().__init__()
#         self.encoder = Encoder()
#         self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
#         self.decoder = Decoder()
        
#         self.encoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
#         self.decoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
    
#     def forward(self, x, h_enc=None, h_dec=None):
#         # x: (B, 1, T)
#         z_e = self.encoder(x) # (B, C, T_latent)
        
#         # --- Stateful TFM ---
#         z_e_tfm_in = z_e.permute(0, 2, 1) # (B, T_latent, C)
#         z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
#         z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
#         # ---
        
#         z_q, vq_loss, indices = self.quantizer(z_e_tfm_out)

#         # --- Stateful TFM ---
#         z_q_tfm_in = z_q.permute(0, 2, 1) # (B, T_latent, C)
#         z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q_tfm_in, h_dec)
#         z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
#         # ---
        
#         x_hat = self.decoder(z_q_tfm_out)
        
#         return x_hat, vq_loss, (h_enc_new, h_dec_new)

#     def encode(self, x, h_enc):
#         """For streaming: encode audio to indices."""
#         z_e = self.encoder(x) # (B, C, 40)
#         z_e_tfm_in = z_e.permute(0, 2, 1)
#         z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
#         z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1)
        
#         _, _, indices = self.quantizer(z_e_tfm_out) # (B, 40)
#         return indices, h_enc_new

#     def decode(self, indices, h_dec):
#         """For streaming: decode indices to audio."""
#         z_q = self.quantizer.embedding(indices) # (B, 40, C)
        
#         # TFM model input is (B, 40, C)
#         z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q, h_dec)
#         z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, 40)
        
#         x_hat = self.decoder(z_q_tfm_out) # (B, 1, 320)
#         return x_hat, h_dec_new

# # --- MODEL 3: ScoreDec (Diffusion Post-Filter) ---

# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         return torch.cat((emb.sin(), emb.cos()), dim=-1)

# class DiffusionBlock(nn.Module):
#     """A single U-Net block for the diffusion model."""
#     def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
#         super().__init__()
#         self.time_mlp = nn.Linear(time_emb_dim, out_channels)
#         self.cond_mlp = nn.Linear(cond_dim, out_channels)
        
#         self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x, t_emb, cond_emb):
#         h = self.relu(self.bn1(self.conv1(x)))
        
#         # Add time and condition embeddings
#         time_emb = self.relu(self.time_mlp(t_emb))
#         cond_emb = self.relu(self.cond_mlp(cond_emb))
#         h = h + time_emb.unsqueeze(-1) + cond_emb.unsqueeze(-1)
        
#         h = self.relu(self.bn2(self.conv2(h)))
#         return h

# class DiffusionUNet1D(nn.Module):
#     """
#     A 1D *CAUSAL* Denoising model, conditioned on the low-quality codec output.
#     This is a simple "WaveNet" style stack, not a U-Net, to maintain causality.
#     """
#     def __init__(self, in_channels=1, model_channels=64, time_emb_dim=256, cond_dim=1):
#         super().__init__()
        
#         # --- Simple Causal Stack ---
#         self.time_mlp_simple = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, model_channels), nn.ReLU())
#         self.cond_proj_simple = CausalConv1d(cond_dim, model_channels, 1)
#         self.in_conv = CausalConv1d(in_channels, model_channels, 1)
        
#         # FIX: Removed padding=1. CausalConv1d handles its own padding.
#         # This was the source of the "16002 vs 16000" error.
#         self.blocks = nn.ModuleList([
#             CausalConv1d(model_channels, model_channels, 3) for _ in range(4) # 4 residual blocks
#         ])
#         self.out_conv = CausalConv1d(model_channels, in_channels, 1)
#         # --- End Simple Causal Stack ---

#     def forward(self, x, time, cond):
#         # t_emb shape: (B, C, 1)
#         t_emb = self.time_mlp_simple(time).unsqueeze(-1) 
#         # c_emb shape: (B, C, T)
#         c_emb = self.cond_proj_simple(cond) 
#         # x_in shape: (B, C, T)
#         x_in = self.in_conv(x) 
        
#         # Add time and condition embeddings
#         h = x_in + t_emb + c_emb 
#         for block in self.blocks:
#             h = block(h) + h # Residual connection
#         return self.out_conv(h)

# class ScoreDecPostFilter(nn.Module):
#     """
#     Wraps the diffusion U-Net and provides the enhancement logic.
#     This is what is called by the streaming/evaluation tabs.
#     """
#     def __init__(self, timesteps=50, model_channels=64):
#         super().__init__()
#         self.timesteps = timesteps
#         self.model = DiffusionUNet1D(model_channels=model_channels)
        
#         betas = torch.linspace(1e-4, 0.02, timesteps)
#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, axis=0)

#         self.register_buffer('betas', betas)
#         self.register_buffer('alphas_cumprod', alphas_cumprod)
#         self.register_buffer('alphas', alphas)
        
#     def q_sample(self, x_start, t, noise=None):
#         """Forward diffusion: noise the clean signal."""
#         if noise is None: noise = torch.randn_like(x_start)
#         sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(x_start.shape[0], 1, 1)
#         sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().view(x_start.shape[0], 1, 1)
        
#         # FIX: Return both the noised tensor and the noise itself
#         noised_tensor = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
#         return noised_tensor, noise

#     @torch.no_grad()
#     def p_sample(self, x_t, t, cond):
#         """One step of the reverse diffusion process."""
#         t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
#         alpha_t = self.alphas[t].view(-1, 1, 1)
#         alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        
#         predicted_noise = self.model(x_t, t_tensor.float(), cond)
        
#         # DDPM sampling step
#         x_prev = (x_t - ((1-alpha_t) / (1-alpha_cumprod_t).sqrt()) * predicted_noise) / alpha_t.sqrt()
        
#         if t > 0:
#             noise = torch.randn_like(x_t)
#             alpha_cumprod_prev_t = self.alphas_cumprod[t-1]
#             posterior_variance = (1-alpha_cumprod_prev_t) / (1-alpha_cumprod_t) * self.betas[t]
#             x_prev += torch.sqrt(posterior_variance.view(-1, 1, 1)) * noise
#         return x_prev

#     @torch.no_grad()
#     def enhance(self, x_low_quality, timesteps=10):
#         """
#         The main enhancement function.
#         x_low_quality is the output from the GRU_Codec.
#         This is SLOW and NOT real-time.
#         """
#         # Start from the low-quality audio, but add noise
#         t_start = timesteps - 1
#         x_t = self.q_sample(x_low_quality, torch.tensor([t_start], device=x_low_quality.device))
        
#         for i in reversed(range(timesteps)):
#             x_t = self.p_sample(x_t, i, cond=x_low_quality)
            
#         return torch.tanh(x_t)


# # --- TRADITIONAL CODECS (For Baseline Comparison) ---
# class MuLawCodec:
#     def __init__(self, quantization_channels=256): self.mu = float(quantization_channels - 1)
#     def encode(self, x):
#         mu_t = torch.tensor(self.mu, device=x.device, dtype=torch.float32)
#         encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
#         return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
#     def decode(self, z):
#         z_float = z.to(torch.float32)
#         mu_t = torch.tensor(self.mu, device=z.device, dtype=torch.float32)
#         y = (z_float / self.mu) * 2.0 - 1.0
#         return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

# class ALawCodec:
#     def __init__(self): self.A = 87.6
#     def encode(self, x):
#         a_t = torch.tensor(self.A, device=x.device, dtype=torch.float32)
#         abs_x = torch.abs(x)
#         encoded = torch.zeros_like(x)
#         cond = abs_x < (1 / self.A)
#         encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
#         encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
#         return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
#     def decode(self, z):
#         z_float = z.to(torch.float32)
#         a_t = torch.tensor(self.A, device=z.device, dtype=torch.float32)
#         y = (z_float / 127.5) - 1.0
#         abs_y = torch.abs(y)
#         decoded = torch.zeros_like(y)
#         cond = abs_y < (1 / (1 + torch.log(a_t)))
#         decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
#         decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
#         return decoded.unsqueeze(1)

# # --- DATASET & TRAINING ---
# TRAIN_CHUNK_SIZE = 16000 # 1 second

# class AudioChunkDataset(Dataset):
#     def __init__(self, directory, chunk_size=TRAIN_CHUNK_SIZE, sample_rate=16000):
#         self.chunk_size, self.sample_rate = chunk_size, sample_rate
#         self.file_paths = [os.path.join(r, f) for r, _, fs in os.walk(directory) for f in fs if f.lower().endswith(('.wav', '.flac'))]
#         if not self.file_paths: raise ValueError("No audio files found.")
#     def __len__(self): return len(self.file_paths)
#     def __getitem__(self, idx):
#         try:
#             waveform, sr = torchaudio.load(self.file_paths[idx])
#             if sr != self.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
#             if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
#             if waveform.shape[1] > self.chunk_size:
#                 start = np.random.randint(0, waveform.shape[1] - self.chunk_size)
#                 waveform = waveform[:, start:start + self.chunk_size]
#             else:
#                 waveform = F.pad(waveform, (0, self.chunk_size - waveform.shape[1]))
#             return waveform
#         except Exception as e:
#             print(f"Warning: Skipping file {self.file_paths[idx]}. Error: {e}")
#             return torch.zeros((1, self.chunk_size))

# def train_model(dataset_path, epochs, learning_rate, batch_size, model_save_path, progress_callback, stop_event, model_type):
#     """
#     Main training function. Now handles 'gru', 'transformer', and 'scoredec' types.
#     """
#     try:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         progress_callback.emit(f"Using device: {device}")
        
#         dataset = AudioChunkDataset(directory=dataset_path)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
#         progress_callback.emit(f"Dataset loaded with {len(dataset)} files.")

#         if model_type == 'gru':
#             model = GRU_Codec().to(device)
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#             stft_criterion = MultiResolutionSTFTLoss().to(device)
#             l1_criterion = nn.L1Loss().to(device)
#             progress_callback.emit(f"Starting training for GRU_Codec model...")
        
#         elif model_type == 'transformer':
#             model = TS3_Codec().to(device)
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#             stft_criterion = MultiResolutionSTFTLoss().to(device)
#             l1_criterion = nn.L1Loss().to(device)
#             progress_callback.emit(f"Starting training for TS3_Codec model...")
        
#         elif model_type == 'scoredec':
#             progress_callback.emit("--- Starting ScoreDec Post-Filter Training ---")
#             progress_callback.emit("Loading pre-trained GRU_Codec...")
#             try:
#                 gru_codec = GRU_Codec().to(device)
#                 gru_codec.load_state_dict(torch.load("low_latency_codec_gru.pth", map_location=device))
#                 gru_codec.eval()
#                 for param in gru_codec.parameters():
#                     param.requires_grad = False
#                 progress_callback.emit("GRU_Codec loaded and frozen.")
#             except FileNotFoundError:
#                 progress_callback.emit("ERROR: 'low_latency_codec_gru.pth' not found. You must train the GRU_Codec first.")
#                 return
            
#             model = ScoreDecPostFilter().to(device)
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#             l1_criterion = nn.L1Loss().to(device)
#             progress_callback.emit("Starting training for ScoreDec model...")
            
#         else:
#             raise ValueError(f"Unknown model type for training: {model_type}")

#         # --- Main Training Loop ---
#         for epoch in range(epochs):
#             if stop_event.is_set():
#                 progress_callback.emit("Training stopped by user."); break
            
#             for i, data in enumerate(dataloader):
#                 inputs = data.to(device)
#                 optimizer.zero_grad()
                
#                 if model_type in ['gru', 'transformer']:
#                     h_enc, h_dec = None, None # Reset state per batch
#                     x_hat, vq_loss, (h_enc, h_dec) = model(inputs, h_enc, h_dec)
                    
#                     # --- FIX: Defensively pad output to match input length ---
#                     # This is the most likely source of the STFT error
#                     input_len = inputs.shape[-1]
#                     output_len = x_hat.shape[-1]
                    
#                     if output_len < input_len:
#                         # Pad x_hat on the right if it's shorter
#                         padding = input_len - output_len
#                         x_hat = F.pad(x_hat, (0, padding))
#                     elif output_len > input_len:
#                         # Trim x_hat if it's longer
#                         x_hat = x_hat[..., :input_len]
#                     # --- End Fix ---
                    
#                     stft_loss = stft_criterion(x_hat, inputs)
#                     l1_loss = l1_criterion(x_hat, inputs)
#                     loss = stft_loss + 0.1 * l1_loss + vq_loss
                    
#                     if i % 20 == 19:
#                         progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss.item():.5f} (STFT: {stft_loss.item():.4f}, VQ: {vq_loss.item():.4f})")

#                 elif model_type == 'scoredec':
#                     with torch.no_grad():
#                         x_hat_low_quality, _, _ = gru_codec(inputs)
                        
#                         # --- FIX: Ensure low-quality output matches input ---
#                         input_len = inputs.shape[-1]
#                         output_len = x_hat_low_quality.shape[-1]
#                         if output_len < input_len:
#                             padding = input_len - output_len
#                             x_hat_low_quality = F.pad(x_hat_low_quality, (0, padding))
#                         elif output_len > input_len:
#                             x_hat_low_quality = x_hat_low_quality[..., :input_len]
#                         # --- End Fix ---
                        
#                         x_hat_low_quality = x_hat_low_quality.detach()
                    
#                     # Train diffusion model
#                     t = torch.randint(0, model.timesteps, (inputs.shape[0],), device=device).long()
                    
#                     # This line was the source of the "unpack" error
#                     # It is now fixed because q_sample returns (x_t, noise)
#                     x_t, noise = model.q_sample(x_start=inputs, t=t)
                    
#                     predicted_noise = model.model(x_t, t.float(), cond=x_hat_low_quality)
#                     loss = l1_criterion(predicted_noise, noise)
                    
#                     if i % 20 == 19:
#                         progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Denoising Loss: {loss.item():.5f}")

#                 loss.backward()
#                 optimizer.step()

#             progress_callback.emit(f"--- Epoch {epoch + 1} finished ---")

#         if not stop_event.is_set():
#             progress_callback.emit("Training finished. Saving model...")
#             torch.save(model.state_dict(), model_save_path)
#             progress_callback.emit(f"Model saved to {model_save_path}")
#     except Exception as e:
#         progress_callback.emit(f"ERROR in training: {e}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import numpy as np
import torch.nn.functional as F
import math

# --- Perceptual Loss Function ---
class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss, common in audio generation models.
    This is a key part of achieving high quality (PESQ/STOI).
    """
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
            spec_hat = torch.stft(y_hat.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            spec = torch.stft(y.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            
            sc_loss += torch.norm(torch.abs(spec) - torch.abs(spec_hat), p='fro') / torch.norm(torch.abs(spec), p='fro')
            mag_loss += F.l1_loss(torch.log(torch.abs(spec).clamp(min=1e-9)), torch.log(torch.abs(spec_hat).clamp(min=1e-9)))
            
        return (sc_loss / len(self.fft_sizes)) + (mag_loss / len(self.fft_sizes))

# --- Causal Convolution ---
class CausalConv1d(nn.Conv1d):
    """
    A 1D convolution that is causal (cannot see the future).
    This is critical for the < 20ms latency goal.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Calculate the padding needed to make it causal
        self.causal_padding = self.kernel_size[0] - 1

    def forward(self, x):
        # Pad on the left (past) only
        return super().forward(F.pad(x, (self.causal_padding, 0)))

class CausalConvTranspose1d(nn.ConvTranspose1d):
    """
    A 1D *transpose* convolution that is causal.
    It removes output samples that would "see the future".
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.kernel_size[0] - self.stride[0]

    def forward(self, x):
        x = super().forward(x)
        # Remove the invalid, "future-seeing" samples from the end
        if self.causal_padding != 0:
            return x[..., :-self.causal_padding]
        return x

# --- Vector Quantizer (The heart of the *COMPRESSION*) ---
class VectorQuantizer(nn.Module):
    """
    The Vector Quantizer (VQ) module. This is what enables low-bitrate compression.
    It maps continuous latent vectors to a discrete set of "codes" from a codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings # Codebook size (e.g., 256)
        self.embedding_dim = embedding_dim # Dimension of each code
        self.commitment_cost = commitment_cost # 'beta' in VQ-VAE
        
        # The codebook
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z_e):
        # z_e shape: (B, C, T) -> (B*T, C)
        z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        
        # Find the closest codebook vector (L2 distance)
        distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))
        
        # Get the indices of the closest vectors
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize: Map indices back to codebook vectors
        z_q = self.embedding(encoding_indices).view(z_e.shape[0], -1, self.embedding_dim)
        z_q = z_q.permute(0, 2, 1).contiguous() # (B, C, T)

        # VQ-VAE Loss (Commitment Loss)
        e_loss = F.mse_loss(z_q.detach(), z_e) * self.commitment_cost
        q_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = q_loss + e_loss
        
        # Straight-Through Estimator (STE)
        # This copies the gradient from z_q to z_e
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, vq_loss, encoding_indices.view(z_e.shape[0], -1) # (B, T)

# --- The New Codec Architecture Components ---
# These are based on the SoundStream model, but simplified.

HOP_SIZE = 320 # 20ms frame (320 samples / 16000 Hz = 0.02s)
LATENT_DIM = 64
VQ_EMBEDDINGS = 256 # 8 bits per code
# 16000 bits/sec / 50 frames/sec = 320 bits/frame
# 320 bits / 8 bits/index = 40 indices per frame
NUM_QUANTIZERS = 40 # This is our 16kbps target (40 bytes * 50 fps = 2000 B/s = 16 kbps)

class Encoder(nn.Module):
    """
    Causal encoder. Takes raw audio and produces latent vectors.
    Takes a 320-sample chunk and produces 40 latent vectors.
    Total stride must be 320 / 40 = 8.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(1, 32, 7), nn.ELU(),
            CausalConv1d(32, 64, 5, stride=2), nn.ELU(), # 320 -> 160
            CausalConv1d(64, 64, 5, stride=2), nn.ELU(), # 160 -> 80
            CausalConv1d(64, LATENT_DIM, 5, stride=2), nn.ELU() # 80 -> 40
        )
        # Output shape: (B, LATENT_DIM, 40)
        # This is exactly what we need.

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """
    Causal decoder. Takes quantized latents and reconstructs audio.
    Must be the inverse of the Encoder.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            CausalConvTranspose1d(LATENT_DIM, 64, 5, stride=2), nn.ELU(), # 40 -> 80
            CausalConvTranspose1d(64, 64, 5, stride=2), nn.ELU(), # 80 -> 160
            CausalConvTranspose1d(64, 32, 5, stride=2), nn.ELU(), # 160 -> 320
            CausalConv1d(32, 1, 7), nn.Tanh() # Final output
        )
    
    def forward(self, x):
        return self.net(x)

# --- Causal Transformer (for the Transformer-based Codec) ---
class CausalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
    
    def get_causal_mask(self, sz, device):
        # Returns a mask of shape (sz, sz)
        # FIX: Create a boolean mask directly on the target device to avoid UserWarning
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x, state):
        # x shape: (B, T, C) e.g. (1, 40, 64)
        # state shape: (B, S, C) e.g. (1, 100, 64)
        
        if state is None:
            # First frame, no state
            inp = x
        else:
            # Append new frame to old state
            inp = torch.cat([state, x], dim=1)
        
        # We must limit the state size to avoid OOM
        # Let's say, 10 frames of history (10 * 40 = 400 steps)
        STATE_LEN = 400
        if inp.shape[1] > STATE_LEN:
            inp = inp[:, -STATE_LEN:, :]
        
        new_state = inp.detach() # The new state is the full input
        
        # Create a causal mask for the *full input sequence*
        mask = self.get_causal_mask(inp.shape[1], x.device)
        
        # Process the full sequence
        out = self.transformer(inp, mask=mask)
        
        # Only return the *new* frames, corresponding to x
        # This is how we make it stateful
        out = out[:, -x.shape[1]:, :] # (B, T, C)
        
        return out, new_state


# --- MODEL 1: GRU Codec (Fast) ---
class GRU_Codec(nn.Module):
    """
    A stateful, causal codec using a GRU (RNN) as the core.
    This is the "simpler neural approach" and is very fast.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
        self.decoder = Decoder()
        
        self.encoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
        self.decoder_rnn = nn.GRU(LATENT_DIM, LATENT_DIM, batch_first=True)
    
    def forward(self, x, h_enc=None, h_dec=None):
        # x: (B, 1, T)
        z_e = self.encoder(x) # (B, C, T_latent)
        
        # --- Stateful RNN ---
        # (B, C, T_latent) -> (B, T_latent, C)
        z_e_rnn_in = z_e.permute(0, 2, 1)
        z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
        # (B, T_latent, C) -> (B, C, T_latent)
        z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
        # ---
        
        z_q, vq_loss, indices = self.quantizer(z_e_rnn_out)

        # --- Stateful RNN ---
        # (B, C, T_latent) -> (B, T_latent, C)
        z_q_rnn_in = z_q.permute(0, 2, 1)
        z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
        # (B, T_latent, C) -> (B, C, T_latent)
        z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
        # ---
        
        x_hat = self.decoder(z_q_rnn_out)
        
        return x_hat, vq_loss, (h_enc_new, h_dec_new)

    def encode(self, x, h_enc):
        """For streaming: encode audio to indices."""
        z_e = self.encoder(x) # (B, C, 40)
        z_e_rnn_in = z_e.permute(0, 2, 1)
        z_e_rnn_out, h_enc_new = self.encoder_rnn(z_e_rnn_in, h_enc)
        z_e_rnn_out = z_e_rnn_out.permute(0, 2, 1)
        
        # We don't need vq_loss here, just indices
        _, _, indices = self.quantizer(z_e_rnn_out) # (B, 40)
        return indices, h_enc_new

    def decode(self, indices, h_dec):
        """For streaming: decode indices to audio."""
        # Convert indices (B, 40) to codebook vectors (B, C, 40)
        z_q = self.quantizer.embedding(indices) # (B, 40, C)
        z_q = z_q.permute(0, 2, 1) # (B, C, 40)
        
        z_q_rnn_in = z_q.permute(0, 2, 1)
        z_q_rnn_out, h_dec_new = self.decoder_rnn(z_q_rnn_in, h_dec)
        z_q_rnn_out = z_q_rnn_out.permute(0, 2, 1)
        
        x_hat = self.decoder(z_q_rnn_out) # (B, 1, 320)
        return x_hat, h_dec_new


# --- MODEL 2: TS3 Codec (Transformer) ---
class TS3_Codec(nn.Module):
    """
    A stateful, causal codec using a Causal Transformer as the core.
    This directly addresses your "Transformer" requirement.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(VQ_EMBEDDINGS, LATENT_DIM, 0.25)
        self.decoder = Decoder()
        
        self.encoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
        self.decoder_tfm = CausalTransformerEncoder(LATENT_DIM, nhead=4, num_layers=2)
    
    def forward(self, x, h_enc=None, h_dec=None):
        # x: (B, 1, T)
        z_e = self.encoder(x) # (B, C, T_latent)
        
        # --- Stateful TFM ---
        z_e_tfm_in = z_e.permute(0, 2, 1) # (B, T_latent, C)
        z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
        z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
        # ---
        
        z_q, vq_loss, indices = self.quantizer(z_e_tfm_out)

        # --- Stateful TFM ---
        z_q_tfm_in = z_q.permute(0, 2, 1) # (B, T_latent, C)
        z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q_tfm_in, h_dec)
        z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, T_latent)
        # ---
        
        x_hat = self.decoder(z_q_tfm_out)
        
        return x_hat, vq_loss, (h_enc_new, h_dec_new)

    def encode(self, x, h_enc):
        """For streaming: encode audio to indices."""
        z_e = self.encoder(x) # (B, C, 40)
        z_e_tfm_in = z_e.permute(0, 2, 1)
        z_e_tfm_out, h_enc_new = self.encoder_tfm(z_e_tfm_in, h_enc)
        z_e_tfm_out = z_e_tfm_out.permute(0, 2, 1)
        
        _, _, indices = self.quantizer(z_e_tfm_out) # (B, 40)
        return indices, h_enc_new

    def decode(self, indices, h_dec):
        """For streaming: decode indices to audio."""
        z_q = self.quantizer.embedding(indices) # (B, 40, C)
        
        # TFM model input is (B, 40, C)
        z_q_tfm_out, h_dec_new = self.decoder_tfm(z_q, h_dec)
        z_q_tfm_out = z_q_tfm_out.permute(0, 2, 1) # (B, C, 40)
        
        x_hat = self.decoder(z_q_tfm_out) # (B, 1, 320)
        return x_hat, h_dec_new

# --- MODEL 3: ScoreDec (Diffusion Post-Filter) ---

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class DiffusionBlock(nn.Module):
    """A single U-Net block for the diffusion model."""
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_dim, out_channels)
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb, cond_emb):
        h = self.relu(self.bn1(self.conv1(x)))
        
        # Add time and condition embeddings
        time_emb = self.relu(self.time_mlp(t_emb))
        cond_emb = self.relu(self.cond_mlp(cond_emb))
        h = h + time_emb.unsqueeze(-1) + cond_emb.unsqueeze(-1)
        
        h = self.relu(self.bn2(self.conv2(h)))
        return h

class DiffusionUNet1D(nn.Module):
    """
    A 1D *CAUSAL* Denoising model, conditioned on the low-quality codec output.
    This is a simple "WaveNet" style stack, not a U-Net, to maintain causality.
    """
    def __init__(self, in_channels=1, model_channels=64, time_emb_dim=256, cond_dim=1):
        super().__init__()
        
        # --- Simple Causal Stack ---
        self.time_mlp_simple = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, model_channels), nn.ReLU())
        self.cond_proj_simple = CausalConv1d(cond_dim, model_channels, 1)
        self.in_conv = CausalConv1d(in_channels, model_channels, 1)
        
        # FIX: Removed padding=1. CausalConv1d handles its own padding.
        # This was the source of the "16002 vs 16000" error.
        self.blocks = nn.ModuleList([
            CausalConv1d(model_channels, model_channels, 3) for _ in range(4) # 4 residual blocks
        ])
        self.out_conv = CausalConv1d(model_channels, in_channels, 1)
        # --- End Simple Causal Stack ---

    def forward(self, x, time, cond):
        # t_emb shape: (B, C, 1)
        t_emb = self.time_mlp_simple(time).unsqueeze(-1) 
        # c_emb shape: (B, C, T)
        c_emb = self.cond_proj_simple(cond) 
        # x_in shape: (B, C, T)
        x_in = self.in_conv(x) 
        
        # Add time and condition embeddings
        h = x_in + t_emb + c_emb 
        for block in self.blocks:
            h = block(h) + h # Residual connection
        return self.out_conv(h)

class ScoreDecPostFilter(nn.Module):
    """
    Wraps the diffusion U-Net and provides the enhancement logic.
    This is what is called by the streaming/evaluation tabs.
    """
    def __init__(self, timesteps=50, model_channels=64):
        super().__init__()
        self.timesteps = timesteps
        self.model = DiffusionUNet1D(model_channels=model_channels)
        
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas', alphas)
        
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: noise the clean signal."""
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(x_start.shape[0], 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().view(x_start.shape[0], 1, 1)
        
        # FIX: Return both the noised tensor and the noise itself
        noised_tensor = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noised_tensor, noise

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        """One step of the reverse diffusion process."""
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        
        predicted_noise = self.model(x_t, t_tensor.float(), cond)
        
        # DDPM sampling step
        x_prev = (x_t - ((1-alpha_t) / (1-alpha_cumprod_t).sqrt()) * predicted_noise) / alpha_t.sqrt()
        
        if t > 0:
            noise = torch.randn_like(x_t)
            alpha_cumprod_prev_t = self.alphas_cumprod[t-1]
            posterior_variance = (1-alpha_cumprod_prev_t) / (1-alpha_cumprod_t) * self.betas[t]
            x_prev += torch.sqrt(posterior_variance.view(-1, 1, 1)) * noise
        return x_prev

    @torch.no_grad()
    def enhance(self, x_low_quality, timesteps=10):
        """
        The main enhancement function.
        x_low_quality is the output from the GRU_Codec.
        This is SLOW and NOT real-time.
        """
        # Start from the low-quality audio, but add noise
        t_start = timesteps - 1
        
        # FIX: Unpack the tuple from q_sample. We only need the noised tensor.
        # This resolves the "'tuple' object has no attribute 'shape'" error.
        x_t, _ = self.q_sample(x_low_quality, torch.tensor([t_start], device=x_low_quality.device))
        
        for i in reversed(range(timesteps)):
            x_t = self.p_sample(x_t, i, cond=x_low_quality)
            
        return torch.tanh(x_t)


# --- TRADITIONAL CODECS (For Baseline Comparison) ---
class MuLawCodec:
    def __init__(self, quantization_channels=256): self.mu = float(quantization_channels - 1)
    def encode(self, x):
        mu_t = torch.tensor(self.mu, device=x.device, dtype=torch.float32)
        encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
        return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
    def decode(self, z):
        z_float = z.to(torch.float32)
        mu_t = torch.tensor(self.mu, device=z.device, dtype=torch.float32)
        y = (z_float / self.mu) * 2.0 - 1.0
        return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

class ALawCodec:
    def __init__(self): self.A = 87.6
    def encode(self, x):
        a_t = torch.tensor(self.A, device=x.device, dtype=torch.float32)
        abs_x = torch.abs(x)
        encoded = torch.zeros_like(x)
        cond = abs_x < (1 / self.A)
        encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
        encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
        return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
    def decode(self, z):
        z_float = z.to(torch.float32)
        a_t = torch.tensor(self.A, device=z.device, dtype=torch.float32)
        y = (z_float / 127.5) - 1.0
        abs_y = torch.abs(y)
        decoded = torch.zeros_like(y)
        cond = abs_y < (1 / (1 + torch.log(a_t)))
        decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
        decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
        return decoded.unsqueeze(1)

# --- DATASET & TRAINING ---
TRAIN_CHUNK_SIZE = 16000 # 1 second

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
            if waveform.shape[1] > self.chunk_size:
                start = np.random.randint(0, waveform.shape[1] - self.chunk_size)
                waveform = waveform[:, start:start + self.chunk_size]
            else:
                waveform = F.pad(waveform, (0, self.chunk_size - waveform.shape[1]))
            return waveform
        except Exception as e:
            print(f"Warning: Skipping file {self.file_paths[idx]}. Error: {e}")
            return torch.zeros((1, self.chunk_size))

def train_model(dataset_path, epochs, learning_rate, batch_size, model_save_path, progress_callback, stop_event, model_type):
    """
    Main training function. Now handles 'gru', 'transformer', and 'scoredec' types.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        progress_callback.emit(f"Using device: {device}")
        
        dataset = AudioChunkDataset(directory=dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        progress_callback.emit(f"Dataset loaded with {len(dataset)} files.")

        if model_type == 'gru':
            model = GRU_Codec().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            stft_criterion = MultiResolutionSTFTLoss().to(device)
            l1_criterion = nn.L1Loss().to(device)
            progress_callback.emit(f"Starting training for GRU_Codec model...")
        
        elif model_type == 'transformer':
            model = TS3_Codec().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            stft_criterion = MultiResolutionSTFTLoss().to(device)
            l1_criterion = nn.L1Loss().to(device)
            progress_callback.emit(f"Starting training for TS3_Codec model...")
        
        elif model_type == 'scoredec':
            progress_callback.emit("--- Starting ScoreDec Post-Filter Training ---")
            progress_callback.emit("Loading pre-trained GRU_Codec...")
            try:
                gru_codec = GRU_Codec().to(device)
                gru_codec.load_state_dict(torch.load("low_latency_codec_gru.pth", map_location=device))
                gru_codec.eval()
                for param in gru_codec.parameters():
                    param.requires_grad = False
                progress_callback.emit("GRU_Codec loaded and frozen.")
            except FileNotFoundError:
                progress_callback.emit("ERROR: 'low_latency_codec_gru.pth' not found. You must train the GRU_Codec first.")
                return
            
            model = ScoreDecPostFilter().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            l1_criterion = nn.L1Loss().to(device)
            progress_callback.emit("Starting training for ScoreDec model...")
            
        else:
            raise ValueError(f"Unknown model type for training: {model_type}")

        # --- Main Training Loop ---
        for epoch in range(epochs):
            if stop_event.is_set():
                progress_callback.emit("Training stopped by user."); break
            
            for i, data in enumerate(dataloader):
                inputs = data.to(device)
                optimizer.zero_grad()
                
                if model_type in ['gru', 'transformer']:
                    h_enc, h_dec = None, None # Reset state per batch
                    x_hat, vq_loss, (h_enc, h_dec) = model(inputs, h_enc, h_dec)
                    
                    # --- FIX: Defensively pad output to match input length ---
                    # This is the most likely source of the STFT error
                    input_len = inputs.shape[-1]
                    output_len = x_hat.shape[-1]
                    
                    if output_len < input_len:
                        # Pad x_hat on the right if it's shorter
                        padding = input_len - output_len
                        x_hat = F.pad(x_hat, (0, padding))
                    elif output_len > input_len:
                        # Trim x_hat if it's longer
                        x_hat = x_hat[..., :input_len]
                    # --- End Fix ---
                    
                    stft_loss = stft_criterion(x_hat, inputs)
                    l1_loss = l1_criterion(x_hat, inputs)
                    loss = stft_loss + 0.1 * l1_loss + vq_loss
                    
                    if i % 20 == 19:
                        progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss.item():.5f} (STFT: {stft_loss.item():.4f}, VQ: {vq_loss.item():.4f})")

                elif model_type == 'scoredec':
                    with torch.no_grad():
                        x_hat_low_quality, _, _ = gru_codec(inputs)
                        
                        # --- FIX: Ensure low-quality output matches input ---
                        input_len = inputs.shape[-1]
                        output_len = x_hat_low_quality.shape[-1]
                        if output_len < input_len:
                            padding = input_len - output_len
                            x_hat_low_quality = F.pad(x_hat_low_quality, (0, padding))
                        elif output_len > input_len:
                            x_hat_low_quality = x_hat_low_quality[..., :input_len]
                        # --- End Fix ---
                        
                        x_hat_low_quality = x_hat_low_quality.detach()
                    
                    # Train diffusion model
                    t = torch.randint(0, model.timesteps, (inputs.shape[0],), device=device).long()
                    
                    # This line was the source of the "unpack" error
                    # It is now fixed because q_sample returns (x_t, noise)
                    x_t, noise = model.q_sample(x_start=inputs, t=t)
                    
                    predicted_noise = model.model(x_t, t.float(), cond=x_hat_low_quality)
                    loss = l1_criterion(predicted_noise, noise)
                    
                    if i % 20 == 19:
                        progress_callback.emit(f"[Epoch {epoch + 1}, Batch {i + 1}] Denoising Loss: {loss.item():.5f}")

                loss.backward()
                optimizer.step()

            progress_callback.emit(f"--- Epoch {epoch + 1} finished ---")

        if not stop_event.is_set():
            progress_callback.emit("Training finished. Saving model...")
            torch.save(model.state_dict(), model_save_path)
            progress_callback.emit(f"Model saved to {model_save_path}")
    except Exception as e:
        progress_callback.emit(f"ERROR in training: {e}")

