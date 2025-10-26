# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# import os
# import struct
# import subprocess
# import sys

# # --- Global Configuration ---
# HOP_SIZE = 160 # 10ms frame (160 samples / 16000 Hz = 0.01s)
# SR = 16000
# CHANNELS = 1
# LATENT_DIM = 64
# BLOCKS = 3
# HEADS = 4
# KERNEL_SIZE = 3
# STRIDES = [2, 2, 2, 2] # 16x total downsampling

# # Models cache directory
# MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
# if not os.path.exists(MODELS_CACHE_DIR):
#     os.makedirs(MODELS_CACHE_DIR)

# # --- DAC INTEGRATION ---
# try:
#     import dac
#     DAC_AVAILABLE = True
# except ImportError:
#     DAC_AVAILABLE = False
#     print("Warning: DAC not available. Install with: pip install descript-audio-codec")

# # --- ENCODEC INTEGRATION ---
# try:
#     import encodec
#     ENCODEC_AVAILABLE = True
#     print("EnCodec available (official Meta implementation)")
# except ImportError:
#     ENCODEC_AVAILABLE = False
#     print("Warning: EnCodec not available. Install with: pip install encodec")

# # --- CUSTOM TINY CODEC COMPONENTS (COPIED from tiny_codec_train.py for standalone model file) ---
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super().__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
#         self.norm = nn.GroupNorm(1, out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.relu(self.norm(self.conv(x)))

# class TransformerBlock(nn.Module):
#     def __init__(self, dim, heads):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim)
#         )

#     def forward(self, x):
#         # x is (B, C, T) -> transpose to (B, T, C) for attention
#         x_attn = x.transpose(1, 2) 
#         attn_output, _ = self.attn(x_attn, x_attn, x_attn)
#         x_attn = self.norm1(x_attn + attn_output)
#         ffn_output = self.ffn(x_attn)
#         x_attn = self.norm2(x_attn + ffn_output)
#         # transpose back to (B, C, T)
#         return x_attn.transpose(1, 2) 

# class TinyTransformerCodec(nn.Module):
#     """
#     Ultra-Lightweight Neural Audio Codec with Transformer Bottleneck.
#     Input: (B, 1, T) raw audio (normalized)
#     Output: (B, 1, T) reconstructed audio
#     Latency: Designed for 10-20ms chunks.
#     """
#     def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS, sr=SR):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.sr = sr
#         self.downsampling_factor = np.prod(STRIDES)

#         # Encoder (CNN-based)
#         self.encoder_convs = nn.ModuleList()
#         in_c = CHANNELS
#         current_c = latent_dim // (2**(blocks-1))
#         for i in range(blocks):
#             out_c = min(latent_dim, current_c * 2**i)
#             self.encoder_convs.append(ConvBlock(in_c, out_c, KERNEL_SIZE, STRIDES[i]))
#             in_c = out_c
        
#         self.pre_transformer = ConvBlock(in_c, latent_dim, KERNEL_SIZE, 1)

#         # Transformer Bottleneck
#         self.transformer = TransformerBlock(latent_dim, heads)

#         # Decoder (Transposed CNN-based)
#         self.decoder_tconvs = nn.ModuleList()
#         in_c = latent_dim
#         for i in reversed(range(blocks)):
#             out_c = self.encoder_convs[i].conv.in_channels
#             self.decoder_tconvs.append(
#                 nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, STRIDES[i], padding=STRIDES[i]//2, output_padding=STRIDES[i]//2)
#             )
#             in_c = out_c
        
#         self.post_decoder = nn.Conv1d(in_c, CHANNELS, 1)

#     @classmethod
#     def load_model(cls, model_path):
#         """Loads the model weights and returns the initialized model."""
#         model = cls()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if not os.path.exists(model_path):
#              raise FileNotFoundError(f"Trained model not found at path: {model_path}")
             
#         try:
#              # Load state_dict from a checkpoint (which contains 'model_state_dict')
#             checkpoint = torch.load(model_path, map_location=device)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             model.to(device)
#             model.eval()
#             print(f"TinyTransformerCodec loaded successfully from {model_path}.")
#             return model
#         except Exception as e:
#             print(f"Error loading model state dict: {e}")
#             raise

#     def encode(self, x):
#         x = x.view(x.size(0), CHANNELS, -1)

#         for layer in self.encoder_convs:
#             x = layer(x)
        
#         x = self.pre_transformer(x)
#         codes = self.transformer(x)
        
#         # We use sequence length for later cropping
#         return codes, x.shape[-1] 

#     def decode(self, codes, original_length=None):
#         x = codes
        
#         # Ensure codes is the tensor, not the tuple/list format sometimes used in streaming
#         if isinstance(codes, (list, tuple)):
#              codes = codes[0]
        
#         # CNN Decoding
#         for i, layer in enumerate(self.decoder_tconvs):
#             x = F.relu(layer(x))

#         x = torch.tanh(self.post_decoder(x))
        
#         if original_length is not None:
#              target_len = original_length
#              if x.shape[-1] > target_len:
#                  x = x[..., :target_len]
        
#         return x.view(x.size(0), CHANNELS, -1)


# # --- TRADITIONAL CODECS (Baseline Comparison) ---
# class MuLawCodec:
#     """μ-law codec for baseline comparison"""
#     def __init__(self, quantization_channels=256):  
#         self.mu = float(quantization_channels - 1)
    
#     def encode(self, x):
#         mu_t = torch.tensor(self.mu, dtype=torch.float32, device=x.device)
#         encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
#         return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
    
#     def decode(self, z):
#         z_float = z.to(torch.float32)
#         mu_t = torch.tensor(self.mu, dtype=torch.float32, device=z.device)
#         y = (z_float / self.mu) * 2.0 - 1.0
#         return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

# class ALawCodec:
#     """A-law codec for baseline comparison"""
#     def __init__(self):  
#         self.A = 87.6
    
#     def encode(self, x):
#         a_t = torch.tensor(self.A, dtype=torch.float32, device=x.device)
#         abs_x = torch.abs(x)
#         encoded = torch.zeros_like(x)
#         cond = abs_x < (1 / self.A)
#         encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
#         encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
#         return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
    
#     def decode(self, z):
#         z_float = z.to(torch.float32)
#         a_t = torch.tensor(self.A, dtype=torch.float32, device=z.device)
#         y = (z_float / 127.5) - 1.0
#         abs_y = torch.abs(y)
#         decoded = torch.zeros_like(y)
#         cond = abs_y < (1 / (1 + torch.log(a_t)))
#         decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
#         decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
#         return decoded.unsqueeze(1)

# # --- DAC CODEC (EXTERNAL) ---
# class DACCodec:
#     """Wrapper for Descript Audio Codec (DAC)"""
#     def __init__(self, model_path=None, model_type="16khz"):
#         if not DAC_AVAILABLE:
#             raise ImportError("DAC is not installed. Install with: pip install descript-audio-codec")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_type = model_type
#         print(f"Loading DAC {model_type} model...")
#         try:
#             model_path = dac.utils.download(model_type=model_type)
#             self.model = dac.DAC.load(model_path)
#             print(f"DAC model loaded successfully")
#         except Exception as e:
#             print(f"Error loading DAC model: {e}")
#             raise
#         self.model.to(self.device)
#         self.model.eval()
#         self.sample_rate = 16000 if "16khz" in model_type else 44100
#         self.hop_size = 320
#         self.chunk_size = 320
        
#     def encode(self, audio_tensor):
#         with torch.no_grad():
#             if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
#             audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
#             original_length = audio_tensor.shape[-1]
#             if original_length % self.hop_size != 0:
#                 pad_length = self.hop_size - (original_length % self.hop_size)
#                 audio_tensor = F.pad(audio_tensor, (0, pad_length))
#             _, codes, _, _, _ = self.model.encode(audio_tensor)
#             return codes, original_length
    
#     def decode(self, codes, original_length=None):
#         with torch.no_grad():
#             if not isinstance(codes, torch.Tensor): codes = torch.tensor(codes, dtype=torch.long)
#             codes = codes.to(self.device)
#             z = self.model.quantizer.from_codes(codes)[0]
#             audio_recon = self.model.decode(z)
#             if original_length is not None and audio_recon.shape[-1] > original_length:
#                 audio_recon = audio_recon[..., :original_length]
#             return audio_recon

# # --- ENCODEC CODEC (EXTERNAL) ---
# class EnCodecModel:
#     """Official Meta EnCodec implementation"""
#     def __init__(self, model_path=None, bandwidth=6.0, model_type="encodec_24khz"):
#         if not ENCODEC_AVAILABLE:
#             raise ImportError("EnCodec is not installed. Install with: pip install encodec")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.bandwidth = bandwidth
#         print(f"Loading EnCodec model ({model_type}, {bandwidth} kbps)...")
#         try:
#             if "48khz" in model_type: self.model = encodec.EncodecModel.encodec_model_48khz(); self.sample_rate = 48000
#             else: self.model = encodec.EncodecModel.encodec_model_24khz(); self.sample_rate = 24000
#             print(f"EnCodec model loaded successfully (sample rate: {self.sample_rate} Hz)")
#         except Exception as e:
#             print(f"Error loading EnCodec model: {e}"); raise ImportError(f"Could not load EnCodec model: {e}")
#         self.model.to(self.device)
#         self.model.eval()
#         self.model.set_target_bandwidth(self.bandwidth)
#         self.input_sample_rate = 16000
        
#     def encode(self, audio_tensor):
#         with torch.no_grad():
#             if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
#             elif audio_tensor.dim() == 1: audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(1)
#             audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
#             original_length = audio_tensor.shape[-1]
#             if self.input_sample_rate != self.sample_rate:
#                 import torchaudio
#                 audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=self.input_sample_rate, new_freq=self.sample_rate)
#             encoded_frames = self.model.encode(audio_tensor)
#             return encoded_frames, original_length
    
#     def decode(self, encoded_frames, original_length=None):
#         with torch.no_grad():
#             audio_values = self.model.decode(encoded_frames)
#             if isinstance(audio_values, (list, tuple)): audio_values = audio_values[0]
#             if self.input_sample_rate != self.sample_rate:
#                 import torchaudio
#                 audio_values = torchaudio.functional.resample(audio_values, orig_freq=self.sample_rate, new_freq=self.input_sample_rate)
#             if original_length is not None and audio_values.shape[-1] > original_length:
#                 audio_values = audio_values[..., :original_length]
#             return audio_values

# # --- UTILITY FUNCTIONS (unchanged) ---
# def get_model_cache_info():
#     """Returns information about cached models"""
#     cache_info = {}
#     if os.path.exists(MODELS_CACHE_DIR):
#         for file in os.listdir(MODELS_CACHE_DIR):
#             file_path = os.path.join(MODELS_CACHE_DIR, file)
#             if os.path.isfile(file_path):
#                 file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
#                 cache_info[file] = f"{file_size:.2f} MB"
#     return cache_info

# def clear_model_cache():
#     """Clears all cached models"""
#     import shutil
#     if os.path.exists(MODELS_CACHE_DIR):
#         shutil.rmtree(MODELS_CACHE_DIR)
#         os.makedirs(MODELS_CACHE_DIR)
#         print("Model cache cleared.")


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import struct
import subprocess
import sys
import math

# --- Global Configuration (Matching tiny_codec_trainer.py) ---
SR = 16000
CHANNELS = 1
LATENT_DIM = 64
BLOCKS = 4 # Increased from 3
HEADS = 4
KERNEL_SIZE = 3
STRIDES = [2, 2, 4, 2] # Total DOWN_FACTOR = 32
DOWN_FACTOR = np.prod(STRIDES)
NUM_CODEBOOKS = 2
CODEBOOK_SIZE = 512
COMMITMENT_COST = 1.0
TRANSFORMER_BLOCKS = 3

# Calculated HOP size matching the downsampling factor (SR / DOWN_FACTOR = 500 Hz -> 2ms)
# The trainer uses CHUNK_DURATION=0.04s (640 samples) which is 20 latent frames.
HOP_SIZE = int(SR / DOWN_FACTOR) * 2 # Set to 10ms (160 samples) for low latency streaming

# Models cache directory
MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
if not os.path.exists(MODELS_CACHE_DIR):
    os.makedirs(MODELS_CACHE_DIR)

# --- DAC INTEGRATION (Unchanged) ---
try:
    import dac
    DAC_AVAILABLE = True
except ImportError:
    DAC_AVAILABLE = False

# --- ENCODEC INTEGRATION (Unchanged) ---
try:
    import encodec
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False

# --- CUSTOM TINY VQ-CODEC COMPONENTS (Causal Architecture) ---
class CausalConvBlock(nn.Module):
    """Causal Conv1d block matching the architecture from the trainer script."""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.padding_amount = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.norm = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        # Apply padding only to the left (causal)
        x = F.pad(x, (self.padding_amount, 0), mode='constant', value=0)
        x = self.relu(self.norm(self.conv(x)))
        return x

class CausalTransformerBlock(nn.Module):
    """Causal Transformer block matching the architecture from the trainer script."""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, T = x.shape
        x_attn = x.transpose(1, 2)
        
        # Causal mask (upper triangle, diagonal=1 attends to self and past)
        attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
        attn_output, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=attn_mask, is_causal=False)
        x_attn = self.norm1(x_attn + attn_output)
        
        ffn_output = self.ffn(x_attn)
        x_attn = self.norm2(x_attn + ffn_output)
        
        return x_attn.transpose(1, 2)

class VectorQuantizer(nn.Module):
    """Vector Quantization layer with Straight-Through Estimator (STE)."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=COMMITMENT_COST):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # Flatten input (B, C, T) -> (B*T, C)
        flat_input = inputs.transpose(1, 2).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                      + torch.sum(self.embedding.weight**2, dim=1)
                      - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Find the nearest codebook vector index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Create one-hot vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize vector
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2], -1).transpose(1, 2)
        
        # Apply STE: pass the quantized vector but calculate gradients w.r.t. the input (only relevant during training)
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, encoding_indices

class TinyTransformerCodec(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS, sr=SR):
        super().__init__()
        self.latent_dim = latent_dim
        self.sr = sr
        self.downsampling_factor = DOWN_FACTOR
        self.num_codebooks = NUM_CODEBOOKS

        # --- Encoder ---
        self.encoder_convs = nn.ModuleList()
        in_c = CHANNELS
        encoder_channels = []
        for i in range(blocks):
            out_c = min(latent_dim, 8 * (2**i)) 
            encoder_channels.append(out_c)
            stride = STRIDES[i]
            self.encoder_convs.append(CausalConvBlock(in_c, out_c, KERNEL_SIZE, stride))
            in_c = out_c
        
        self.pre_quant = CausalConvBlock(in_c, LATENT_DIM * NUM_CODEBOOKS, KERNEL_SIZE, 1)

        # --- Vector Quantization ---
        self.quantizers = nn.ModuleList([
            VectorQuantizer(CODEBOOK_SIZE, LATENT_DIM, commitment_cost=COMMITMENT_COST)
            for _ in range(NUM_CODEBOOKS)
        ])
        
        # --- Transformer ---
        self.transformer = nn.Sequential(*[
            CausalTransformerBlock(latent_dim * NUM_CODEBOOKS, heads)
            for _ in range(TRANSFORMER_BLOCKS)
        ])
        self.post_transformer = nn.Conv1d(latent_dim * NUM_CODEBOOKS, latent_dim * NUM_CODEBOOKS, 1)

        # --- Decoder ---
        self.decoder_tconvs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        in_c = latent_dim * NUM_CODEBOOKS
        for i in range(blocks):
            idx = blocks - 1 - i
            stride = STRIDES[idx]
            
            if idx > 0:
                out_c = encoder_channels[idx - 1]
            else:
                out_c = 16 
            
            self.decoder_tconvs.append(
                nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, stride, padding=KERNEL_SIZE//2)
            )
            
            if idx > 0:
                skip_in_channels = encoder_channels[idx - 1]
                self.skip_convs.append(
                    nn.Conv1d(out_c + skip_in_channels, out_c, kernel_size=1)
                )
            in_c = out_c
        
        self.post_decoder_final = nn.Conv1d(in_c, CHANNELS, 1)

    @classmethod
    def load_model(cls, model_path):
        """Loads the model weights and returns the initialized model."""
        model = cls()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at path: {model_path}")
            
        try:
            # Load state_dict, supporting both full checkpoint and model_state_dict
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"TinyTransformerCodec (VQ-Codec) loaded successfully from {model_path}.")
            return model
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            raise

    def encode(self, x):
        """
        Encodes audio into quantized latent codes/indices.
        Returns:
            quantized_codes (Tensor): The VQ-quantized float tensor (B, C, T_latent). Used for evaluation.
            indices_list (list): List of integer index tensors (compressed data). Used for streaming.
            input_length (int): Original audio length.
            encoder_outputs (list): Skip connection features.
        """
        x = x.view(x.size(0), CHANNELS, -1)
        input_length = x.shape[-1]
        encoder_outputs = []
        
        # Encoder
        for layer in self.encoder_convs:
            x = layer(x)
            encoder_outputs.append(x)
        
        # Pre-quantization
        z_e = self.pre_quant(x)
        
        # Vector Quantization
        z_q_list = []
        indices_list = []
        z_e_split = z_e.chunk(self.num_codebooks, dim=1)
        
        for i in range(self.num_codebooks):
            # In app mode, we ignore the VQ loss, only get quantized and indices
            z_q, indices = self.quantizers[i](z_e_split[i]) 
            z_q_list.append(z_q)
            indices_list.append(indices)
        
        # Concatenated quantized latent features (used to enter Transformer/Decoder path)
        quantized_codes = torch.cat(z_q_list, dim=1)
        
        return quantized_codes, indices_list, input_length, encoder_outputs

    def decode(self, codes_or_indices_list, input_length=None, encoder_outputs=None):
        """
        Decodes from VQ-quantized float codes (for evaluation) OR integer indices (for streaming).
        
        Args:
            codes_or_indices_list (Tensor or list): The full float tensor from encode() 
                                                    OR the list of integer index tensors from encode().
            input_length (int): The target output length.
            encoder_outputs (list): Skip connection features (only used if decoding from full codes).
        """
        
        if isinstance(codes_or_indices_list, list):
            # Case 1: Decoding from raw integer indices (streaming receiver)
            indices_list = codes_or_indices_list
            z_q_list = []
            
            for i, indices in enumerate(indices_list):
                # Lookup indices in the corresponding codebook's embedding weights
                quantized = self.quantizers[i].embedding(indices.squeeze(1))
                # Reshape back to (B, C, T)
                z_q_list.append(quantized.transpose(1, 2))
            
            x = torch.cat(z_q_list, dim=1)
            
            # NOTE: Skip connections are typically omitted when decoding only from indices (streaming)
            encoder_outputs = None 
            
        elif isinstance(codes_or_indices_list, torch.Tensor):
            # Case 2: Decoding from the quantized float tensor (evaluation or training loss)
            x = codes_or_indices_list
        else:
            raise ValueError("Decoding input must be a torch.Tensor (quantized codes) or a list of Tensors (indices).")

        # Transformer
        x = self.transformer(x)
        x = self.post_transformer(x)
        
        # Decoder with skip connections
        for i, tconv in enumerate(self.decoder_tconvs):
            x = F.relu(tconv(x))
            
            # Apply skip connection if available (for evaluation/full reconstruction path)
            if encoder_outputs and i < len(self.skip_convs):
                encoder_idx = len(self.encoder_convs) - 2 - i
                
                if 0 <= encoder_idx < len(encoder_outputs):
                    skip_features = encoder_outputs[encoder_idx]
                    min_len = min(skip_features.shape[-1], x.shape[-1])
                    skip_features = skip_features[..., :min_len]
                    x_trim = x[..., :min_len]
                    
                    x_cat = torch.cat([x_trim, skip_features], dim=1)
                    x_processed = self.skip_convs[i](x_cat)
                    
                    if x.shape[-1] > min_len:
                        x = torch.cat([x_processed, x[..., min_len:]], dim=-1)
                    else:
                        x = x_processed
        
        # Final output
        x = torch.tanh(self.post_decoder_final(x))
        
        # Match input length
        if input_length is not None:
            if x.shape[-1] > input_length:
                x = x[..., :input_length]
            elif x.shape[-1] < input_length:
                x = F.pad(x, (0, input_length - x.shape[-1]))
        
        return x.view(x.size(0), CHANNELS, -1)

# --- TRADITIONAL CODECS (Unchanged) ---
class MuLawCodec:
    """μ-law codec for baseline comparison"""
    def __init__(self, quantization_channels=256):
        self.mu = float(quantization_channels - 1)
    
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
    """A-law codec for baseline comparison"""
    def __init__(self):
        self.A = 87.6
    
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

# --- DAC CODEC (Unchanged) ---
class DACCodec:
    """Wrapper for Descript Audio Codec (DAC)"""
    def __init__(self, model_path=None, model_type="16khz"):
        if not DAC_AVAILABLE:
            raise ImportError("DAC is not installed. Install with: pip install descript-audio-codec")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        print(f"Loading DAC {model_type} model...")
        try:
            model_path = dac.utils.download(model_type=model_type)
            self.model = dac.DAC.load(model_path)
            print(f"DAC model loaded successfully")
        except Exception as e:
            print(f"Error loading DAC model: {e}")
            raise
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = 16000 if "16khz" in model_type else 44100
        self.hop_size = 320
        self.chunk_size = 320
        
    def encode(self, audio_tensor):
        with torch.no_grad():
            if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
            audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
            original_length = audio_tensor.shape[-1]
            if original_length % self.hop_size != 0:
                pad_length = self.hop_size - (original_length % self.hop_size)
                audio_tensor = F.pad(audio_tensor, (0, pad_length))
            _, codes, _, _, _ = self.model.encode(audio_tensor)
            return codes, original_length # Returns the integer codes tensor and original length
    
    def decode(self, codes, original_length=None):
        with torch.no_grad():
            if not isinstance(codes, torch.Tensor): codes = torch.tensor(codes, dtype=torch.long)
            codes = codes.to(self.device)
            z = self.model.quantizer.from_codes(codes)[0]
            audio_recon = self.model.decode(z)
            if original_length is not None and audio_recon.shape[-1] > original_length:
                audio_recon = audio_recon[..., :original_length]
            return audio_recon

# --- ENCODEC CODEC (Unchanged) ---
class EnCodecModel:
    """Official Meta EnCodec implementation"""
    def __init__(self, model_path=None, bandwidth=6.0, model_type="encodec_24khz"):
        if not ENCODEC_AVAILABLE:
            raise ImportError("EnCodec is not installed. Install with: pip install encodec")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bandwidth = bandwidth
        print(f"Loading EnCodec model ({model_type}, {bandwidth} kbps)...")
        try:
            if "48khz" in model_type: self.model = encodec.EncodecModel.encodec_model_48khz(); self.sample_rate = 48000
            else: self.model = encodec.EncodecModel.encodec_model_24khz(); self.sample_rate = 24000
            print(f"EnCodec model loaded successfully (sample rate: {self.sample_rate} Hz)")
        except Exception as e:
            print(f"Error loading EnCodec model: {e}"); raise ImportError(f"Could not load EnCodec model: {e}")
        self.model.to(self.device)
        self.model.eval()
        self.model.set_target_bandwidth(self.bandwidth)
        self.input_sample_rate = 16000
        
    def encode(self, audio_tensor):
        with torch.no_grad():
            if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
            elif audio_tensor.dim() == 1: audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(1)
            audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
            original_length = audio_tensor.shape[-1]
            if self.input_sample_rate != self.sample_rate:
                import torchaudio
                audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=self.input_sample_rate, new_freq=self.sample_rate)
            encoded_frames = self.model.encode(audio_tensor)
            return encoded_frames, original_length
    
    def decode(self, encoded_frames, original_length=None):
        with torch.no_grad():
            audio_values = self.model.decode(encoded_frames)
            if isinstance(audio_values, (list, tuple)): audio_values = audio_values[0]
            if self.input_sample_rate != self.sample_rate:
                import torchaudio
                audio_values = torchaudio.functional.resample(audio_values, orig_freq=self.sample_rate, new_freq=self.input_sample_rate)
            if original_length is not None and audio_values.shape[-1] > original_length:
                audio_values = audio_values[..., :original_length]
            return audio_values

# --- UTILITY FUNCTIONS (Unchanged) ---
def get_model_cache_info():
    """Returns information about cached models"""
    cache_info = {}
    if os.path.exists(MODELS_CACHE_DIR):
        for file in os.listdir(MODELS_CACHE_DIR):
            file_path = os.path.join(MODELS_CACHE_DIR, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                cache_info[file] = f"{file_size:.2f} MB"
    return cache_info

def clear_model_cache():
    """Clears all cached models"""
    import shutil
    if os.path.exists(MODELS_CACHE_DIR):
        shutil.rmtree(MODELS_CACHE_DIR)
        os.makedirs(MODELS_CACHE_DIR)
        print("Model cache cleared.")
