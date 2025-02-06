#### This script trains diffusion transformer for midi sequence generation based on VQVAE models passed to it


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mido import MidiFile, MidiTrack, Message
import os
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, repeat

import pretty_midi
import glob
import copy
from copy import deepcopy

import math

from sia_vqvae_diff import BarVQVAE

MAX_TIME = 2.0  # Maximum time between events in seconds
MAX_VELOCITY = 127
PITCH_RANGE = 128
EVENT_TYPES = 3  # note_on, note_off, time_shift


class BarSequenceDataset(Dataset):
    def __init__(self, midi_folder, vae_model, bars_per_sequence=8, bar_length=512, device='cuda'):
        self.bars_per_sequence = bars_per_sequence
        self.vae_model = vae_model
        self.device = device
        self.bar_length = bar_length
        self.sequences = self.prepare_sequences(midi_folder)

    def extract_bars_from_midi(self, midi_file):
        mid = MidiFile(midi_file)
        ticks_per_bar = mid.ticks_per_beat * 4  # Assuming 4/4 time signature
        
        C4 = 60 ### only extract right hand

        # Collect all MIDI events
        events = []
        for track in mid.tracks:
            time = 0
            for msg in track:
                time += msg.time
                if msg.type in ['note_on', 'note_off'] and msg.note >= C4:
                    events.append((time, msg))
        
        # Sort events by time
        events.sort(key=lambda x: x[0])
        
        # Group events into bars
        bars = []
        current_bar_events = []
        current_bar_number = 0
        
        for time, msg in events:
            bar_number = time // ticks_per_bar
            
            if bar_number > current_bar_number:
                # Process the current bar
                if current_bar_events:
                    # Convert to fixed-length representation
                    bar_tensor = self.process_bar_events(current_bar_events)
                    bars.append(bar_tensor)
                current_bar_events = []
                current_bar_number = bar_number
            
            # Add event to current bar
            current_bar_events.append((time % ticks_per_bar, msg))
    
        # Process the last bar
        if current_bar_events:
            bar_tensor = self.process_bar_events(current_bar_events)
            bars.append(bar_tensor)
            
        return bars

    def process_bar_events(self, bar_events):
        # Create a fixed-length tensor for the bar
        bar_tensor = torch.zeros((self.bar_length, 2))
        
        # Calculate how many events we can fit
        num_events = min(len(bar_events), self.bar_length)
        
        for i in range(num_events):
            time, msg = bar_events[i]
            event_type = 0 if msg.type == 'note_on' else 1
            bar_tensor[i] = torch.tensor([event_type, msg.note / 127.0])
        
        return bar_tensor
    

        
    #     return sequences
    def prepare_sequences(self, midi_folder):
        sequences = []
        for filename in os.listdir(midi_folder):
            if filename.endswith(('.mid', '.midi')):
                file_path = os.path.join(midi_folder, filename)
                try:
                    bars = self.extract_bars_from_midi(file_path)
                    
                    # Original sequence
                    sequences.extend(self.create_sequences(self.encode_bars(bars)))
                    
                    #### discrete diffusion
                    # list_bar_indices = self.encode_bars_indices(bars)
                    # created_seq = self.create_sequences(list_bar_indices) 
                    # # sequences.exted(created_seq)
                    
                    # Transpositions
                    for transpose in [-3, -2, -1, 0, 1, 2, 3]:
                        transposed_bars = [self.transpose_bar(bar, transpose) for bar in bars]
                        sequences.extend(self.create_sequences(self.encode_bars(transposed_bars)))
                        
                        #### discrete diffusion
                        # self.create_sequences(self.encode_bars_indices(transposed_bars))
                    
                    # Time stretching
                    for stretch_factor in [0.9, 1.1]:
                        stretched_bars = [self.time_stretch(bar, stretch_factor) for bar in bars]
                        sequences.extend(self.create_sequences(self.encode_bars(stretched_bars)))
                        
                        #### discrete diffusion
                        # self.create_sequences(self.encode_bars_indices(stretched_bars))
                    
                    # # Velocity adjustments
                    # for velocity_factor in [0.8, 1.2]:
                    #     velocity_bars = [self.velocity_adjust(bar, velocity_factor) for bar in bars]
                    #     sequences.extend(self.create_sequences(self.encode_bars(velocity_bars)))
                    
                    # # Reversed sequences
                    # reversed_bars = self.reverse_sequence(bars)
                    # sequences.extend(self.create_sequences(self.encode_bars(reversed_bars)))
                    
                    # # Rotated sequences
                    # for rotation in range(1, 4):
                    #     rotated_bars = self.rotate_sequence(bars, rotation)
                    #     sequences.extend(self.create_sequences(self.encode_bars(rotated_bars)))
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
                
        return sequences
    
    
    

    def transpose_bar(self, bar, transpose_amount):
        """
        Transpose a bar by a given number of semitones
        bar: tensor of shape [bar_length, 2] containing (event_type, note) pairs
        transpose_amount: number of semitones to transpose (-12 to 12)
        """
        # Create a copy of the bar
        transposed = bar.clone()
        
        # Only transpose note values (second column) for note events (event_type 0 or 1)
        note_events_mask = (bar[:, 0] == 0) | (bar[:, 0] == 1)
        if note_events_mask.any():
            # Convert normalized note values back to MIDI note numbers
            notes = bar[note_events_mask, 1] * 127
            # Transpose
            notes = notes + transpose_amount
            # Clip to valid MIDI note range
            notes = torch.clamp(notes, 0, 127)
            # Normalize back
            transposed[note_events_mask, 1] = notes / 127

        return transposed
    


    def time_stretch(self, bar, factor):
        """Stretch or compress time values"""
        stretched = bar.clone()
        time_events_mask = (bar[:, 0] == 2)
        if time_events_mask.any():
            time_values = bar[time_events_mask, 1]
            time_values = time_values * factor
            stretched[time_events_mask, 1] = torch.clamp(time_values, 0, 1)
        return stretched

    def velocity_adjust(self, bar, scale_factor):
        """Adjust velocity of note-on events"""
        adjusted = bar.clone()
        note_on_mask = (bar[:, 0] == 0)
        if note_on_mask.any():
            velocities = bar[note_on_mask, 1] * scale_factor
            adjusted[note_on_mask, 1] = torch.clamp(velocities, 0, 1)
        return adjusted

    def reverse_sequence(self, bars):
        """Reverse the sequence of bars"""
        return bars[::-1]

    def rotate_sequence(self, bars, shift):
        """Rotate the sequence by N bars"""
        return bars[shift:] + bars[:shift]
    


    ### for continuous diffusion
    def encode_bars(self, bars):
        """
        Encode a list of bars using the VAE
        bars: list of tensors, each of shape [bar_length, 2]
        """
        latent_bars = []
        with torch.no_grad():
            for bar in bars:
                # Move to device and add batch dimension
                bar_tensor = bar.to(self.device).unsqueeze(0)
                # Encode
                latent = self.vae_model.encode(bar_tensor)
                # Move back to CPU to save memory
                latent_bars.append(latent.cpu())
        return latent_bars

    ### for discrete diffusion
    def encode_bars_indices(self, bars):
        ### return indices of the latent space
        latent_bar_indices = []
        with torch.no_grad():
            for bar in bars:
                # Move to device and add batch dimension
                bar_tensor = bar.to(self.device).unsqueeze(0)
                # Encode
                # latent = self.vae_model.encode(bar_tensor)
                z, vq_loss, indices = self.vae_model(bar_tensor)
                # Move back to CPU to save memory
                latent_bar_indices.append(indices.cpu())
        return latent_bar_indices
    

    def create_sequences(self, latent_bars):
        """
        Create sequences of consecutive bars from list of latent vectors
        """
        sequences = []
        for i in range(len(latent_bars) - self.bars_per_sequence):
            input_seq = latent_bars[i:i + self.bars_per_sequence]
            target_seq = latent_bars[i + 1:i + self.bars_per_sequence + 1]
            ### continuous diffusion
            sequences.append( 
                (torch.stack(input_seq), torch.stack(target_seq))
                )

            ## discrete diffusion
            # sequences.append( ( input_seq, target_seq ) )
        return sequences


    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return input_seq, target_seq
    

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        return self.dropout(self.norm(x + attn_out))

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context):
        attn_out, _ = self.mha(x, context, context)
        return self.dropout(self.norm(x + attn_out))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads)
        self.cross_attn = CrossAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)

    def forward(self, x, context=None):
        x = self.self_attn(x)
        if context is not None:
            x = self.cross_attn(x, context)
        x = self.ff(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class DiffusionSequenceTransformer(nn.Module):
    def __init__(self, latent_dim, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.num_layers = num_layers
        self.nhead = nhead
        
        # Input embedding
        # self.input_embed = nn.Linear(latent_dim, d_model)
        # Improved input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # self.time_embed = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     nn.SiLU(),
        #     nn.Linear(d_model * 4, d_model)
        # )

        # More sophisticated time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # # Transformer layers
        # self.layers = nn.ModuleList([
        #     TransformerBlock(d_model, nhead) for _ in range(num_layers)
        # ])

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        
        # Output projection
        # self.output_proj = nn.Linear(d_model, latent_dim)

        # Output projection now specifically for noise prediction
        # self.noise_proj = nn.Sequential(
        #     nn.Linear(d_model, d_model * 2),
        #     nn.GELU(),
        #     nn.Linear(d_model * 2, latent_dim)  # Output dimension matches latent_dim as we predict noise of same shape
        # )

        # Deeper noise projection
        self.noise_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim)
        )


    def forward(self, src, tgt, timesteps):
        # src, tgt shape: [batch_size, seq_len, latent_dim]
        batch_size, seq_len, _ = src.shape
        
        # Embeddings
        src_emb = self.input_embed(src)
        tgt_emb = self.input_embed(tgt)
        
        # Time embeddings
        time_emb = self.get_timestep_embedding(timesteps, self.d_model)
        time_emb = self.time_embed(time_emb)
        
        # Add positional encoding and time embeddings
        src_emb = self.pos_encoder(src_emb) + time_emb.unsqueeze(1)
        tgt_emb = self.pos_encoder(tgt_emb) + time_emb.unsqueeze(1)
        
        # Create masks for decoder
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Encode source sequence
        memory = self.encoder(src_emb)
        
        # Decode target sequence
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        # Project to noise prediction
        predicted_noise = self.noise_proj(output)
        
        # Project back to latent space
        return predicted_noise  # self.output_proj(output)
    

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def get_timestep_embedding(self, timesteps, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers
        }, path)
        print(f"Diffusion model saved to {path}")

class SequenceDiffusion:
    def __init__(self, model, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.device = next(model.parameters()).device

        self.n_steps = n_steps
        self.beta = torch.linspace(beta_start, beta_end, n_steps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma = torch.sqrt(self.beta)

    def diffuse_sequence(self, x_0, t):
        # Add noise to entire sequence
        a_bar = self.alpha_bar[t]
        
         # Expand a_bar to match dimensions
        a_bar = a_bar[:, None, None, None]  # Shape: [batch_size, 1, 1, 1]
        
        eps = torch.randn_like(x_0)
        noised = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * eps
        
        return noised, eps

    def diffuse_sequence(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])
        epsilon = torch.randn_like(x)
        
        # Expand dimensions to match the input shape
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1,1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1,1)
        
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * epsilon, epsilon
    

    @torch.no_grad()
    def sample_sequence(self, shape, device, conditioning=None):
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.n_steps)):
            timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x, timesteps)
            
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise)
            x = x + torch.sqrt(beta) * noise
            
        return x

def train_diffusion_sequence(model, diffusion, dataloader, num_epochs, device, save_path=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx,(src_seq, target_seq) in enumerate(dataloader):
            src_seq = src_seq.to(device)
            tgt_seq = target_seq.to(device)

            batch_size = src_seq.shape[0]
            
            optimizer.zero_grad()
            
            # Sample timesteps
            t = torch.randint(0, diffusion.n_steps, (batch_size,), device=device)
            
            # Add noise to sequence
            noised_src, src_noise = diffusion.diffuse_sequence(src_seq, t)
            noised_tgt, tgt_noise = diffusion.diffuse_sequence(tgt_seq, t)
    
            # Teacher forcing: use target sequence shifted right
            tgt_input = noised_tgt # [:, :-1]  # Remove last token
            tgt_output = tgt_seq # [:, 1:]     # Remove first token


            # Predict
            predicted_noise = model(noised_src.squeeze(2), tgt_input.squeeze(2), t) ### noise predicted


            # # Predict noise
            # predicted_noise = model(noised_seq, t)

            tgt_noise = tgt_noise.squeeze(2)
            tgt_output = tgt_output.squeeze(2)

            noised_tgt = noised_tgt.squeeze(2)

            
            # Calculate diffusion losses
            diff_loss = F.mse_loss(predicted_noise, tgt_noise) # [:, 1:])  # Diffusion loss
            
            
            # Get denoised prediction using predicted noise
            alpha = diffusion.alpha[t]
            alpha_bar = diffusion.alpha_bar[t]
            # Denoise using predicted noise
            denoised = (1 / torch.sqrt(alpha))[:, None, None] * (
                noised_tgt - 
                (1 - alpha)[:, None, None] / torch.sqrt(1 - alpha_bar)[:, None, None] * predicted_noise
            )
            
            recon_loss = F.mse_loss(denoised, tgt_output)        # Reconstruction loss
            
            # L1 regularization
            reg_loss = sum(p.abs().sum() for p in model.parameters())

            loss = diff_loss + recon_loss + 1e-5 * reg_loss


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # #### adjust noise dimension to match predicted_noise
            # batch_size_num_seq, _, latent_dim = predicted_noise.shape
            # noise = noise.view(batch_size_num_seq, _, latent_dim)
            
            # # Calculate loss
            # loss = F.mse_loss(predicted_noise, noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            model.save_model(save_path)
            print(f"Saved best model with loss: {best_loss:.4f}")


##################################################
#### Discrete diffusion 

class DiscreteSequenceDiffusion:
    def __init__(self, model, num_timesteps=1000, num_classes=512):
        self.model = model
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes  # Should match VQ-VAE's num_embeddings
        
        # Create transition matrices for each timestep
        # Q matrices control how likely each token is to change to another token
        alphas = torch.linspace(0.0, 1.0, num_timesteps + 1)
        self.alphas_bar = torch.cumprod(1 - alphas, dim=0)
        
        # Transition probabilities
        self.transition_probs = []
        for t in range(num_timesteps):
            Q = torch.full((num_classes, num_classes), 1/num_classes)
            Q = Q * (1 - self.alphas_bar[t]) + torch.eye(num_classes) * self.alphas_bar[t]
            self.transition_probs.append(Q)

    def q_sample(self, x_0, t):
        """
        Sample from q(x_t | x_0)
        x_0: [batch_size, num_sequences, seq_len] - contains indices into codebook
        """
        batch_size, num_seq, seq_len = x_0.shape
        
        # Get transition matrix for timestep t
        Q = self.transition_probs[t[0]]  # Assuming same t for batch
        Q = Q.to(x_0.device)
        
        # Create probability distribution for each token
        x_0_flat = x_0.reshape(-1)
        probs = Q[x_0_flat]  # Get transition probabilities for each token
        
        # Sample new tokens
        x_t_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        x_t = x_t_flat.reshape(batch_size, num_seq, seq_len)
        
        return x_t

    def sample(self, shape, device):
        """
        Generate new sequence by sampling from p(x_{t-1} | x_t)
        """
        # Start with uniform random tokens
        x = torch.randint(0, self.num_classes, shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            time_tensor = torch.full((shape[0],), t, device=device)
            
            # Get model predictions
            logits = self.model(x, time_tensor)
            
            # Sample new tokens
            probs = F.softmax(logits, dim=-1)
            x = torch.multinomial(probs.view(-1, self.num_classes), 1)
            x = x.view(shape)
        
        return x

class DiscreteTransformer(nn.Module):
    def __init__(self, num_embeddings, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(num_embeddings, d_model)
        
        # Time embedding
        self.time_embedding = nn.Embedding(1000, d_model)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, num_embeddings)

    def forward(self, x, t):
        # x contains indices: [batch_size, num_sequences, seq_len]
        batch_size, num_seq, seq_len = x.shape
        
        # Get embeddings
        token_emb = self.token_embedding(x)
        time_emb = self.time_embedding(t).unsqueeze(1).expand(-1, num_seq * seq_len, -1)
        pos_emb = self.pos_embedding[:, :(num_seq * seq_len)]
        
        # Combine embeddings
        h = token_emb.reshape(batch_size, num_seq * seq_len, -1)
        h = h + time_emb + pos_emb
        
        # Process through transformer
        for layer in self.layers:
            h = layer(h)
        
        # Project to logits
        logits = self.output_proj(h)
        logits = logits.reshape(batch_size, num_seq, seq_len, -1)
        
        return logits
 
 
def train_discrete_diffusion(model, diffusion, dataloader, num_epochs, device, save_path=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x_0, _) in enumerate(dataloader):
            x_0 = x_0.to(device)  # These should be indices from VQ-VAE
            batch_size = x_0.shape[0]
            
            optimizer.zero_grad()
            
            # Sample timestep
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
            
            # Get noisy sample
            x_t = diffusion.q_sample(x_0, t)
            
            # Predict distribution
            logits = model(x_t, t)
            
            # Calculate loss
            loss = criterion(logits.reshape(-1, diffusion.num_classes), x_0.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            model.save_model(save_path)
            print(f"Saved best model with loss: {best_loss:.4f}")


# ############################
def generate_music(vae_model, noise_predictor, diffusion, num_bars=32, sequence_length=8, min_note=60, device='cuda'):
    vae_model.eval()
    noise_predictor.eval()
    
    with torch.no_grad():
        current_sequence = torch.randn(1, sequence_length, vae_model.latent_dim).to(device)
        generated_bars = []
        
        for i in range(num_bars):
            x = current_sequence
            
            for t in reversed(range(diffusion.n_steps)):
                timesteps = torch.full((1,), t, device=device, dtype=torch.long)
                predicted_noise = noise_predictor(x, x, timesteps)
                
                # Use proper diffusion attributes
                alpha = 1 - diffusion.beta[t]
                alpha_bar = diffusion.alpha_bar[t]
                beta = diffusion.beta[t]
                
                noise = torch.randn_like(x) if t > 0 else 0
                x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise)
                x = x + torch.sqrt(beta) * noise
            
            first_bar_latent = x[0, 0].unsqueeze(0)
            decoded_bar = vae_model.decode(first_bar_latent)
           
            # Filter notes below threshold
            note_events_mask = decoded_bar[:, 0] == 0  # note-on events
            decoded_bar[note_events_mask, 1] = torch.clamp(decoded_bar[note_events_mask, 1] * 127, min=min_note) / 127
            

            generated_bars.append(decoded_bar)
            
            current_sequence = torch.cat([
                x[:, 1:],
                torch.randn(1, 1, vae_model.latent_dim).to(device)
            ], dim=1)
    
    full_sequence = torch.cat(generated_bars, dim=0)
    return full_sequence



def scale_to_midi_range(generated_sequence):
    event_types = np.round(generated_sequence[:, 0]).astype(int)
    values = generated_sequence[:, 1]
    scaled_values = np.zeros_like(values)
    
    for i, event_type in enumerate(event_types):
        if event_type == 0 or event_type == 1:  # note on or note off
            scaled_values[i] = np.clip(np.round(values[i] * PITCH_RANGE), 0, 127).astype(int)
        elif event_type == 2:  # time shift
            scaled_values[i] = values[i] * MAX_TIME
    
    return np.stack([event_types, scaled_values], axis=-1)


def save_generated_midi(sequence, output_file, ticks_per_beat=480):
    """
    Convert generated sequence to MIDI file
    Args:
        sequence: tensor of shape [num_bars * bar_length, 2]
        output_file: path to save MIDI file
        ticks_per_beat: MIDI time resolution
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Assuming 4/4 time signature
    ticks_per_bar = ticks_per_beat * 4

    current_time = 0
    active_notes = set()

    sequence = sequence.cpu().detach().numpy()
    
    # Process each bar
    for bar_idx in range(sequence.shape[0]):
        bar = sequence[bar_idx]
        bar_start_time = bar_idx * ticks_per_bar
        last_event_time = 0

        bar = scale_to_midi_range(bar) ### scale it to midi range
        
        # Process events within the bar
        for event_idx in range(bar.shape[0]):

            event = bar[event_idx]
            event_type = int(round(event[0].item()))
            # note = int(round(event[1].item() * 127))
            note = int(round( event[1].item() ) )
            
            # Calculate delta time since last event
            delta_time = event_idx - last_event_time
            
            if event_type == 0:  # note_on
                if note not in active_notes:
                    track.append(Message('note_on', note=note, velocity=64, time=delta_time))
                    active_notes.add(note)
                    last_event_time = event_idx
            elif event_type == 1:  # note_off
                if note in active_notes:
                    track.append(Message('note_off', note=note, velocity=0, time=delta_time))
                    active_notes.remove(note)
                    last_event_time = event_idx
        
        # If we're not at the last bar, add any remaining time to reach the bar boundary
        if bar_idx < sequence.shape[0] - 1:
            remaining_time = ticks_per_bar - last_event_time
            if remaining_time > 0:
                track.append(Message('note_on', note=0, velocity=0, time=remaining_time))

    
    # Turn off any remaining notes
    for note in active_notes:
        track.append(Message('note_off', note=note, velocity=0, time=current_time))
    
    mid.save(output_file)

# Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    midi_folder = '../midi_dataset/piano_maestro/piano_maestro-v1.0.0/2004/'
    
    vqvae_model_path = './models/vqvae_model_melody.pth'

    diffusion_model_path='./models/latent_diffusion_model.pth'

    mode = 'generate' # 'train'
    
    # Load pretrained VAE
    vae_model , checkpoint= BarVQVAE.load_model(vqvae_model_path, device)
    # vae_model.eval()
    
    
    # # Create diffusion sequence model
    model = DiffusionSequenceTransformer(
        latent_dim=vae_model.latent_dim,
        d_model=512,
        nhead=8,
        num_layers=6
    ).to(device)
    
    diffusion = SequenceDiffusion(model)
        
    if mode == 'train':
        # Create dataset with latent sequences
        dataset = BarSequenceDataset(
            midi_folder=midi_folder,
            vae_model=vae_model,
            bars_per_sequence=8,
            bar_length=512,
            device=device
        )

        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Train diffusion model
        train_diffusion_sequence(model, diffusion, dataloader, num_epochs=100, device=device, save_path=diffusion_model_path)

        # ## train discrete diffusion
        # # Create discrete diffusion model
        # discrete_model = DiscreteTransformer(
        #     num_embeddings=vae_model.num_embeddings,
        #     d_model=512,
        #     nhead=8,
        #     num_layers=6
        # ).to(device)

        # discrete_diffusion = DiscreteSequenceDiffusion(discrete_model)

        # train_discrete_diffusion(discrete_model, discrete_diffusion, dataloader, num_epochs=10, device=device,  save_path='./models/diffusion_model_discrete.pth')

    elif mode == 'generate':
        ### generate 
        # Generate music
        generated_sequence = generate_music(
        vae_model=vae_model,
        noise_predictor=model,
        diffusion=diffusion,
        num_bars=32,
        sequence_length=8
        )

        # Save as MIDI
        save_generated_midi(generated_sequence, 'generated_music.mid')
    