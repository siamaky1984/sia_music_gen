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

MAX_TIME = 2.0  # Maximum time between events in seconds
MAX_VELOCITY = 127
PITCH_RANGE = 128
EVENT_TYPES = 3  # note_on, note_off, time_shift

class MIDIDataset(Dataset):
    def __init__(self, midi_folder, seq_len=64, context_length=4):
        self.midi_folder = midi_folder
        self.seq_len = seq_len

        self.context_len = context_length
        self.total_len = seq_len * (context_length + 1)

        self.data = self.load_all_midi_files()



    def load_all_midi_files(self):
        all_sequences = []
        for filename in os.listdir(self.midi_folder):
            if filename.endswith('.mid') or filename.endswith('.midi'):
                file_path = os.path.join(self.midi_folder, filename)
                sequences = self.midi_to_sequence(file_path)
                all_sequences.extend(sequences)
        return all_sequences

    def midi_to_sequence(self, midi_file):
        mid = MidiFile(midi_file)
        events = []
        current_time = 0

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off':
                    if msg.time > 0:
                        events.append(self.encode_time(msg.time))
                    events.append(self.encode_note_event(msg))
                current_time += msg.time

        sequences = []
        for i in range(0, len(events), self.seq_len):
            seq = events[i:i+self.seq_len]
            if len(seq) == self.seq_len:
                sequences.append(torch.tensor(seq, dtype=torch.float32))

        # for i in range(0, len(events) - self.total_len + 1):
        #     full_sequence = events[i:i + self.total_len]
        #     context = full_sequence[:-self.seq_len]
        #     target = full_sequence[-self.seq_len:]
            
        #     sequences.append({
        #         'context': context,
        #         'target': target
        #     })

        return sequences

    def encode_time(self, delta_time):
        time_shift = min(delta_time, MAX_TIME)
        return [2, time_shift / MAX_TIME]  # Use 2 for time_shift event type

    def encode_note_event(self, msg):
        event_type = 0 if msg.type == 'note_on' and msg.velocity > 0 else 1
        return [event_type, msg.note / PITCH_RANGE]  # Normalize pitch to [0, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # return {
        #     'context': torch.tensor(self.data[idx]['context'], dtype=torch.float32),
        #     'target': torch.tensor(self.data[idx]['target'], dtype=torch.float32)
        # }
        return self.data[idx]


class AugmentedMIDIDataset(Dataset):
    def __init__(self, midi_folder, sequence_length=32, fs=16,
                 augment_probability=0.5, cache_size=1000):
        """
        Args:
            midi_folder: Path to MIDI files
            sequence_length: Length of sequences to generate
            fs: Sampling rate in time steps per bar
            augment_probability: Probability of applying each augmentation
            cache_size: Maximum number of MIDI files to cache in memory
        """
        # self.midi_paths = glob.glob(os.path.join(midi_folder, "**/*.mid"), recursive=True)
        self.midi_paths = glob.glob(os.path.join(midi_folder, "*.mid"), recursive= True) + \
                    glob.glob(os.path.join(midi_folder, "*.midi"), recursive=True)


        self.sequence_length = sequence_length
        self.fs = fs
        self.augment_probability = augment_probability

        # LRU Cache for MIDI files
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []

        # Augmentation parameters
        self.transpose_range = (-6, 6)        # Semitones
        self.tempo_range = (0.8, 1.2)         # Tempo multiplier
        self.velocity_range = (0.7, 1.3)      # Velocity multiplier
        self.time_stretch_range = (0.9, 1.1)  # Time stretch factor
        self.pitch_shift_prob = 0.3           # Probability of shifting individual notes
        self.note_density_range = (0.8, 1.2)  # Note density multiplier

    def _cache_midi(self, midi_path):
        """Cache MIDI file with LRU implementation"""
        if midi_path in self.cache:
            # Move to end of cache order (most recently used)
            self.cache_order.remove(midi_path)
            self.cache_order.append(midi_path)
        else:
            # Add new file to cache
            if len(self.cache) >= self.cache_size:
                # Remove least recently used
                lru_path = self.cache_order.pop(0)
                del self.cache[lru_path]

            midi_data = pretty_midi.PrettyMIDI(midi_path)
            self.cache[midi_path] = midi_data
            self.cache_order.append(midi_path)

        return self.cache[midi_path]

    def _augment_transpose(self, midi_data):
        """Transpose entire piece"""
        try:
            midi_data = copy.deepcopy(midi_data)
            semitones = np.random.randint(*self.transpose_range)

            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    note.pitch = np.clip(note.pitch + semitones, 0, 127)
            # print('Transpose complete')
            return midi_data

        except Exception as e:
            print(f"Error in transpose: {str(e)}")
            return midi_data  # Return original if augmentation fails

        

    def _augment_tempo(self, midi_data):
        """Modify tempo by scaling note timings"""
        try:
            midi_data = copy.deepcopy(midi_data)
            tempo_multiplier = np.random.uniform(*self.tempo_range)
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    note.start /= tempo_multiplier
                    note.end /= tempo_multiplier
            return midi_data
        
        except Exception as e:
            print(f"Error in transpose: {str(e)}")
            return midi_data

    def _augment_velocity(self, midi_data):
        """Change note velocities"""
        midi_data = copy.deepcopy(midi_data)
        velocity_multiplier = np.random.uniform(*self.velocity_range)

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note.velocity = int(np.clip(note.velocity * velocity_multiplier, 1, 127))

        return midi_data

    def _augment_time_stretch_local(self, midi_data):
        """Apply different time stretching to different sections"""
        midi_data = copy.deepcopy(midi_data)

        # Divide piece into sections
        section_length = 4.0  # 4 beats per section
        max_time = max(note.end for instr in midi_data.instruments
                      for note in instr.notes)
        num_sections = int(max_time / section_length) + 1

        # Generate stretch factors for each section
        stretch_factors = [np.random.uniform(*self.time_stretch_range)
                         for _ in range(num_sections)]

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                section = int(note.start / section_length)
                stretch = stretch_factors[section]
                note.start *= stretch
                note.end *= stretch

        return midi_data

    def _augment_note_density(self, midi_data):
        """Randomly remove or duplicate notes"""
        midi_data = copy.deepcopy(midi_data)
        density_factor = np.random.uniform(*self.note_density_range)

        for instrument in midi_data.instruments:
            notes = instrument.notes.copy()
            if density_factor < 1.0:
                # Remove notes
                keep_prob = max(0.5, density_factor)
                instrument.notes = [note for note in notes
                                  if np.random.random() < keep_prob]
            else:
                # Add notes (duplicates with slight modifications)
                num_new_notes = int((density_factor - 1.0) * len(notes))
                for _ in range(num_new_notes):
                    if notes:  # If there are notes to duplicate
                        orig_note = np.random.choice(notes)
                        new_note = copy.deepcopy(orig_note)
                        # Slightly modify the new note
                        new_note.start += np.random.uniform(-0.1, 0.1)
                        new_note.end += np.random.uniform(-0.1, 0.1)
                        new_note.pitch += np.random.randint(-2, 3)
                        instrument.notes.append(new_note)

        return midi_data

    def _augment_articulation(self, midi_data):
        """Modify note lengths (articulation)"""
        midi_data = copy.deepcopy(midi_data)

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note_length = note.end - note.start
                # Randomly shorten or lengthen notes
                length_factor = np.random.uniform(0.8, 1.2)
                new_length = note_length * length_factor
                note.end = note.start + new_length

        return midi_data

    def _get_piano_roll(self, midi_data):
        """Convert MIDI to piano roll format"""
        fs = int(1 / self.sequence_length * 100)
        piano_roll = midi_data.get_piano_roll(fs=fs)
        piano_roll = (piano_roll > 0).astype(np.float32)
        return piano_roll

    def __len__(self):
        return len(self.midi_paths)

    def __getitem__(self, idx):
        """Get a randomly augmented MIDI sequence"""
        midi_path = self.midi_paths[idx]
        midi_data = self._cache_midi(midi_path)

        # Apply random augmentations
        augmentations = [
            (self._augment_transpose, self.augment_probability),
            (self._augment_tempo, self.augment_probability),
            (self._augment_velocity, self.augment_probability),
            (self._augment_time_stretch_local, self.augment_probability * 0.7),
            (self._augment_note_density, self.augment_probability * 0.5),
            (self._augment_articulation, self.augment_probability * 0.7)
        ]

        for aug_func, prob in augmentations:
            if np.random.random() < prob:
                try:
                    midi_data = aug_func(midi_data)
                except Exception as e:
                    print(f"Augmentation {aug_func.__name__} failed: {str(e)}")
                    continue

        # Convert to piano roll
        piano_roll = self._get_piano_roll(midi_data)

        # Handle sequence length
        if piano_roll.shape[1] < self.sequence_length:
            padding = np.zeros((piano_roll.shape[0],
                              self.sequence_length - piano_roll.shape[1]))
            piano_roll = np.concatenate([piano_roll, padding], axis=1)

        # Take random sequence if longer than sequence_length
        if piano_roll.shape[1] > self.sequence_length:
            start = np.random.randint(0, piano_roll.shape[1] - self.sequence_length)
            piano_roll = piano_roll[:, start:start + self.sequence_length]

        # Add channel dimension and convert to tensor
        sequence = torch.from_numpy(piano_roll).unsqueeze(0).float()
        return sequence
    

class MIDIEncoder(nn.Module):
    def __init__(self, sequence_length, feature_dim,  hidden_dim=512, latent_dim=64):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.input_dim = sequence_length * feature_dim  # Total flattened input size
        
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.input_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.LayerNorm(hidden_dim // 2),
        #     nn.ReLU(),
        # )
        
        # self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        # self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
                
        # Deeper architecture with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            ResidualBlock(hidden_dim // 2),
        )
        
        # Separate pathways for mu and logvar
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, latent_dim)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, latent_dim)
        )


    def forward(self, x):
        # Flatten the input: [batch_size, sequence_length, feature_dim] -> [batch_size, sequence_length * feature_dim]
        x = x.view(x.size(0), -1)
        
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, log_var

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + self.block(x)
    

class MIDIDecoder(nn.Module):
    def __init__(self, sequence_length, feature_dim, latent_dim, hidden_dim=512):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.output_dim = sequence_length * feature_dim
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim // 2),
        #     nn.LayerNorm(hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, self.output_dim),
        # )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            ResidualBlock(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, self.output_dim),
        )


    def forward(self, z):
        x = self.decoder(z)
        # Reshape back to sequence form: [batch_size, sequence_length * feature_dim] -> [batch_size, sequence_length, feature_dim]
        x = x.view(-1, self.sequence_length, self.feature_dim)
        return x

class MIDIVAE(nn.Module):
    def __init__(self, sequence_length, feature_dim, hidden_dim=512, latent_dim=64):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = MIDIEncoder(sequence_length, feature_dim, hidden_dim, latent_dim)
        self.decoder = MIDIDecoder(sequence_length, feature_dim, latent_dim, hidden_dim)
        
    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim
        }, path)
        print(f"VAE model saved to {path}")

    @classmethod
    def load_model(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            sequence_length=checkpoint['sequence_length'],
            feature_dim=checkpoint['feature_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            latent_dim=checkpoint['latent_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model
    

def train_vae(vae, dataloader, num_epochs=100, device="cuda", save_path=None):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    best_loss = float('inf')

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
    )
    
    kl_start = 0
    kl_end = 1
    kl_cycles = 3  # Number of cycles for cyclical annealing
    kl_ratio = 0.5  # Portion of epochs for annealing
    

    for epoch in range(num_epochs):
        total_loss = 0
        recon_loss = 0
        kl_loss = 0


        # Calculate KL weight for this epoch
        if epoch < kl_ratio * num_epochs:
            # Cyclical annealing
            cycle = np.floor(epoch / (kl_ratio * num_epochs / kl_cycles))
            cycle_progress = (epoch % (kl_ratio * num_epochs / kl_cycles)) / (kl_ratio * num_epochs / kl_cycles)
            kl_weight = kl_start + (kl_end - kl_start) * (cycle_progress if cycle % 2 == 0 else 1 - cycle_progress)
        else:
            kl_weight = kl_end

        
        for batch in dataloader:
            # print(batch)
            
            batch = batch.to(device)

            optimizer.zero_grad()

            
            recon_batch, mu, log_var = vae(batch)

            # recon = F.mse_loss(recon_batch, batch)
            # kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # loss = recon + kl

            mask = (batch != 0).float()
            recon = F.mse_loss(recon_batch * mask, batch * mask, reduction='sum') / batch.sum()
               # KL divergence loss
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch.size(0)
            
            # Total loss with weighting
            loss = recon + kl_weight * kl

            
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

            optimizer.step()
            
            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss += kl.item()
        

        avg_loss = total_loss / len(dataloader)
        avg_recon = recon_loss / len(dataloader)
        avg_kl = kl_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} (Recon = {avg_recon:.4f}, KL = {avg_kl:.4f})")
        
        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            vae.save_model(save_path)
            print(f"Saved best model with loss: {best_loss:.4f}")

        scheduler.step()
        


####################### VQVAE###############################


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = self.norm1(x)
        att_output, _ = self.attention(x, x, x)
        x = x + att_output
        x = x + self.mlp(self.norm2(x))
        return x
    

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # inputs shape: [batch_size, sequence_length, embedding_dim]
        batch_size, sequence_length, embedding_dim = inputs.shape

        # # Convert inputs to flat form
        # flat_inputs = inputs.reshape(-1, self.embedding_dim)
        # Reshape inputs while keeping batch dimension
        flat_inputs = inputs.reshape(batch_size * sequence_length, embedding_dim)
        
        
        # Calculate distances
        distances = torch.cdist(flat_inputs, self.embedding.weight, p=2.0)
        
        # Get nearest codebook vectors
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = self.embedding(encoding_indices)
        quantized = quantized.reshape(inputs.shape)
        
        # Compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices



class HierarchicalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Local processing
        self.local_attention = AttentionBlock(hidden_dim, num_heads)
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        
        # Global processing
        self.global_attention = AttentionBlock(hidden_dim, num_heads)
        
        # Downsampling layers
        self.downsample = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        x = self.input_proj(x)
        
        # Local processing
        local_features = self.local_attention(x)
        local_features = rearrange(local_features, 'b n d -> b d n')
        local_features = self.local_conv(local_features)
        local_features = rearrange(local_features, 'b d n -> b n d')
        
        # Global processing
        global_features = self.global_attention(local_features)
        
        # Downsampling
        features = rearrange(global_features, 'b n d -> b d n')
        features = self.downsample(features)
        features = rearrange(features, 'b d n -> b n d')
        
        return features



class HierarchicalDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        )
        
        # Attention blocks
        self.attention1 = AttentionBlock(hidden_dim, num_heads)
        self.attention2 = AttentionBlock(hidden_dim, num_heads)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Upsampling
        x = rearrange(x, 'b n d -> b d n')
        x = self.upsample(x)
        x = rearrange(x, 'b d n -> b n d')
        
        # Apply attention
        x = self.attention1(x)
        x = self.attention2(x)
        
        # Project to output space
        x = self.output_proj(x)
        return x



class HierarchicalVQVAE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim=512, 
        num_embeddings=512, 
        commitment_cost=0.25,
        num_heads=8
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # This is effectively our latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.num_heads = num_heads
        
        # Make latent_dim accessible as a property
        self.latent_dim = hidden_dim


        self.encoder = HierarchicalEncoder(input_dim, hidden_dim, num_heads)
        self.vq = VectorQuantizer(num_embeddings, hidden_dim, commitment_cost)
        self.decoder = HierarchicalDecoder(input_dim, hidden_dim, num_heads)

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Vector quantization
        z_q, vq_loss, perplexity, indices = self.vq(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, perplexity, indices
    
    def encode(self, x):
        z = self.encoder(x)
        z_q, _, _, indices = self.vq(z)
        return z_q #, indices
    
    def decode(self, indices):
        z_q = self.vq.embedding(indices)
        x_recon = self.decoder(z_q)
        return x_recon
    

    # Save the trained model
    def save_model(self, path, optimizer=None, epoch=None):
        """
        Save model state along with configuration and training state
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_embeddings': self.num_embeddings,
                'commitment_cost': self.commitment_cost,
                'num_heads': self.num_heads
            }
        }
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch
            
        torch.save(save_dict, path)
        print(f"Model saved to {path}")


    @classmethod
    def load_model(cls, path, device='cuda', strict=True):
        """
        Load model state and configuration from a saved file
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model instance with saved config
        model = cls(
            input_dim=checkpoint['model_config']['input_dim'],
            hidden_dim=checkpoint['model_config']['hidden_dim'],
            num_embeddings=checkpoint['model_config']['num_embeddings'],
            # commitment_cost=checkpoint['model_config']['commitment_cost'],
            num_heads=checkpoint['model_config']['num_heads']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model = model.to(device)
        model.eval()
        
        return model, checkpoint


def train_vqvae(model, dataloader, num_epochs=100, device="cuda", save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_loss = float('inf')

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
    )
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_perplexity = 0
        
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            
            x = batch.to(device)
            x_recon, vq_loss, perplexity, _ = model(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)
            
            # Total loss
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_vq_loss = total_vq_loss / len(dataloader)
        avg_perplexity = total_perplexity / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  VQ Loss: {avg_vq_loss:.4f}")
        print(f"  Codebook Perplexity: {avg_perplexity:.2f}")


        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            model.save_model(save_path)
            print(f"Saved best model with loss: {best_loss:.4f}")

############################################################



class DiffusionModel:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, n_steps=1000):
        self.model = model
        self.device = next(model.parameters()).device
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])
        epsilon = torch.randn_like(x)
        
        # Expand dimensions to match the input shape
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1)
        
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * epsilon, epsilon
    
        # return sqrt_alpha_bar.unsqueeze(-1).unsqueeze(-1) * x + sqrt_one_minus_alpha_bar.unsqueeze(-1).unsqueeze(-1) * epsilon, epsilon

    @torch.no_grad()
    def sample(self, shape , context=None):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(len(self.betas))):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)
            predicted_noise = self.model(x, t_tensor, context)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise
        
        return x




class LatentDiffusionTransformer(nn.Module):
    def __init__(self, latent_dim=64, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.latent_projection = nn.Linear(latent_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, latent_dim)

    def forward(self, x, t):
        x = self.latent_projection(x)
        t_emb = self.get_timestep_embedding(t, self.d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = x + t_emb
        x = self.transformer(x)
        return self.output_projection(x)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers
        }, path)
        print(f"Diffusion model saved to {path}")

    @classmethod
    def load_model(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            latent_dim=checkpoint['latent_dim'],
            d_model=checkpoint['d_model'],
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model




def train_latent_diffusion(transformer, vae, dataloader, num_epochs=100, device="cuda", save_path=None):
    diffusion = DiffusionModel(transformer)  # Create diffusion model wrapper
    
    optimizer = torch.optim.Adam(transformer.parameters())
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.to(device)
            
            # Encode batch to latent space using pre-trained VAE
            with torch.no_grad():
                latent_batch = vae.encode(batch)
            
            # Add noise and predict
            t = torch.randint(0, 1000, (batch.size(0),), device=device)
            noisy_latents, noise = diffusion.add_noise(latent_batch, t)
            # predicted_noise = diffusion(noisy_latents, t)
            predicted_noise = transformer(noisy_latents, t)
            
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            transformer.save_model(save_path)
            print(f"Saved best model with loss: {best_loss:.4f}")


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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

def sequence_to_midi(sequence, output_file):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    current_time = 0
    last_event_time = 0

    # for event in sequence:
    #     event_type, note, time = event
        
    #     # Ensure time delta is non-negative
    #     time_delta = max(0, int(time * mid.ticks_per_beat - last_event_time))
    #     last_event_time = int(time * mid.ticks_per_beat)
        
    #     msg_type = 'note_on' if event_type == 0 else 'note_off'
    #     velocity = 64 if msg_type == 'note_on' else 0
        
    #     # Create MIDI message with non-negative time
    #     track.append(Message(msg_type, note=int(note), velocity=velocity, time=time_delta))
    

    for event in sequence:
        event_type, value = event
        if event_type == 2:  # Time shift
            if value<0:
                value =0
            current_time += value
        else:
           
            note = int(value)

            msg_type = 'note_on' if event_type == 0 else 'note_off'
            velocity = 64 if msg_type == 'note_on' else 0
            track.append(Message(msg_type, note=note, velocity=velocity, time=int(current_time * mid.ticks_per_beat)))
            current_time = 0
    
    mid.save(output_file)


def generate_midi(model_path, output_file, seq_len=64, num_sequences=8, context_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    # Extract the position embedding size from the checkpoint
    pos_embed_size = checkpoint['model_state_dict']['position_embedding.weight'].size(0)
    
    model = DiffusionTransformer(max_seq_len=pos_embed_size + context_size).to(device)
    diffusion = DiffusionModel(model)
    

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully.")
    
    full_sequence = []
    context = None
    with torch.no_grad():
        for i in range(num_sequences):
            generated = diffusion.sample((1, seq_len, 2))
            generated_sequence = generated[0].cpu().numpy()
            scaled_sequence = scale_to_midi_range(generated_sequence)
            full_sequence.append(scaled_sequence)
            
            # Update context for next iteration
            if i < num_sequences - 1:  # No need to update context for the last iteration
                context = torch.from_numpy(scaled_sequence[-context_size:]).float().unsqueeze(0).to(device)
    
    full_sequence = np.concatenate(full_sequence, axis=0)
    
    # Post-process the full sequence
    processed_sequence = post_process_sequence(full_sequence)
    
    sequence_to_midi(processed_sequence, output_file)
    print(f"Generated MIDI saved to {output_file}")


def post_process_sequence(sequence):
    processed = []
    current_notes = set()
    
    for event in sequence:
        event_type, value = event
        if event_type == 0:  # note on
            if value not in current_notes:
                processed.append(event)
                current_notes.add(value)
        elif event_type == 1:  # note off
            if value in current_notes:
                processed.append(event)
                current_notes.remove(value)
        else:  # time shift
            processed.append(event)
    
    # Ensure all notes are turned off at the end
    for note in current_notes:
        processed.append([1, note])
    
    return np.array(processed)


# And update the main usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Usage example
    midi_folder = '../midi_dataset/piano_maestro/piano_maestro-v1.0.0/2004/'
    sequence_len = 32
    feature_dim = 2
    batch_size = 32

    num_epochs = 2

    dataset = MIDIDataset(midi_folder, seq_len = sequence_len)
    # dataset = AugmentedMIDIDataset(midi_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mode = 'train_diffusion_trans'  # 'train_vqvae' # # 'train_vqvae' # 'train_diffusion_trans' # 'train_vqvae' # 'train_vae'
    ### 'train_diffusion_trans', 'inference'
    
    # Training mode
    if mode == 'train_vae':  # Change to False for inference mode
        # Initialize models
        input_dim = batch_size * sequence_len  # Set your input dimension
            # Initialize VAE with correct dimensions
        vae = MIDIVAE(
            sequence_length=sequence_len,
            feature_dim=feature_dim,
            hidden_dim=512,
            latent_dim=64
        ).to(device)
            
        vae.apply(weights_init)

        # Train and save VAE
        train_vae(vae, dataloader, num_epochs=num_epochs, device=device, save_path='vqvae_model.pth')

    elif mode == 'train_vqvae':
        # Assuming your MIDI data has shape [batch_size, sequence_length, feature_dim]
        model = HierarchicalVQVAE(
            input_dim=2,        # Your MIDI feature dimension
            hidden_dim=512,     # Size of hidden representations
            num_embeddings=512, # Size of codebook
            num_heads=8
        ).to(device)

         # Train the model
        train_vqvae(model, dataloader, num_epochs=num_epochs, device=device)
        
        # # Save the trained model
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'model_config': {
        #         'input_dim': 2,
        #         'hidden_dim': 512,
        #         'num_embeddings': 512,
        #         'num_heads': 8
        #     }
        # }, 'hierarchical_vqvae.pt')
        

    elif mode =='train_diffusion_trans':
        # vae = MIDIVAE.load_model('vae_model.pth', device)
        vae , checkpoint = HierarchicalVQVAE.load_model('hierarchical_vqvae.pt', device)  ### VQVAE
        # Initialize and train transformer using trained VAE
        transformer = LatentDiffusionTransformer(latent_dim=vae.latent_dim).to(device)
        train_latent_diffusion(transformer, vae, dataloader, num_epochs=num_epochs, 
                             device=device, save_path='diffusion_model.pth')
    
    # Inference mode
    else:
        # Load trained models
        vae = MIDIVAE.load_model('hierarchical_vqvae.pt', device)
        transformer = LatentDiffusionTransformer.load_model('diffusion_model.pth', device)
        diffusion = DiffusionModel(transformer)
        
        # Generate new music
        with torch.no_grad():
            # # Start with random noise or encode existing MIDI
            # if start_from_existing:
            #     latent_input = vae.encode(your_input.to(device))
            # else:
            #     latent_input = torch.randn(1, vae.latent_dim).to(device)
            
            latent_input = torch.randn(1, vae.latent_dim).to(device)
            
            # Generate with diffusion model
            generated_latent = diffusion.sample(latent_input.shape)
            # Decode back to MIDI space
            generated_midi = vae.decode(generated_latent)

# mode = 'inference' #'train'

# if mode == 'train':

#     # Usage example
#     midi_folder = '../midi_dataset/piano_maestro-v1.0.0/2004/'
#     dataset = MIDIDataset(midi_folder)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#     model = DiffusionTransformer().to(device)
#     diffusion = DiffusionModel(model)

#     # Training loop
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     num_epochs = 2

#     # for epoch in range(num_epochs):
#     #     print('epoch', epoch)
#     #     context = None
#     #     for batch in dataloader:
#     #         batch = batch.to(device)
#     #         optimizer.zero_grad()
#     #         t = torch.randint(0, 1000, (batch.size(0),), device=device)
            
#     #         # print(batch.shape, t.shape)
#     #         noisy_batch, noise = diffusion.add_noise(batch, t)
#     #         # print('>>>', noisy_batch.shape, t.shape, noise.shape)

#     #         predicted_noise = model(noisy_batch, t, context)
#     #         # print(predicted_noise.shape, noise.shape)
            
#     #         loss = F.mse_loss(predicted_noise, noise)

#     #         loss.backward()
#     #         optimizer.step()
        
#     #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

#     for epoch in range(num_epochs):
#         print('epoch', epoch)
#         for batch in dataloader:

#             target = batch['target'].to(device)
#             context = batch['context'].to(device)

#             optimizer.zero_grad()

#             t = torch.randint(0, 1000, (target.size(0),), device=device)
#             noise = torch.randn_like(target)

#             noisy_batch, noise = diffusion.add_noise(target, t)
#             predicted_noise = model(noisy_batch, t, context)

#             loss = F.mse_loss(predicted_noise, noise)

#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



#     # Save the model after training
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': num_epochs,
#         'loss': loss.item(),
#     }, 'midi_diffusion_model.pth')

#     print("Model saved successfully.")

# elif mode =='inference':

#     # Example usage of the inference function
#     model_path = './models/diffusion_trans_cont_model/midi_diffusion_model.pth'
#     generate_midi(model_path, 'generated_music.mid', seq_len=512, num_sequences=8, context_size=0)