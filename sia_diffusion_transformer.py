'''

### diffusion transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mido import MidiFile
import os
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, midi_folder, chunk_size=32):
        self.midi_folder = midi_folder
        self.chunk_size = chunk_size
        self.chunker = MIDIChunker(chunk_size)
        self.chunks = self.load_all_midi_files()

    def load_all_midi_files(self):
        all_chunks = []
        for filename in os.listdir(self.midi_folder):
            print('filename', filename)
            if filename.endswith('.mid') or filename.endswith('.midi'):
                file_path = os.path.join(self.midi_folder, filename)
                chunks = self.chunker.chunk_midi(file_path)
                all_chunks.append(chunks)
        return torch.cat(all_chunks, dim=0)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

class MIDIChunker:
    def __init__(self, chunk_size=32):
        self.chunk_size = chunk_size

    def chunk_midi(self, midi_file):
        mid = MidiFile(midi_file)
        chunks = []
        current_chunk = []
        
        for msg in mid:
            if not msg.is_meta:
                current_chunk.append(msg)
                if len(current_chunk) == self.chunk_size:
                    chunks.append(self.encode_chunk(current_chunk))
                    current_chunk = []
        
        if current_chunk:
            chunks.append(self.encode_chunk(current_chunk))
        
        return torch.tensor(chunks)

    def encode_chunk(self, chunk):
        # Simplified encoding: [time, note, velocity]
        for msg in chunk:
            print(msg)
            try:
                print(msg.note)
                if msg.note:
                    return [msg.time, msg.note, msg.velocity] 
            except:
                print('nsf ')
                return []
        # return [[msg.time, msg.note, msg.velocity] for msg in chunk]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x,x,x)
        x = self.norm1(x+attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x+ff_output)
        return x 


class DiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock( d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, 3)

    def forward(self, x, t):
        x = self.embedding(x)
        t = t.unsqueeze(-1).expand(-1, x.size(1), -1)
        x = torch.cat([x, t], dim=-1)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        return self.fc_out(x)


class DiffusionModel:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, n_steps=1000):
        self.model = model 
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1 - self.betas 
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0)

    def add_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt( 1 - self.alpha_bars[t])
        epsilon = torch.randn_like(x)
        print('>>>>', x.shape)
        return sqrt_alpha_bar *x + sqrt_one_minus_alpha_bar * epsilon , epsilon

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape)
        for t in range(len(self.betas)):
            t_tensor = torch.full((shape[0],), t, dtype = torch.long)
            predicted_noise = self.model(x, t_tensor)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zaros_like(x)

            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise
        
        return x


# Usage example
midi_folder = './midi_piano'
dataset = MIDIDataset(midi_folder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


diffTrans = DiffusionTransformer(d_model = 64, nhead=4, num_layers=3, dim_feedforward=256)
diffusion_mod = DiffusionModel(diffTrans)


# Training loop (simplified)
optimizer = torch.optim.Adam(diffTrans.parameters(), lr=1e-4)

num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        t = torch.randint(0, 1000, (batch.size(0),))
        noisy_batch, noise = diffusion_mod.add_noise(batch, t)
        predicted_noise = diffTrans(noisy_batch, t)
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



# Generate new MIDI
generated = diffusion_mod.sample((32, 32, 3))  # Generate 32 chunks of size 32x3


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mido import MidiFile, MidiTrack, Message
import os
from torch.utils.data import Dataset, DataLoader

MAX_TIME = 2.0  # Maximum time between events in seconds
MAX_VELOCITY = 127
PITCH_RANGE = 128
EVENT_TYPES = 3  # note_on, note_off, time_shift

class MIDIDataset(Dataset):
    def __init__(self, midi_folder, seq_len=512):
        self.midi_folder = midi_folder
        self.seq_len = seq_len
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
                sequences.append(torch.tensor(seq))

        return sequences

    def encode_time(self, delta_time):
        # Encode time shift as a single token
        time_shift = min(delta_time, MAX_TIME)
        return EVENT_TYPES - 1, int(time_shift * 100)  # Quantize to centiseconds

    def encode_note_event(self, msg):
        event_type = 0 if msg.type == 'note_on' and msg.velocity > 0 else 1
        return event_type, msg.note

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DiffusionTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.event_embedding = nn.Embedding(EVENT_TYPES, d_model)
        self.value_embedding = nn.Embedding(max(PITCH_RANGE, MAX_VELOCITY, int(MAX_TIME * 100)), d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, EVENT_TYPES + max(PITCH_RANGE, MAX_VELOCITY, int(MAX_TIME * 100)))

    def forward(self, x, t):
        event_type, values = x[:, :, 0], x[:, :, 1]
        
        event_emb = self.event_embedding(event_type)
        value_emb = self.value_embedding(values)
        pos_emb = self.position_embedding(torch.arange(x.size(1), device=x.device))
        
        x = event_emb + value_emb + pos_emb
        t_emb = self.get_timestep_embedding(t, x.shape[-1]).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, t_emb], dim=-1)
        
        x = self.transformer(x)
        return self.fc_out(x)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

class DiffusionModel:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, n_steps=1000):
        self.model = model
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])
        epsilon = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * epsilon, epsilon

    @torch.no_grad()
    def sample(self, shape):
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)
        for t in reversed(range(len(self.betas))):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)
            predicted_noise = self.model(x, t_tensor)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise
        
        return x

def sequence_to_midi(sequence, output_file):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    current_time = 0
    for event_type, value in sequence:
        if event_type == 2:  # Time shift
            current_time += value / 100  # Convert back from centiseconds
        else:
            msg_type = 'note_on' if event_type == 0 else 'note_off'
            track.append(Message(msg_type, note=value, velocity=64, time=int(current_time * mid.ticks_per_beat)))
            current_time = 0
    
    mid.save(output_file)

# Usage example
midi_folder = '../Midi_dataset/piano_maestro-v2.0.0/2004/'
dataset = MIDIDataset(midi_folder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = DiffusionTransformer()
diffusion = DiffusionModel(model)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        t = torch.randint(0, 1000, (batch.size(0),))
        noisy_batch, noise = diffusion.add_noise(batch, t)
        predicted_noise = model(noisy_batch, t)
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Generate new MIDI
generated = diffusion.sample((1, 512, 2))
generated_sequence = generated[0].cpu().numpy().astype(int)
sequence_to_midi(generated_sequence, 'generated_music.mid')

