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
                sequences.append(torch.tensor(seq, dtype=torch.float32))

        return sequences

    def encode_time(self, delta_time):
        # Encode time shift as a continuous value
        time_shift = min(delta_time, MAX_TIME)
        return [2.0, time_shift / MAX_TIME]  # Use 2 for time_shift event type

    def encode_note_event(self, msg):
        event_type = 0 if msg.type == 'note_on' and msg.velocity > 0 else 1
        return [event_type, msg.note / PITCH_RANGE]  # Normalize pitch to [0, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DiffusionTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model
        self.event_type_embedding = nn.Embedding(EVENT_TYPES, d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, 2)  # Output event type and value

    def forward(self, x, t):
        batch_size, seq_len, _ = x.shape
        event_type = x[:, :, 0].long()
        values = x[:, :, 1].unsqueeze(-1)
        
        # Ensure event_type is within valid range
        event_type = torch.clamp(event_type, 0, EVENT_TYPES - 1)
        
        event_emb = self.event_type_embedding(event_type)
        value_emb = self.value_projection(values)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        x = event_emb + value_emb + pos_emb
        
        # Reshape t_emb to match x's sequence length
        t_emb = self.get_timestep_embedding(t, self.d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        x = torch.cat([x, t_emb], dim=-1)
        
        # Project back to d_model dimension
        x = nn.Linear(x.size(-1), self.d_model, device=x.device)(x)
        
        x = self.transformer(x)
        return self.fc_out(x)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
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
        return sqrt_alpha_bar.unsqueeze(-1).unsqueeze(-1) * x + sqrt_one_minus_alpha_bar.unsqueeze(-1).unsqueeze(-1) * epsilon, epsilon

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


# Usage example
midi_folder = '../Midi_dataset/piano_maestro-v2.0.0/2004/'
dataset = MIDIDataset(midi_folder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionTransformer().to(device)
diffusion = DiffusionModel(model)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 3

for epoch in range(num_epochs):
    print('epoch', epoch)
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        t = torch.randint(0, 1000, (batch.size(0),), device=device)
        
        # print(batch.shape, t.shape)
        noisy_batch, noise = diffusion.add_noise(batch, t)
        # print('>>>', noisy_batch.shape, t.shape, noise.shape)

        predicted_noise = model(noisy_batch, t)
        # print(predicted_noise.shape, noise.shape)
        
        loss = F.mse_loss(predicted_noise, noise)

        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


# Save the model after training
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'loss': loss.item(),
}, 'midi_diffusion_model.pth')

print("Model saved successfully.")



def scale_to_midi_range(generated_sequence):
    # Scale event types to integers
    event_types = np.round(generated_sequence[:, 0]).astype(int)
    
    # Scale values based on event type
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
    for event in sequence:
        event_type, value = event
        if event_type == 2:  # Time shift
            current_time += value
        else:
            msg_type = 'note_on' if event_type == 0 else 'note_off'
            note = int(value)
            velocity = 64 if msg_type == 'note_on' else 0
            track.append(Message(msg_type, note=note, velocity=velocity, time=int(current_time * mid.ticks_per_beat)))
            current_time = 0
    
    mid.save(output_file)

def generate_midi(model_path, output_file, seq_len=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTransformer().to(device)
    diffusion = DiffusionModel(model)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully.")
    
    with torch.no_grad():
        generated = diffusion.sample((1, seq_len, 2))
    
    generated_sequence = generated[0].cpu().numpy()
    
    # Scale the generated sequence to appropriate MIDI ranges
    scaled_sequence = scale_to_midi_range(generated_sequence)
    
    sequence_to_midi(scaled_sequence, output_file)
    print(f"Generated MIDI saved to {output_file}")

# Example usage of the inference function
generate_midi('midi_diffusion_model.pth', 'generated_music.mid')
