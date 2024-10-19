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
        time_shift = min(delta_time, MAX_TIME)
        return [2, time_shift / MAX_TIME]  # Use 2 for time_shift event type

    def encode_note_event(self, msg):
        event_type = 0 if msg.type == 'note_on' and msg.velocity > 0 else 1
        return [event_type, msg.note / PITCH_RANGE]  # Normalize pitch to [0, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



class DiffusionTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.event_type_embedding = nn.Embedding(EVENT_TYPES, d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
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
        
        event_type = torch.clamp(event_type, 0, EVENT_TYPES - 1)
        
        event_emb = self.event_type_embedding(event_type)
        value_emb = self.value_projection(values)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        x = event_emb + value_emb + pos_emb
        
        t_emb = self.get_timestep_embedding(t, self.d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        x = torch.cat([x, t_emb], dim=-1)
        
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
        self.device = next(model.parameters()).device
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])
        epsilon = torch.randn_like(x)
        return sqrt_alpha_bar.unsqueeze(-1).unsqueeze(-1) * x + sqrt_one_minus_alpha_bar.unsqueeze(-1).unsqueeze(-1) * epsilon, epsilon

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(len(self.betas))):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)
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




def scale_to_midi_range(generated_sequence):
    event_types = np.clip(np.round(generated_sequence[:, 0]).astype(int), 0, EVENT_TYPES - 1)
    values = generated_sequence[:, 1]
    scaled_values = np.zeros_like(values)
    
    for i, event_type in enumerate(event_types):
        if event_type == 0 or event_type == 1:  # note on or note off
            scaled_values[i] = np.clip(np.round(values[i] * PITCH_RANGE), 0, 127).astype(int)
        elif event_type == 2:  # time shift
            scaled_values[i] = np.clip(values[i] * MAX_TIME, 0, MAX_TIME)
    
    return np.stack([event_types, scaled_values], axis=-1)


def post_process_sequence(sequence):
    processed = []
    current_notes = set()
    accumulated_time = 0
    
    for event in sequence:
        event_type, value = event
        event_type = int(event_type)
        
        if event_type == 2:  # time shift
            accumulated_time += max(0, value)  # Ensure non-negative time
        elif event_type == 0:  # note on
            if value not in current_notes:
                processed.append([event_type, int(value), accumulated_time])
                current_notes.add(value)
                accumulated_time = 0
        elif event_type == 1:  # note off
            if value in current_notes:
                processed.append([event_type, int(value), accumulated_time])
                current_notes.remove(value)
                accumulated_time = 0
    
    # Ensure all notes are turned off at the end
    for note in current_notes:
        processed.append([1, int(note), 0])
    
    return processed


def sequence_to_midi(sequence, output_file):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    for event in sequence:
        event_type, note, time = event
        msg_type = 'note_on' if event_type == 0 else 'note_off'
        velocity = 64 if msg_type == 'note_on' else 0
        
        # Ensure time is non-negative and convert to ticks
        ticks = max(0, int(time * mid.ticks_per_beat))
        
        track.append(Message(msg_type, note=note, velocity=velocity, time=ticks))
    
    mid.save(output_file)

def generate_midi(model_path, output_file, seq_len=512, num_sequences=8):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    device = get_device()

    model = DiffusionTransformer(max_seq_len=seq_len).to(device)
    diffusion = DiffusionModel(model)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully.")
    
    full_sequence = []
    with torch.no_grad():
        for _ in range(num_sequences):
            generated = diffusion.sample((1, seq_len, 2))
            generated_sequence = generated[0].cpu().numpy()
            scaled_sequence = scale_to_midi_range(generated_sequence)
            full_sequence.append(scaled_sequence)
    
    full_sequence = np.concatenate(full_sequence, axis=0)
    
    # Post-process the full sequence
    processed_sequence = post_process_sequence(full_sequence)
    
    sequence_to_midi(processed_sequence, output_file)
    print(f"Generated MIDI saved to {output_file}")



if __name__=='__main__':
    mode = 'train' # 'inference' #'train'

    if mode == 'train':

        # Usage example
        midi_folder = '../Midi_dataset/piano_maestro-v2.0.0/2004/'
        dataset = MIDIDataset(midi_folder)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        model = DiffusionTransformer().to(device)
        diffusion = DiffusionModel(model)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        num_epochs = 2

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

    elif mode =='inference':

        # Example usage of the inference function
        generate_midi('midi_diffusion_model.pth', 'generated_music.mid', seq_len=512, num_sequences=8 )