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


class MIDIBarsDataset(Dataset):
    def __init__(self, midi_folder, bars_per_sequence=8, bar_length=512):
        self.midi_folder = midi_folder
        self.bars_per_sequence = bars_per_sequence
        self.bar_length = bar_length
        self.sequences = self.load_midi_files()


    def transpose_bar(self, bar, transpose_amount):
        transposed = bar.clone()
        note_events_mask = (bar[:, 0] == 0) | (bar[:, 0] == 1)
        if note_events_mask.any():
            notes = bar[note_events_mask, 1] * 127
            notes = notes + transpose_amount
            notes = torch.clamp(notes, 0, 127)
            transposed[note_events_mask, 1] = notes / 127
        return transposed

    def time_stretch(self, bar, factor):
        stretched = bar.clone()
        time_events_mask = (bar[:, 0] == 2)
        if time_events_mask.any():
            time_values = bar[time_events_mask, 1]
            time_values = time_values * factor
            stretched[time_events_mask, 1] = torch.clamp(time_values, 0, 1)
        return stretched

    def velocity_adjust(self, bar, scale_factor):
        adjusted = bar.clone()
        note_on_mask = (bar[:, 0] == 0)
        if note_on_mask.any():
            velocities = bar[note_on_mask, 1] * scale_factor
            adjusted[note_on_mask, 1] = torch.clamp(velocities, 0, 1)
        return adjusted
    

    def load_midi_files(self):
        sequences = []
        for filename in os.listdir(self.midi_folder):
            if filename.endswith(('.mid', '.midi')):
                file_path = os.path.join(self.midi_folder, filename)
                bars = self.extract_bars_from_midi(file_path)
                
                # Create sequences of consecutive bars
                for i in range(0, len(bars) - self.bars_per_sequence + 1):
                    bar_sequence = bars[i:i + self.bars_per_sequence]
                    sequences.append(bar_sequence)


                    ##### augmentation 
                    # Transpose
                    for transpose in [-3, -2, -1, 1, 2, 3]:
                        transposed_sequence = [self.transpose_bar(bar, transpose) for bar in bar_sequence]
                        sequences.append(transposed_sequence)

                    # Time stretch
                    for stretch in [0.9, 1.1]:
                        stretched_sequence = [self.time_stretch(bar, stretch) for bar in bar_sequence]
                        sequences.append(stretched_sequence)

                    # # Velocity adjustment
                    # for velocity in [0.8, 1.2]:
                    #     adjusted_sequence = [self.velocity_adjust(bar, velocity) for bar in bar_sequence]
                    #     sequences.append(adjusted_sequence)
        
        return sequences


    def extract_bars_from_midi(self, midi_file):
        mid = MidiFile(midi_file)
        ticks_per_bar = mid.ticks_per_beat * 4  # Assuming 4/4 time signature

        C4 = 60
        
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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.stack(sequence)



####################### VQVAE###############################


def train_vqvae_on_bars(model, dataloader, num_epochs=100, device="cuda", save_path=None):
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

            # batch shape: [batch_size, bars_per_sequence, bar_length, 2]
            # Flatten the sequence dimension to process each bar independently
            bars = batch.view(-1, batch.size(2), batch.size(3))
            x= bars.to(device)

            
            # x = batch.to(device)
            # x_recon, vq_loss, perplexity, _ = model(x)

            x_recon, vq_loss, indices = model(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)
            
            # Total loss
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            # total_perplexity += perplexity.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_vq_loss = total_vq_loss / len(dataloader)
        # avg_perplexity = total_perplexity / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  VQ Loss: {avg_vq_loss:.4f}")
        # print(f"  Codebook Perplexity: {avg_perplexity:.2f}")


        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            model.save_model(save_path)
            print(f"Saved best model with loss: {best_loss:.4f}")

############################################################

##############barVQVAE #################################

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # inputs shape: [batch_size, embedding_dim]
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = torch.cdist(flat_input, self.embedding.weight)
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = self.embedding(encoding_indices)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return self.block(x) + self.shortcut(x)
    
class BarEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class BarDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class BarVQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=64, num_embeddings=512, bar_length=512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings

        self.bar_length = bar_length
        
        hidden_dims = [hidden_dim] * 3
        hidden_dims[1] = hidden_dim // 2
        hidden_dims[2] = hidden_dim // 4

        # self.encoder = BarEncoder(input_dim, hidden_dim, latent_dim)
        # Deeper encoder with residual connections
        self.encoder = nn.Sequential(
            ResidualBlock(input_dim, hidden_dims[0]),
            ResidualBlock(hidden_dims[0], hidden_dims[1]),
            ResidualBlock(hidden_dims[1], hidden_dims[2]),
            nn.Linear(hidden_dims[2], latent_dim)
        )

        self.vq = VectorQuantizer(num_embeddings, latent_dim)


        # self.decoder = BarDecoder(latent_dim, hidden_dim, input_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(latent_dim, hidden_dims[2]),
            ResidualBlock(hidden_dims[2], hidden_dims[1]),
            ResidualBlock(hidden_dims[1], hidden_dims[0]),
            nn.Linear(hidden_dims[0], input_dim)
        )

    def forward(self, x):
        # Flatten the bar sequence
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(batch_size, seq_len * features)
        
        # Encode
        z = self.encoder(x_flat)
        z_q, vq_loss, indices = self.vq(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        x_recon = x_recon.reshape(batch_size, seq_len, features)
        
        return x_recon, vq_loss, indices

    def encode(self, x):
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(batch_size, seq_len * features)
        z = self.encoder(x_flat)
        z_q, _, indices = self.vq(z)
        return z_q

    def decode(self, z_q):
        x_recon = self.decoder(z_q)
        return x_recon.reshape(-1, int(self.input_dim/2), 2)
        # return x_recon.reshape(-1, self.bar_length, 2)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim':self.hidden_dim,
                'latent_dim': self.latent_dim,
                'num_embeddings': self.num_embeddings
            }
        }, path)

    @classmethod
    # def load_model(cls, path, device='cuda'):
    #     checkpoint = torch.load(path, map_location=device)
    #     model = cls(**checkpoint['model_config'])
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     return model.to(device)
    def load_model(cls, path, device='cuda', strict=True):
        """
        Load model state and configuration from a saved file
        """
        checkpoint = torch.load(path, map_location=device)
        
        # # Create model instance with saved config
        # model = cls(
        #     input_dim=checkpoint['model_config']['input_dim'],
        #     hidden_dim=checkpoint['model_config']['hidden_dim'],
        #     num_embeddings=checkpoint['model_config']['num_embeddings'],
        #     # commitment_cost=checkpoint['model_config']['commitment_cost'],
        #     # num_heads=checkpoint['model_config']['num_heads']
        # )

        model = cls(
            input_dim=checkpoint['model_config']['input_dim'],
            hidden_dim=checkpoint['model_config']['hidden_dim'],
            latent_dim=checkpoint['model_config']['latent_dim'],
            num_embeddings=checkpoint['model_config']['num_embeddings']
        )

        # model = cls(**checkpoint['model_config'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model = model.to(device)
        model.eval()
        
        return model, checkpoint


########################################################


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

def sequence_to_midi(sequence, output_file, ticks_per_beat=480):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    current_time = 0
    last_event_time = 0


    for event in sequence:
        event_type, note, time = event 

        # if event_type == 2:  # Time shift
        #     if value<0:
        #         value =0
        #     current_time += value
        # else:
           
            # note = int(value)

            # msg_type = 'note_on' if event_type == 0 else 'note_off'
            # velocity = 64 if msg_type == 'note_on' else 0
            # track.append(Message(msg_type, note=note, velocity=velocity, time=int(current_time * mid.ticks_per_beat)))
            # current_time = 0
            
        # Calculate delta time
        delta_time = int((time - current_time) * ticks_per_beat)
        delta_time = max(0, delta_time)  # Ensure non-negative
        current_time = time
        
        # Create MIDI message
        msg_type = 'note_on' if event_type == 0 else 'note_off'
        velocity = 64 if msg_type == 'note_on' else 0
        
        track.append(Message(msg_type, note=int(note), velocity=velocity, time=delta_time))
    

    mid.save(output_file)





# Generation code
def generate_music(vae, diffusion, num_bars=8, device='cuda'):
    # Create input shape: [batch_size, num_bars, latent_dim]
    # latent_input = torch.randn(1, num_bars, vae.latent_dim).to(device)
    
    latent_input = torch.randn(1, num_bars,  64).to(device)
    
    # Generate with diffusion model
    generated_latent = diffusion.sample(latent_input.shape)
    
    # Decode each bar separately
    generated_bars = []
    for i in range(num_bars):
        bar_latent = generated_latent[0, i]  # [latent_dim]
        decoded_bar = vae.decode(bar_latent.unsqueeze(0))  # Add batch dimension # [1, bar_length, 2]
        generated_bars.append(decoded_bar)
    

    # Concatenate all bars
    generated_sequence = torch.cat(generated_bars, dim=0)  

    # generated_midi = generated_midi.reshape(-1, generated_midi.shape[2]) # [num_bars * bar_length, 2]

    return generated_sequence




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
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Usage example
    midi_folder = '../midi_dataset/piano_maestro/piano_maestro-v1.0.0/all_years/'
    sequence_len = 32
    feature_dim = 2
    batch_size = 8

    bars_per_sequence=1

    num_epochs = 100

    # dataset = MIDIDataset(midi_folder, seq_len = sequence_len)
    # dataset = AugmentedMIDIDataset(midi_folder)
    
    dataset = MIDIBarsDataset(midi_folder, bars_per_sequence=bars_per_sequence)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mode =  'train_vqvae' # 'train_diffusion_trans' #'train_vqvae' # 'train_diffusion_trans' #  # # 'train_vqvae' # 'train_diffusion_trans' # 'train_vqvae' # 'train_vae'
    # mode = 'generate'
    ### 'train_diffusion_trans', 'inference'

    vqvae_model_path = './models/vqvae_model_melody.pth'
    latent_diffusion_model_path='./models/diffusion_model_new.pth'
    
    # # Training mode
    # if mode == 'train_vae':  # Change to False for inference mode
    #     # Initialize models
    #     input_dim = batch_size * sequence_len  # Set your input dimension
    #         # Initialize VAE with correct dimensions
    #     vae = MIDIVAE(
    #         sequence_length=sequence_len,
    #         feature_dim=feature_dim,
    #         hidden_dim=512,
    #         latent_dim=64
    #     ).to(device)
            
    #     vae.apply(weights_init)

    #     # Train and save VAE
    #     train_vae(vae, dataloader, num_epochs=num_epochs, device=device, save_path='vqvae_model.pth')

    if mode == 'train_vqvae':
        # Assuming your MIDI data has shape [batch_size, sequence_length, feature_dim]
        # model = HierarchicalVQVAE(
        #     input_dim=2,        # Your MIDI feature dimension
        #     hidden_dim=512,     # Size of hidden representations
        #     embedding_dim=8,
        #     num_embeddings=512, # Size of codebook
        #     num_heads=8
        # ).to(device)


        # Create and train VQVAE
        model = BarVQVAE(
            input_dim=512 * 2,  # bar_length * feature_dim
            hidden_dim=512,
            latent_dim=64,
            num_embeddings=512,
            bar_length= 512 
        ).to(device)


         # Train the model
        train_vqvae_on_bars(model, dataloader, num_epochs=num_epochs, device=device, save_path= vqvae_model_path)
        
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
        

    # elif mode =='train_diffusion_trans':
    #     # vae = MIDIVAE.load_model('vae_model.pth', device)
    #     # vae , checkpoint = HierarchicalVQVAE.load_model(vqvae_model_path, device)  ### VQVAE
    #     vae , checkpoint = BarVQVAE.load_model(vqvae_model_path, device)

    #     print('vae.latent_dim', vae.latent_dim)
    #     # Initialize and train transformer using trained VAE
    #     transformer = LatentDiffusionTransformer(latent_dim=vae.latent_dim).to(device)
    #     train_latent_diffusion(transformer, vae, dataloader, num_epochs=num_epochs, 
    #                          device=device, save_path=latent_diffusion_model_path)
    
    # # Inference mode
    # elif mode == 'generate':
    #     # Load trained models
    #     vae, checkpoint_vae = BarVQVAE.load_model(vqvae_model_path, device)
    #     # transformer = LatentDiffusionTransformer.load_model(latent_diffusion_model_path, device)
    #     # Load Diffusion model
    #     transformer = LatentDiffusionTransformer(
    #         latent_dim=64, # vae.latent_dim,  # Now we can access this
    #         d_model=512,
    #         nhead=8,
    #         num_layers=6
    #     ).to(device)
    #     diffusion = DiffusionModel(transformer)
        
    #     # Load diffusion model weights
    #     checkpoint = torch.load(latent_diffusion_model_path, map_location=device)
    #     transformer.load_state_dict(checkpoint['model_state_dict'])


    #     generated_seq = generate_music(vae, diffusion, num_bars=16, device=device)

    #     save_generated_midi(generated_seq, './generated_vqvaeDiffTrans.mid')

        # sequence_to_midi(generated_seq, './generated_vqvaeDiffTrans.mid')

# mode = 'inference' #'train'
