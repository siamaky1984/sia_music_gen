import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from music21 import converter, note, chord, stream
import os
from tqdm import tqdm
from datetime import datetime



class EnhancedMIDIDataset(Dataset):
    def __init__(self, midi_folder, sequence_length=16, context_length=4):
        self.sequence_length = sequence_length
        self.context_length = context_length  # Number of previous sequences to include
        self.total_length = sequence_length * (context_length + 1)  # Total length including context
        self.sequences = []
        
        for file in os.listdir(midi_folder):
            if file.endswith('.mid') or file.endswith('.midi'):
                path = os.path.join(midi_folder, file)
                self.sequences.extend(self._process_midi_with_context(path))
    
    def _process_midi_with_context(self, midi_path):
        sequences = []
        midi = converter.parse(midi_path)
        
        # Extract notes and chords
        notes = []
        for element in midi.flatten():
            if isinstance(element, note.Note):
                notes.append(element.pitch.midi)
            elif isinstance(element, chord.Chord):
                notes.append(element.root().midi)
        
        # Create sequences with context
        for i in range(0, len(notes) - self.total_length + 1):
            # Get the full sequence including context
            full_sequence = notes[i:i + self.total_length]
            
            # print('>>>>>>>>', full_sequence)

            # Split into context and target sequences
            context = full_sequence[:-self.sequence_length]
            target = full_sequence[-self.sequence_length:]
            
            sequences.append({
                'context': context,
                'target': target
            })
            
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'context': torch.tensor(self.sequences[idx]['context'], dtype=torch.long),
            'target': torch.tensor(self.sequences[idx]['target'], dtype=torch.long)
        }

class PolyphonicMIDIDataset(Dataset):
    def __init__(self, midi_folder, sequence_length=16,  context_length=4, time_step=0.5):
        self.sequence_length = sequence_length
        self.time_step = time_step  # Duration of each time step in quarters
        self.context_length = context_length
        self.total_length = sequence_length * (context_length + 1)  # Total length including context
        self.sequences = []
        
        for file in os.listdir(midi_folder):
            if file.endswith('.mid') or file.endswith('.midi'):
                path = os.path.join(midi_folder, file)
                self.sequences.extend(self._process_polyphonic_midi(path))
    
    def _process_polyphonic_midi(self, midi_path):
        sequences = []
        score = converter.parse(midi_path)
        
        # Create a piano roll representation
        # Each time step will contain all simultaneously playing notes
        piano_roll = []
        current_notes = set()
        
        # Quantize score to fixed time steps
        quantized = score.quantize((self.time_step,)) #, processHierarchically=True)
        
        for element in quantized.flat.notesAndRests:
            if isinstance(element, note.Note):
                current_notes.add(element.pitch.midi)
            elif isinstance(element, chord.Chord):
                for pitch in element.pitches:
                    current_notes.add(pitch.midi)
            
            # If this is a new time step
            if element.duration.quarterLength == self.time_step:
                # Add current notes to piano roll
                piano_roll.append(sorted(list(current_notes)))
                current_notes = set()
        
        # Only process if we have enough notes for at least one sequence
        if len(piano_roll) >= self.total_length:
            # Create sequences of specified length
            for i in range(len(piano_roll) - self.sequence_length + 1):
                # sequence = piano_roll[i:i + self.sequence_length]
                # sequences.append(sequence)
                full_sequence = piano_roll[i:i + self.total_length]
                # Split into context and target
                context = full_sequence[:-self.sequence_length]
                target = full_sequence[-self.sequence_length:]
                
                # sequences.append({
                #     'context': self._encode_sequence(context),
                #     'target': self._encode_sequence(target)
                # })

                sequences.append({
                            'context': self._encode_sequence(context, self.sequence_length * self.context_length),
                            'target': self._encode_sequence(target, self.sequence_length)
                        })

        return sequences
    
    # def _encode_sequence(self, sequence):
    #     """Convert note sequence to multi-hot encoding"""
    #     encoded = torch.zeros((len(sequence), 128))
    #     for t, notes in enumerate(sequence):
    #         encoded[t, notes] = 1
    #     return encoded
    
    def _encode_sequence(self, sequence, desired_length):
        """Convert note sequence to multi-hot encoding with padding"""
        # Initialize tensor with zeros (padding)
        encoded = torch.zeros((desired_length, 128))
        
        # Fill in the actual sequence
        for t, notes in enumerate(sequence):
            if t < desired_length:  # Only fill up to desired length
                encoded[t, notes] = 1
        
        return encoded
    
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # # Convert sequence to multi-hot encoding
        # sequence = self.sequences[idx]
        # encoded = torch.zeros((self.sequence_length, 128))  # 128 MIDI notes
        
        # for t, notes in enumerate(sequence):
        #     encoded[t, notes] = 1
        
        # return encoded
        return {
            'context': self.sequences[idx]['context'],
            'target': self.sequences[idx]['target']
        }


class TemporalTransformerDenoiser(nn.Module):
    def __init__(self, vocab_size=128, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Add specific encodings for context and target sequences
        self.context_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        self.target_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, context, target, timestep):
        # Embed both context and target
        context_emb = self.embedding(context)
        target_emb = self.embedding(target)
        
        # Add position and sequence type encodings
        context_emb = context_emb + self.pos_encoding[:, :context.size(1), :] + self.context_encoding[:, :context.size(1), :]
        target_emb = target_emb + self.pos_encoding[:, :target.size(1), :] + self.target_encoding[:, :target.size(1), :]
        
        # Concatenate context and target
        x = torch.cat([context_emb, target_emb], dim=1)
        
        # Add timestep embedding
        time_embed = torch.sin(timestep[:, None, None] * torch.exp(torch.linspace(0, -10, x.size(-1))[None, None, :]).to(self.device))
        x = x + time_embed.to(x.device)
        
        # Transform
        x = self.transformer(x)
        
        # Only predict the target sequence
        x = x[:, -target.size(1):, :]
        
        return self.fc_out(x)



class PolyphonicTransformerDenoiser(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, device='cuda'):
        super().__init__()
        # self.embedding = nn.Linear(128, d_model)  # 128 MIDI notes
        # Correct initialization of Linear layers
        self.embedding = nn.Linear(in_features=128, out_features=d_model, device=device)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model, device=device))
        
        # Separate encodings for context and target
        self.context_encoding = nn.Parameter(torch.randn(1, 1000, d_model, device=device))
        self.target_encoding = nn.Parameter(torch.randn(1, 1000, d_model, device=device))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
            device =  device
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.output = nn.Linear(d_model, 128)  # Output probabilities for each note
        
        # Output layer
        self.output = nn.Linear(in_features=d_model, out_features=128, device= device)
        
    # def forward(self, x, timestep):
    #     # x shape: [batch_size, seq_length, 128]
    #     x = self.embedding(x)
        
    #     # Add positional encoding
    #     seq_len = x.size(1)
    #     x = x + self.pos_encoding[:, :seq_len, :]
        
    #     # Add timestep embedding
    #     time_embed = self.get_timestep_embedding(timestep, x.shape[-1])
    #     x = x + time_embed.unsqueeze(1)
        
    #     x = self.transformer(x)
    #     return self.output(x)  # Returns logits for each MIDI note
    
    def forward(self, context, target, timestep):
        # Embed both context and target
        context_emb = self.embedding(context)
        target_emb = self.embedding(target)
        
        # Add position and sequence type encodings
        context_emb = context_emb + self.pos_encoding[:, :context.size(1), :] + self.context_encoding[:, :context.size(1), :]
        target_emb = target_emb + self.pos_encoding[:, :target.size(1), :] + self.target_encoding[:, :target.size(1), :]
        
        # Concatenate context and target
        x = torch.cat([context_emb, target_emb], dim=1)
        
        device='cuda'

        # Add timestep embedding
        time_embed = torch.sin(timestep[:, None, None] * torch.exp(torch.linspace(0, -10, x.size(-1))[None, None, :]).to(device) )
        x = x + time_embed.to(x.device)
        
        # Transform
        x = self.transformer(x)
        
        # Only predict the target sequence
        x = x[:, -target.size(1):, :]

        return self.output(x)
    


class EnhancedDiscreteDiffusion:
    def __init__(self, n_steps=1000, vocab_size=128, device='cuda', sequence_length = 16):
        self.n_steps = n_steps
        self.vocab_size = vocab_size
        self.device = device
        self.corruption_rates = torch.linspace(0, 0.99, n_steps).to(device)
        # self.model = TemporalTransformerDenoiser(vocab_size=vocab_size).to(device)
        self.model = PolyphonicTransformerDenoiser().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.sequence_length = sequence_length

    def corrupt_sequence(self, sequence, step):
        """Add noise to the sequence based on current step"""
        corrupted = sequence.clone()
        
        
        
        # ## for melody only 
        # batch_size, seq_length = corrupted.shape
        # ## Reshape corruption rates to match sequence shape
        # #[batch_size] -> [batch_size, 1] -> [batch_size, seq_length]
        # corruption_rate = self.corruption_rates[step][:, None].expand(-1, seq_length)
        # # Generate random mask matching the sequence shape
        # mask = (torch.rand_like(corrupted.float()) < corruption_rate)



        ### for polyphonic 
        batch_size, seq_length, num_notes = corrupted.shape
        corruption_rate = self.corruption_rates[step][:, None, None].expand(batch_size, seq_length, num_notes)
        # mask = (torch.rand_like(corrupted.float()) < self.corruption_rates[step].to(self.device) )

        # Generate random mask
        mask = (torch.rand_like(corrupted) < corruption_rate)


        # random_notes = torch.randint_like(corrupted, 0, self.vocab_size)

        ### polphonic
        random_notes = torch.rand_like(corrupted) # > 0.9  # Random sparse activations

        corrupted[mask] = random_notes[mask]

        return corrupted
    
    def train_step(self, batch):
        self.model.train()

        context = batch['context'].to(self.device)
        target = batch['target'].to(self.device)

        print('target size', target.size())
        
        # Randomly select timesteps
        # t = torch.randint(0, self.n_steps, (target.size(0),), device=self.device)

        # Generate timesteps
        batch_size = target.size(0)
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        
        # Corrupt only the target sequence
        corrupted_target = self.corrupt_sequence(target, t).to(self.device)
        
        # Predict original sequence using context
        pred = self.model(context, corrupted_target, t)
        
        # Calculate loss
        loss = F.cross_entropy(pred.view(-1, self.vocab_size), target.view(-1, self.vocab_size))

        # Calculate loss polyphonic
        # loss = F.binary_cross_entropy_with_logits(pred, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, epochs=100):
        """Train the model"""
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
                loss = self.train_step(batch)
                total_loss += loss
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

        
    ### save model 

    def save_checkpoint(self, epoch, loss, checkpoint_dir='checkpoints_poly'):
        """Save model checkpoint"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'diffusion_model_poly_epoch{epoch}_{timestamp}.pt'
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'n_steps': self.n_steps,
            'vocab_size': self.vocab_size,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        return checkpoint_path
    

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Verify model configuration matches
        assert self.n_steps == checkpoint['n_steps'], "n_steps mismatch"
        assert self.vocab_size == checkpoint['vocab_size'], "vocab_size mismatch"
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    # @torch.no_grad()
    # def generate(self, context, sequence_length=16, temperature=1.0):
    #     """Generate new sequence given context"""
    #     self.model.eval()
    #     context = context.to(self.device)
        
    #     # Start with random sequence
    #     sequence = torch.randint(0, self.vocab_size, (1, sequence_length), device=self.device)
        
    #     # Gradually denoise while considering context
    #     for step in reversed(range(self.n_steps)):
    #         pred = self.model(context, sequence, torch.tensor([step], device=self.device))
    #         pred = F.softmax(pred / temperature, dim=-1)
    #         sequence = torch.multinomial(pred.view(-1, self.vocab_size), 1).view(1, -1)
        
    #     return sequence.cpu().numpy()[0]

    # @torch.no_grad()
    # def generate(self, context=None, sequence_length=None, temperature=1.0):
    #     """
    #     Generate new sequence with optional context
    #     Args:
    #         context: Optional tensor of shape [1, context_length] containing previous notes
    #                 If None, will use random notes as context
    #         sequence_length: Length of sequence to generate
    #         temperature: Controls randomness in generation
    #     """
    #     self.model.eval()

    #             # Use instance sequence_length if none provided
    #     if sequence_length is None:
    #         sequence_length = self.sequence_length
        
    #     # If no context provided, create random context
    #     if context is None:
    #         context_length = sequence_length * 4  # Same as training context length
    #         context = torch.randint(0, self.vocab_size, 
    #                               (1, context_length), 
    #                               device=self.device)
    #     else:
    #         context = context.to(self.device)
        
    #     # Start with random sequence
    #     sequence = torch.randint(0, self.vocab_size, 
    #                            (1, sequence_length), 
    #                            device=self.device)
        
    #     # Gradually denoise
    #     for step in reversed(range(self.n_steps)):
    #         # Get model prediction using context
    #         pred = self.model(context, sequence, 
    #                         torch.tensor([step], device=self.device))
            
    #         # Apply temperature and sample
    #         pred = F.softmax(pred / temperature, dim=-1)
    #         sequence = torch.multinomial(pred.view(-1, self.vocab_size), 
    #                                   1).view(1, -1)
        
    #     return sequence.cpu().numpy()[0], context.cpu().numpy()[0]
    
    @torch.no_grad()
    def generate(self, context=None, sequence_length=16, temperature=1.0):
        """Generate new sequence with context"""
        self.model.eval()
        
        # Debug prints for initial setup
        print("Starting generation...")

        if context is None:
            context = torch.zeros((1, sequence_length * 4, 128)).to(self.device)
        else:
            context = context.to(self.device)
        
        print(f"Context stats - min: {context.min().item():.4f}, max: {context.max().item():.4f}")
        

        # Start with random sequence
        sequence = torch.ones((1, sequence_length, 128)).to(self.device)* 0.01 #  > 0.9
        
        # Gradually denoise
        for step in reversed(range(self.n_steps)):
            if torch.isnan(sequence).any():
                print(f"NaN detected in sequence at step {step}")
                break
    
            pred = self.model(context, sequence, torch.tensor([step], device=self.device))


            # Clip predictions to prevent extreme values
            pred = torch.clamp(pred, -20, 20)
            
            # Apply sigmoid with temperature
            temp = max(0.1, temperature)  # Prevent division by zero

            pred = torch.sigmoid(pred / temperature)

            # sequence = (pred > 0.5).float()
            sequence = pred

            # Print stats every 100 steps
            if step % 100 == 0:
                print(f"Step {step} stats:")
                print(f"Sequence - min: {sequence.min().item():.4f}, max: {sequence.max().item():.4f}")
            

        # Final checks and cleanup
        sequence = torch.nan_to_num(sequence, 0.0)  # Replace any NaN with 0
        sequence = torch.clamp(sequence, 0, 1)      # Ensure valid range
        
        print("Generation complete")
        print(f"Final sequence stats - min: {sequence.min().item():.4f}, max: {sequence.max().item():.4f}")
        

        return sequence.cpu().numpy()[0], context.cpu().numpy()[0]
    
        # return sequence, context 



def train_and_save(midi_folder, num_epochs=100, save_interval=10):
    """Train model and save checkpoints periodically"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and dataset
    diffusion = EnhancedDiscreteDiffusion(n_steps=100, vocab_size=128, device=device)
    # dataset = EnhancedMIDIDataset(midi_folder=midi_folder)
    dataset = PolyphonicMIDIDataset(midi_folder=midi_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            loss = diffusion.train_step(batch)
            total_loss += loss
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            diffusion.save_checkpoint(epoch + 1, avg_loss)
    
    # Save final model
    final_path = diffusion.save_checkpoint(num_epochs, avg_loss)
    return final_path



def generate_from_checkpoint(checkpoint_path, num_sequences=8, sequence_length=32):
    """Load model and generate new MIDI sequences"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and load checkpoint
    diffusion = EnhancedDiscreteDiffusion(n_steps=100, vocab_size=128, device=device, sequence_length=sequence_length)
    diffusion.load_checkpoint(checkpoint_path)
    
    # Generate sequences
    # generated_notes = generate_midi_sequence(diffusion, num_sequences, sequence_length)

    context = None 
    all_sequences = []

    for i in range(num_sequences):
        print(f"\nGenerating sequence {i+1}/{num_sequences}")
        
        # Generate new sequence
        sequence, context = diffusion.generate(
            context=context,
            sequence_length=16,
            temperature=1.0
        )
    

        all_sequences.append(sequence)
        context = torch.tensor(sequence).unsqueeze(0)
    
    # Save this sequence
    output_file = f"generated_sequence.mid"
    generate_polyphonic_midi(all_sequences, output_file)

    # Optionally combine all sequences into one file
    if num_sequences > 1:
        combined_sequence = np.concatenate(all_sequences, axis=0)
        generate_polyphonic_midi(combined_sequence, "combined_sequence.mid")
    


    return all_sequences




# Example usage for generating a sequence of MIDI files
def generate_midi_sequence(diffusion_model, num_sequences=8, sequence_length= 32):
    all_notes = []
    context = None
    
    print('>>>>>>>>', num_sequences)
    for i in range(num_sequences):
        # Generate new sequence using previous context
        new_sequence, context = diffusion_model.generate(context=context,  sequence_length=sequence_length)
        all_notes.extend(new_sequence)
        
        print('context', len(context))
        print(len(all_notes))

        # Update context for next generation
        # Use the last portion of combined context and new sequence
        full_sequence = np.concatenate([context, new_sequence])
        context_length = len(context)
        context = torch.tensor(full_sequence[-context_length:]).unsqueeze(0)
    
    # Convert to MIDI
    s = stream.Stream()
    for pitch in all_notes:
        n = note.Note(pitch, quarterLength=0.5)
        s.append(n)
    print(len(s))
    s.write('midi', fp='generated_sequence.mid')
    return all_notes


# def generate_polyphonic_midi(sequence, output_file="generated.mid"):
#     """Convert multi-hot encoded sequence to MIDI file"""
#     s = stream.Stream()
    
#     for step, notes in enumerate(sequence):
#         # Get active notes (where probability > threshold)
#         # active_notes = torch.where(notes > 0.5)[0].cpu().numpy()

#         active_notes = np.where( notes> 0.5)[0]
        
#         if len(active_notes) > 0:
#             # If multiple notes, create a chord
#             if len(active_notes) > 1:
#                 c = chord.Chord([int(n) for n in active_notes])
#                 c.quarterLength = 0.5
#                 s.append(c)
#             # If single note
#             else:
#                 n = note.Note(int(active_notes[0]))
#                 n.quarterLength = 0.5
#                 s.append(n)
#         else:
#             # Rest if no notes
#             r = note.Rest()
#             r.quarterLength = 0.5
#             s.append(r)
    
#     s.write('midi', fp=output_file)


def generate_polyphonic_midi(sequences, output_file="generated.mid"):
    """
    Convert numpy array of note probabilities to MIDI file
    Args:
        sequence: numpy array of shape [sequence_length, 128] with values between 0 and 1
        output_file: path to save the MIDI file
    """
    from music21 import stream, note, chord
        
    print(f"\nGenerating MIDI file: {output_file}")

    
    s = stream.Stream()
    total_notes = 0

    # Process each sequence
    for seq_idx, sequence in enumerate(sequences):
        # Replace any NaN values and ensure values are in [0,1]
        sequence = np.nan_to_num(sequence, 0.0)
        sequence = np.clip(sequence, 0, 1)

        print(f"Sequence shape: {sequence.shape}")
        print(f"Value range: {sequence.min():.4f} to {sequence.max():.4f}")
    
        for step, step_notes in enumerate(sequence):
            # Get notes above threshold
            active_notes = np.where(step_notes > 0.3)[0]
            total_notes += len(active_notes)
            
            if len(active_notes) > 0:
                # Create chord if multiple notes
                if len(active_notes) > 1:
                    c = chord.Chord([int(n) for n in active_notes])
                    c.quarterLength = 0.5
                    s.append(c)
                # Create single note
                else:
                    n = note.Note(int(active_notes[0]))
                    n.quarterLength = 0.5
                    s.append(n)
            else:
                # Add rest if no notes
                r = note.Rest()
                r.quarterLength = 0.5
                s.append(r)
    
    print(f"Total notes in sequence: {total_notes}")
    s.write('midi', fp=output_file)



def main():
    # # Initialize dataset and dataloader
    # dataset = EnhancedMIDIDataset(midi_folder="../midi_dataset/piano_maestro-v1.0.0/all_years/", sequence_length=16)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # # Initialize and train model
    # diffusion = EnhancedDiscreteDiffusion(n_steps=100, device='cuda' if torch.cuda.is_available() else 'cpu')
    # diffusion.train(dataloader, epochs=20)
    
    # # # Generate new sequence
    # # generated = diffusion.generate(sequence_length=16)
    # # Then generate a sequence of connected pieces
    # generated_notes = generate_midi_sequence(diffusion, num_sequences=4)

    
    # # Convert to MIDI
    # s = stream.Stream()
    # for pitch in generated:
    #     n = note.Note(pitch, quarterLength=0.5)
    #     s.append(n)
    # s.write('midi', fp='generated_discrete_diff.mid')



    import argparse
    
    parser = argparse.ArgumentParser(description='Train or generate MIDI sequences')
    parser.add_argument('--mode', type=str, default= 'generate',
                       help='Whether to train model or generate sequences')
    parser.add_argument('--midi_folder', type=str, help='Folder containing MIDI files for training')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for generation')
    parser.add_argument('--num_sequences', type=int, default=8,
                       help='Number of sequences to generate')

    args = parser.parse_args()
    args.mode = 'train'
    args.checkpoint = './checkpoints_poly/diffusion_model_poly_epoch50_20241206_185448.pt'
    args.midi_folder = '../midi_dataset/piano_maestro/piano_maestro-v1.0.0/all_years/'
   

    if args.mode == 'train':
        if not args.midi_folder:
            raise ValueError("midi_folder must be specified for training")
        checkpoint_path = train_and_save(args.midi_folder, num_epochs = 10)
        print(f"Training complete. Model saved to {checkpoint_path}")
    
    elif args.mode == 'generate':
        if not args.checkpoint:
            raise ValueError("checkpoint must be specified for generation")
        generated_notes = generate_from_checkpoint(
            args.checkpoint, 
            num_sequences=args.num_sequences
        )
        print(f"Generated {len(generated_notes)} notes")

if __name__ == "__main__":
    main()