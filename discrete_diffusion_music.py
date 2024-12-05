import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from music21 import converter, note, chord, stream
import os
from tqdm import tqdm
from datetime import datetime

'''
class MIDIDataset(Dataset):
    def __init__(self, midi_folder, sequence_length=16):
        self.sequence_length = sequence_length
        self.sequences = []
        
        # Process all MIDI files in the folder
        for file in os.listdir(midi_folder):
            if file.endswith('.mid') or file.endswith('.midi'):
                path = os.path.join(midi_folder, file)
                self.sequences.extend(self._process_midi(path))
                
    def _process_midi(self, midi_path):
        sequences = []
        midi = converter.parse(midi_path)
        
        # Extract notes and chords
        notes = []
        for element in midi.flat:
            if isinstance(element, note.Note):
                print('element', element.pitch.midi )
                notes.append(element.pitch.midi)
            elif isinstance(element, chord.Chord):
                print('element', element.root().midi )
                notes.append(element.root().midi)

            print('notes', len(notes) )
        
        # Create sequences of specified length
        for i in range(0, len(notes) - self.sequence_length + 1):
            sequence = notes[i:i + self.sequence_length]
            sequences.append(sequence)
            
        print('sequence', len(sequence) )
        print('sequences', len(sequences) )
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

class TransformerDenoiser(nn.Module):
    def __init__(self, vocab_size=128, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, timestep):
        # Embed the input sequence
        x = self.embedding(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Add timestep embedding
        time_embed = torch.sin(timestep[:, None, None] * torch.exp(torch.linspace(0, -10, x.size(-1))[None, None, :]))
        x = x + time_embed.to(x.device)
        
        # Transform
        x = self.transformer(x)
        
        # Project to vocabulary
        return self.fc_out(x)

class DiscreteDiffusion:
    def __init__(self, n_steps=1000, vocab_size=128, device='cuda'):
        self.n_steps = n_steps
        self.vocab_size = vocab_size
        self.device = device
        self.corruption_rates = torch.linspace(0, 0.99, n_steps)
        self.model = TransformerDenoiser(vocab_size=vocab_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def corrupt_sequence(self, sequence, step):
        """Add noise to the sequence based on current step"""
        corrupted = sequence.clone()
        mask = torch.rand_like(corrupted.float()) < self.corruption_rates[step]
        random_notes = torch.randint_like(corrupted, 0, self.vocab_size)
        corrupted[mask] = random_notes[mask]
        return corrupted
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        batch = batch.to(self.device)
        
        # Randomly select timesteps
        t = torch.randint(0, self.n_steps, (batch.size(0),), device=self.device)
        
        # Corrupt the sequences
        corrupted = torch.stack([self.corrupt_sequence(seq, step) for seq, step in zip(batch, t)])
        
        # Predict original sequence
        pred = self.model(corrupted, t)
        
        # Calculate loss
        loss = F.cross_entropy(pred.view(-1, self.vocab_size), batch.view(-1))
        
        # Optimize
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
    
    @torch.no_grad()
    def generate(self, sequence_length=16, temperature=1.0):
        """Generate new sequence"""
        self.model.eval()
        
        # Start with random sequence
        sequence = torch.randint(0, self.vocab_size, (1, sequence_length), device=self.device)
        
        # Gradually denoise
        for step in reversed(range(self.n_steps)):
            # Get model prediction
            pred = self.model(sequence, torch.tensor([step], device=self.device))
            pred = F.softmax(pred / temperature, dim=-1)
            
            # Sample from prediction
            sequence = torch.multinomial(pred.view(-1, self.vocab_size), 1).view(1, -1)
        
        return sequence.cpu().numpy()[0]

# Example usage
def main():
    # Initialize dataset and dataloader
    dataset = MIDIDataset(midi_folder="./midi_dataset/piano_maestro-v1.0.0/2004/", sequence_length=16)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train model
    diffusion = DiscreteDiffusion(n_steps=100, device='cuda' if torch.cuda.is_available() else 'cpu')
    diffusion.train(dataloader, epochs=50)
    
    # Generate new sequence
    generated = diffusion.generate(sequence_length=16)
    
    # Convert to MIDI
    s = stream.Stream()
    for pitch in generated:
        n = note.Note(pitch, quarterLength=0.5)
        s.append(n)
    s.write('midi', fp='generated.mid')
'''


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
        for element in midi.flat:
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

class EnhancedDiscreteDiffusion:
    def __init__(self, n_steps=1000, vocab_size=128, device='cuda'):
        self.n_steps = n_steps
        self.vocab_size = vocab_size
        self.device = device
        self.corruption_rates = torch.linspace(0, 0.99, n_steps).to(device)
        self.model = TemporalTransformerDenoiser(vocab_size=vocab_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def corrupt_sequence(self, sequence, step):
        """Add noise to the sequence based on current step"""
        corrupted = sequence.clone()
        batch_size, seq_length = corrupted.shape

        # Reshape corruption rates to match sequence shape
        # [batch_size] -> [batch_size, 1] -> [batch_size, seq_length]
        corruption_rate = self.corruption_rates[step][:, None].expand(-1, seq_length)

        # mask = (torch.rand_like(corrupted.float()) < self.corruption_rates[step].to(corrupted.device) )

        # Generate random mask matching the sequence shape
        mask = (torch.rand_like(corrupted.float()) < corruption_rate)

        random_notes = torch.randint_like(corrupted, 0, self.vocab_size)
        corrupted[mask] = random_notes[mask]

        return corrupted
    
    def train_step(self, batch):
        self.model.train()
        context = batch['context'].to(self.device)
        target = batch['target'].to(self.device)
        
        # Randomly select timesteps
        t = torch.randint(0, self.n_steps, (target.size(0),), device=self.device)
        
        # Corrupt only the target sequence
        corrupted_target = self.corrupt_sequence(target, t)
        
        # Predict original sequence using context
        pred = self.model(context, corrupted_target, t)
        
        # Calculate loss
        loss = F.cross_entropy(pred.view(-1, self.vocab_size), target.view(-1))
        
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

    def save_checkpoint(self, epoch, loss, checkpoint_dir='checkpoints'):
        """Save model checkpoint"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'diffusion_model_epoch{epoch}_{timestamp}.pt'
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

    @torch.no_grad()
    def generate(self, context=None, sequence_length=16, temperature=1.0):
        """
        Generate new sequence with optional context
        Args:
            context: Optional tensor of shape [1, context_length] containing previous notes
                    If None, will use random notes as context
            sequence_length: Length of sequence to generate
            temperature: Controls randomness in generation
        """
        self.model.eval()
        
        # If no context provided, create random context
        if context is None:
            context_length = sequence_length * 4  # Same as training context length
            context = torch.randint(0, self.vocab_size, 
                                  (1, context_length), 
                                  device=self.device)
        else:
            context = context.to(self.device)
        
        # Start with random sequence
        sequence = torch.randint(0, self.vocab_size, 
                               (1, sequence_length), 
                               device=self.device)
        
        # Gradually denoise
        for step in reversed(range(self.n_steps)):
            # Get model prediction using context
            pred = self.model(context, sequence, 
                            torch.tensor([step], device=self.device))
            
            # Apply temperature and sample
            pred = F.softmax(pred / temperature, dim=-1)
            sequence = torch.multinomial(pred.view(-1, self.vocab_size), 
                                      1).view(1, -1)
        
        return sequence.cpu().numpy()[0], context.cpu().numpy()[0]
    



def train_and_save(midi_folder, num_epochs=100, save_interval=10):
    """Train model and save checkpoints periodically"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and dataset
    diffusion = EnhancedDiscreteDiffusion(n_steps=100, vocab_size=128, device=device)
    dataset = EnhancedMIDIDataset(midi_folder=midi_folder)
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



def generate_from_checkpoint(checkpoint_path, num_sequences=4):
    """Load model and generate new MIDI sequences"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and load checkpoint
    diffusion = EnhancedDiscreteDiffusion(n_steps=100, vocab_size=128, device=device)
    diffusion.load_checkpoint(checkpoint_path)
    
    # Generate sequences
    generated_notes = generate_midi_sequence(diffusion, num_sequences)
    return generated_notes




# Example usage for generating a sequence of MIDI files
def generate_midi_sequence(diffusion_model, num_sequences=4):
    all_notes = []
    context = None
    
    for i in range(num_sequences):
        # Generate new sequence using previous context
        new_sequence, context = diffusion_model.generate(context=context)
        all_notes.extend(new_sequence)
        
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
    s.write('midi', fp='generated_sequence.mid')
    return all_notes



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
    parser.add_argument('--mode', type=str, required=True,
                       help='Whether to train model or generate sequences')
    parser.add_argument('--midi_folder', type=str, help='Folder containing MIDI files for training')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for generation')
    parser.add_argument('--num_sequences', type=int, default=8,
                       help='Number of sequences to generate')

    args = parser.parse_args()

    # args.checkpoint = './checkpoints/'
    # args.midi_folder ='../midi_dataset/piano_maestro-v1.0.0/2004/'
    # args.mode = 'train'

    if args.mode == 'train':
        if not args.midi_folder:
            raise ValueError("midi_folder must be specified for training")
        checkpoint_path = train_and_save(args.midi_folder, num_epochs = 20)
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