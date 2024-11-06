import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import os
import glob
from collections import defaultdict
from typing import List, Tuple, Dict
import random
import pickle
import time
import matplotlib.pyplot as plt

import pygame
import librosa

from load_midi import plot_piano_roll

class MIDIEvent:
    """Represents a MIDI event with type, pitch, velocity, and time."""
    def __init__(self, event_type: str, pitch: int, velocity: int, time: float):
        self.event_type = event_type  # 'note_on' or 'note_off'
        self.pitch = pitch
        self.velocity = velocity
        self.time = time

class EventSequenceProcessor:
    """Processes MIDI events into training data."""
    def __init__(self, max_events=1000):
        self.max_events = max_events
        self.event_to_idx = {}
        self.idx_to_event = {}
        self.vocab_size = 0
        
    def _create_event_vocab(self, all_sequences: List[List[MIDIEvent]]):
        """Create vocabulary for all possible events."""
        events = set()
        for sequence in all_sequences:
            for event in sequence:
                # Create a unique identifier for each event type
                event_str = f"{event.event_type}_{event.pitch}_{event.velocity}"
                events.add(event_str)
        
        # Create bidirectional mappings
        self.event_to_idx = {event: idx for idx, event in enumerate(sorted(events))}
        self.idx_to_event = {idx: event for event, idx in self.event_to_idx.items()}
        self.vocab_size = len(self.event_to_idx)
    
    def extract_events(self, midi_file: str) -> List[MIDIEvent]:
        """Extract events from a MIDI file."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            events = []
            
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        # Note on event
                        events.append(MIDIEvent(
                            'note_on', note.pitch, note.velocity, note.start
                        ))
                        # Note off event
                        events.append(MIDIEvent(
                            'note_off', note.pitch, 0, note.end
                        ))
            
            # Sort by time
            events.sort(key=lambda x: x.time)
            return events[:self.max_events]
        
        except Exception as e:
            print(f"Error processing {midi_file}: {str(e)}")
            return []

    def events_to_indices(self, events: List[MIDIEvent]) -> List[int]:
        """Convert events to indices."""
        indices = []
        for event in events:
            event_str = f"{event.event_type}_{event.pitch}_{event.velocity}"
            if event_str in self.event_to_idx:
                indices.append(self.event_to_idx[event_str])
        return indices
    
    def indices_to_events(self, indices: List[int]) -> List[MIDIEvent]:
        """Convert indices back to events."""
        events = []
        current_time = 0.0
        time_increment = 0.1  # 100ms between events
        
        for idx in indices:
            if idx in self.idx_to_event:
                event_str = self.idx_to_event[idx]
                note, on_off, pitch, velocity = event_str.split('_')
                event_type = 'note_on' if on_off == 'on' else 'note_off'

                events.append(MIDIEvent(
                    event_type,
                    int(pitch),
                    int(velocity),
                    current_time
                ))
                current_time += time_increment
        
        return events

class MIDISequenceDataset(Dataset):
    """Dataset for MIDI event sequences."""
    def __init__(self, sequences: List[List[int]], sequence_length: int):
        self.sequences = sequences
        self.sequence_length = sequence_length
        
    def __len__(self):
        return sum(len(seq) - self.sequence_length for seq in self.sequences)
    
    def __getitem__(self, idx):
        # Find which sequence this index belongs to
        sequence_idx = 0
        while idx >= len(self.sequences[sequence_idx]) - self.sequence_length:
            idx -= len(self.sequences[sequence_idx]) - self.sequence_length
            sequence_idx += 1
        
        sequence = self.sequences[sequence_idx]
        # Get input and target sequences
        input_seq = sequence[idx:idx + self.sequence_length]
        target_seq = sequence[idx + 1:idx + self.sequence_length + 1]
        
        return (
            torch.LongTensor(input_seq),
            torch.LongTensor(target_seq)
        )


class AttentionModule(nn.Module):
    """Multi-head self-attention module with fixed dimensions."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear transformations for Q, K, V
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_length = query.size(1)
        
        # Linear transformations and reshape
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Create proper attention mask if provided
        if mask is not None:
            # Expand mask for batch size and number of heads
            # mask shape: [batch_size, num_heads, seq_length, seq_length]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.expand(batch_size, self.num_heads, seq_length, seq_length)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        output = self.output_linear(context)
        
        return output, attention_weights
    


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for focusing on relevant past events."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, hidden_states, lstm_output):
        seq_len = lstm_output.size(1)
        hidden_expanded = hidden_states.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate hidden state with each lstm output
        combined = torch.cat((hidden_expanded, lstm_output), dim=2)
        attention_weights = F.softmax(self.attention(combined).squeeze(-1), dim=1)
        
        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attention_weights



class MIDISequenceLSTM(nn.Module):
    """LSTM model with fixed attention dimensions."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention modules
        self.self_attention = AttentionModule(
            hidden_dim * 2,  # * 2 for bidirectional
            num_heads,
            dropout
        )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for self-attention."""
        # Create mask that prevents attending to future tokens
        # Shape: [seq_len, seq_len]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return (~mask).to(device)
    
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        device = x.device
        
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded, hidden)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Self-attention
        attended_output, attention_weights = self.self_attention(
            lstm_out, lstm_out, lstm_out,
            mask=mask
        )
        
        # Residual connection and layer normalization
        output = self.layer_norm(attended_output + lstm_out)
        
        # Final output
        output = self.fc(self.dropout(output))
        
        return output, (hidden_state, cell_state), attention_weights


def create_midi_from_events(events: List[MIDIEvent], output_file: str):
    """Create a MIDI file from a sequence of events."""
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    active_notes = {}
    
    for event in events:
        if event.event_type == 'note_on' and event.velocity > 0:
            active_notes[event.pitch] = (event.time, event.velocity)
        elif event.event_type == 'note_off' or (event.event_type == 'note_on' and event.velocity == 0):
            if event.pitch in active_notes:
                start_time, velocity = active_notes[event.pitch]
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=event.pitch,
                    start=start_time,
                    end=event.time
                )
                piano.notes.append(note)
                del active_notes[event.pitch]
    
    midi.instruments.append(piano)
    midi.write(output_file)


class MIDISequenceTrainer:
    """Enhanced trainer with attention visualization capabilities."""
    def __init__(self, processor: EventSequenceProcessor, model_params: Dict):
        self.processor = processor
        self.model = MIDISequenceLSTM(
            vocab_size=processor.vocab_size,
            embedding_dim=model_params['embedding_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            num_heads=model_params.get('num_heads', 4),
            dropout=model_params.get('dropout', 0.1)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            # Set memory allocation to be more efficient
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

        self.model.to(self.device)
        
    def train(self, dataloader: DataLoader, num_epochs: int, learning_rate: float,
            accumulation_steps: int = 4, max_grad_norm: float = 1.0):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()  # Zero gradients at start of epoch
            
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                try:
                    input_seq = input_seq.to(self.device)
                    target_seq = target_seq.to(self.device)
                    
                    
                    output, _, attention_weights = self.model(input_seq)
                    
                    # Reshape output and target for loss calculation
                    output = output.view(-1, self.processor.vocab_size)
                    target_seq = target_seq.view(-1)
                    
                    loss = criterion(output, target_seq)/ accumulation_steps
                    loss.backward()

                    # Gradient accumulation
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                
                    total_loss += loss.item() * accumulation_steps

                    # Free up memory
                    del output
                    torch.cuda.empty_cache()


                    if batch_idx % 10 == 0:
                        print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

                except Exception as e:
                    print(f"Error during training: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    if "out of memory" in str(e):
                        print('| WARNING: ran out of memory, skipping batch')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                               
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}')
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Clean up old model file if it exists
                if os.path.exists('best_model.pth'):
                    os.remove('best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'best_model.pth')

            # Clear memory at end of epoch
            torch.cuda.empty_cache()
    
    def generate(self, seed_sequence: List[int], length: int, temperature: float = 1.0) -> Tuple[List[int], Dict]:
        """Generate a new sequence with attention information."""
        self.model.eval()
        attention_maps = []
        generated = list(seed_sequence)
        
        try:
            with torch.no_grad():
                # Process in smaller chunks
                chunk_size = min(50, length)  # Adjust chunk size based on available memory
                remaining_length = length
                
                while remaining_length > 0:
                    # Take the last sequence_length elements as context
                    context = generated[-len(seed_sequence):]
                    current_seq = torch.LongTensor([context]).to(self.device)
                    
                    # Generate next chunk
                    for _ in range(min(chunk_size, remaining_length)):
                        output, _, attn_weights = self.model(current_seq)
                        
                        # Temperature sampling
                        logits = output[0, -1, :] / temperature
                        probs = torch.softmax(logits, dim=0)
                        next_event = torch.multinomial(probs, 1).item()
                        
                        generated.append(next_event)
                        attention_maps.append({
                            'attention': attn_weights.cpu().numpy()
                        })
                        
                        # Update current sequence by sliding the window
                        # Take the last len(seed_sequence) elements including the new event
                        current_seq = torch.LongTensor([generated[-len(seed_sequence):]]).to(self.device)
        
                    
                    remaining_length -= chunk_size
                    
                    # Clear memory after each chunk
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print('| WARNING: ran out of memory during generation, returning partial sequence')
                torch.cuda.empty_cache()
            else:
                raise e

        
        return generated, attention_maps


def play_midi(midi_file_path):
    """
    Play a MIDI file using pygame
    
    Parameters:
        midi_file_path (str): Path to the MIDI file
    """
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load and play the MIDI file
        pygame.mixer.music.load(midi_file_path)
        pygame.mixer.music.play()
        
        # Keep the program running while the music plays
        while pygame.mixer.music.get_busy():
            time.sleep(1)
            
    except FileNotFoundError:
        print(f"Error: Could not find MIDI file at {midi_file_path}")
    except pygame.error as e:
        print(f"Error playing MIDI file: {e}")
    finally:
        # Clean up pygame resources
        pygame.mixer.music.stop()
        # pygame.midi.quit()
        pygame.mixer.quit()

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

def main():
    # Configuration
    mode = 'generate' # 'train' #  'generate'
    sequence_length =  50
    midi_folder = "./midi_dataset/piano_maestro-v1.0.0/all_years/"
    model_file = 'model.pth'
    processor_file = 'processor.pkl'
    
    # Model parameters
    model_params = {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    }

    # Training parameters
    train_params = {
        'batch_size': 8,      # Reduced batch size
        'accumulation_steps': 4,
        'num_epochs': 10,
        'learning_rate': 0.001
    }

    if mode == 'train':
        # Initialize processor
        processor = EventSequenceProcessor(max_events=500)
        
        # Process MIDI files
        midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + \
                    glob.glob(os.path.join(midi_folder, "*.midi"))
        print('len of midi_files', len(midi_files))
        
        # Extract events from all files
        all_sequences = []
        for midi_file in midi_files:
            try:
                events = processor.extract_events(midi_file)
                if events:
                    all_sequences.append(events)
                # Clear memory after processing each file
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing {midi_file}: {str(e)}")
                continue
        
        # Create vocabulary
        processor._create_event_vocab(all_sequences)
        
        # Convert events to indices
        indexed_sequences = [processor.events_to_indices(seq) for seq in all_sequences]
        
        # Create dataset
        dataset = MIDISequenceDataset(indexed_sequences, sequence_length)

        dataloader = DataLoader(
            dataset, 
            batch_size=train_params['batch_size'],
            shuffle=True,
            pin_memory=True  # Faster data transfer to GPU
        )

        
        # Create trainer and train model
        # Train model
        trainer = MIDISequenceTrainer(processor, model_params)
        trainer.train(
            dataloader,
            num_epochs=train_params['num_epochs'],
            learning_rate=train_params['learning_rate'],
            accumulation_steps=train_params['accumulation_steps']
        )
        
        # Save the model and processor
        torch.save(trainer.model.state_dict(), model_file)
        with open(processor_file, 'wb') as f:
            pickle.dump(processor, f)
        
        print(f"Model saved to {model_file}")
        print(f"Processor saved to {processor_file}")

    elif mode == 'generate':
        sequence_length_gen = 64
        # Load the processor
        try:
            with open(processor_file, 'rb') as f:
                processor = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Processor file {processor_file} not found.")
            print("Please run training mode first.")
            return

        # Create trainer with loaded processor
        trainer = MIDISequenceTrainer(processor, model_params)
        
        try:
            # Load the trained model
            trainer.model.load_state_dict(torch.load(model_file))
            trainer.model.eval()
        except FileNotFoundError:
            print(f"Error: Model file {model_file} not found.")
            print("Please run training mode first.")
            return
        
        print("Generating new sequence...")
        
        # Create a simple seed sequence
        # Option 1: Create a random seed
        # seed_sequence = [random.randint(0, processor.vocab_size-1) for _ in range(sequence_length_gen)]
        
        # Option 2: If you have a MIDI file to use as seed
        midi_seed_file = "./midi_dataset/mond_1.mid"
        seed_events = processor.extract_events(midi_seed_file)
        seed_sequence = processor.events_to_indices(seed_events)[:sequence_length_gen]
        
        # Generate new sequence
        try:
            generated_indices, attention_maps = trainer.generate(
                seed_sequence,
                length=200,
                temperature=2
            )

            # Save attention maps
            np.save('attention_maps.npy', attention_maps)
            
            # Convert back to events and create MIDI
            generated_events = processor.indices_to_events(generated_indices)
            output_midi = 'generated_sequence_attn.mid'
            create_midi_from_events(generated_events, output_midi)
            print("Generated sequence saved to 'generated_sequence_attn.mid'")
            print("Attention maps saved to 'attention_maps.npy'")

            # plt.figure(figsize=(8, 4))
            # plot_piano_roll(output_midi, 45, 90)
            # plt.show()

            play_midi(output_midi)
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
