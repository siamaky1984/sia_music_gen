
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

import math

from typing import List, Tuple, Set, Dict
from collections import defaultdict


import pygame
import librosa

from load_midi import plot_piano_roll

class ChordEvent:
    """Represents a set of simultaneous notes with timing."""
    def __init__(self, notes: Set[int], time: float, duration: float, velocity: int):
        self.notes = frozenset(notes)  # Immutable set of MIDI notes
        self.time = time
        self.duration = duration
        self.velocity = velocity
    
    def __repr__(self):
        return f"Chord(notes={self.notes}, time={self.time:.2f}, dur={self.duration:.2f})"


class HarmonicSequenceDataset(Dataset):
    """Dataset for harmonic sequences with sliding window."""
    def __init__(self, sequences: List[List[int]], sequence_length: int):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of sequences, where each sequence is a list of chord indices
            sequence_length: Length of sequences to generate (sliding window size)
        """
        self.sequences = sequences
        self.sequence_length = sequence_length
        
        # Pre-calculate valid indices for faster retrieval
        self.valid_indices = []
        for seq_idx, sequence in enumerate(sequences):
            # For each sequence, find all possible windows
            if len(sequence) >= sequence_length + 1:  # +1 for target
                for start_idx in range(len(sequence) - sequence_length):
                    self.valid_indices.append((seq_idx, start_idx))
    
    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a sequence and its target."""
        sequence_idx, start_idx = self.valid_indices[idx]
        sequence = self.sequences[sequence_idx]
        
        # Get input sequence
        input_seq = sequence[start_idx:start_idx + self.sequence_length]
        # Get target sequence (shifted by one position)
        target_seq = sequence[start_idx + 1:start_idx + self.sequence_length + 1]
        
        return (
            torch.LongTensor(input_seq),
            torch.LongTensor(target_seq)
        )
    
    def get_sequence_at_index(self, idx: int) -> Tuple[List[int], List[int]]:
        """Get the raw sequence and target at a specific index (for debugging)."""
        sequence_idx, start_idx = self.valid_indices[idx]
        sequence = self.sequences[sequence_idx]
        
        input_seq = sequence[start_idx:start_idx + self.sequence_length]
        target_seq = sequence[start_idx + 1:start_idx + self.sequence_length + 1]
        
        return input_seq, target_seq

    def get_random_seed_sequence(self) -> List[int]:
        """Get a random valid sequence to use as seed for generation."""
        if not self.valid_indices:
            raise ValueError("No valid sequences in dataset")
        
        # Choose a random sequence
        seq_idx, start_idx = random.choice(self.valid_indices)
        sequence = self.sequences[seq_idx]
        
        return sequence[start_idx:start_idx + self.sequence_length]

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        total_sequences = len(self.sequences)
        total_windows = len(self.valid_indices)
        sequence_lengths = [len(seq) for seq in self.sequences]
        
        return {
            'total_sequences': total_sequences,
            'total_windows': total_windows,
            'min_sequence_length': min(sequence_lengths),
            'max_sequence_length': max(sequence_lengths),
            'avg_sequence_length': sum(sequence_lengths) / total_sequences
        } 

    
class MultiHeadAttention(nn.Module):
    """Multi-head attention with proper mask handling."""
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_length = query.size(1)
        
        # Linear transformations
        Q = self.q_linear(query)  # (batch_size, seq_length, hidden_dim)
        K = self.k_linear(key)    # (batch_size, seq_length, hidden_dim)
        V = self.v_linear(value)  # (batch_size, seq_length, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Handle mask
        if mask is not None:
            # Adjust mask dimensions to match attention scores
            if mask.dim() == 2:
                # If mask is 2D (seq_length, seq_length), expand it
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                # If mask is 3D (batch_size, seq_length, seq_length), add head dimension
                mask = mask.unsqueeze(1)
            
            # Expand mask to match batch_size and num_heads
            mask = mask.expand(batch_size, self.num_heads, seq_length, seq_length)
            
            # Apply mask
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        output = self.output_linear(context)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal awareness."""
    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, 1, hidden_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0)]



class HarmonicEventProcessor:
    """Processes MIDI files into chord events."""
    def __init__(self, max_events=1000, time_threshold=0.05):
        self.max_events = max_events
        self.time_threshold = time_threshold
        self.chord_to_idx = {}
        self.idx_to_chord = {}
        self.vocab_size = 0
    
    def extract_chord_events(self, midi_file: str) -> List[ChordEvent]:
        """Extract chord events from a MIDI file."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            note_events = defaultdict(list)  # time -> list of (pitch, velocity, type)
            
            # Collect all note events
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        # Round times to handle slight timing discrepancies
                        start_time = round(note.start / self.time_threshold) * self.time_threshold
                        end_time = round(note.end / self.time_threshold) * self.time_threshold
                        
                        # Note onset
                        note_events[start_time].append((note.pitch, note.velocity, 'on'))
                        # Note offset
                        note_events[end_time].append((note.pitch, 0, 'off'))
            
            # Convert to chord events
            chord_events = []
            active_notes = set()
            current_velocity = {}
            last_time = 0
            
            for time in sorted(note_events.keys()):
                if active_notes and time > last_time:
                    # Create chord event for the current state
                    avg_velocity = int(np.mean(list(current_velocity.values())))
                    chord_events.append(ChordEvent(
                        notes=active_notes.copy(),
                        time=last_time,
                        duration=time - last_time,
                        velocity=avg_velocity
                    ))
                
                # Update active notes
                for pitch, vel, event_type in note_events[time]:
                    if event_type == 'on':
                        active_notes.add(pitch)
                        current_velocity[pitch] = vel
                    else:
                        active_notes.discard(pitch)
                        current_velocity.pop(pitch, None)
                
                last_time = time
            
            return chord_events[:self.max_events]
            
        except Exception as e:
            print(f"Error processing {midi_file}: {str(e)}")
            return []
    
    def _create_chord_vocab(self, all_sequences: List[List[ChordEvent]]):
        """Create vocabulary for all unique chords."""
        unique_chords = set()
        for sequence in all_sequences:
            for event in sequence:
                unique_chords.add(event.notes)
        
        self.chord_to_idx = {chord: idx for idx, chord in enumerate(sorted(unique_chords))}
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}
        self.vocab_size = len(self.chord_to_idx)
        print(f"Vocabulary size (unique chords): {self.vocab_size}")
    
    def events_to_indices(self, events: List[ChordEvent]) -> List[int]:
        """Convert chord events to indices."""
        return [self.chord_to_idx[event.notes] for event in events]
    
    def indices_to_events(self, indices: List[int], base_duration=0.25) -> List[ChordEvent]:
        """Convert indices back to chord events."""
        events = []
        current_time = 0.0
        
        for idx in indices:
            if idx in self.idx_to_chord:
                notes = self.idx_to_chord[idx]
                events.append(ChordEvent(
                    notes=notes,
                    time=current_time,
                    duration=base_duration,
                    velocity=80  # Default velocity
                ))
                current_time += base_duration
        
        return events


# class HarmonicSequenceLSTM(nn.Module):
#     """LSTM model for harmonic sequence generation."""
#     def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
#                  num_layers: int, dropout: float = 0.1):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(
#             embedding_dim, 
#             hidden_dim, 
#             num_layers,
#             batch_first=True,
#             dropout=dropout
#         )
#         self.fc = nn.Linear(hidden_dim, vocab_size)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x, hidden=None):
#         embedded = self.embedding(x)
#         lstm_out, hidden = self.lstm(embedded, hidden)
#         output = self.fc(self.dropout(lstm_out))
#         return output, hidden


class HarmonicSequenceLSTM(nn.Module):
    """Enhanced LSTM with multi-head attention for harmonic sequences."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int=3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,  # Explicitly set to 3
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention modules
        self.self_attention = MultiHeadAttention(
            hidden_dim*2,  # * 2 for bidirectional
            num_heads,
            dropout
        )
        
        # Additional layers
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        # self.fc = nn.Linear(hidden_dim , vocab_size)
    
    # def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
    #     """Create causal attention mask."""
    #     # Create mask to prevent attending to future tokens
    #     mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    #     mask = mask.unsqueeze(0).expand(self.num_layers, -1, -1)
    #     return (~mask).to(device)
    

    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = (~mask).to(device)  # True for valid attention positions
        return mask
    
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        device = x.device
        
        # Embedding and positional encoding
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Self-attention
        mask = self.create_attention_mask(seq_len, device)

        attended_out, attention_weights = self.self_attention(
            lstm_out, lstm_out, lstm_out,
            mask=mask
        )
        
        # Residual connection and layer normalization
        lstm_out = self.layer_norm1(lstm_out + attended_out)
        
        # Final output
        output = self.fc(self.dropout(lstm_out))
        
        return output, hidden, attention_weights


class HarmonicSequenceTrainer:
    """Handles training and generation of harmonic sequences."""
    def __init__(self, processor: HarmonicEventProcessor, model_params: Dict):
        self.processor = processor
        self.model = HarmonicSequenceLSTM(
            vocab_size=processor.vocab_size,
            embedding_dim=model_params['embedding_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            dropout=model_params.get('dropout', 0.1)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


        mask, attn_weights = debug_attention_mask(
        self.model,
        batch_size=32,
        seq_length=16
        )
    
    def train(self, dataloader: DataLoader, num_epochs: int, learning_rate: float,
              save_path: str ):
        """Train the harmonic sequence model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # Training loop
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                try:
                    # Move data to device
                    input_seq = input_seq.to(self.device)
                    target_seq = target_seq.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output, _ , attention_weights= self.model(input_seq)
                    
                    # Reshape output and target for loss calculation
                    output = output.view(-1, self.processor.vocab_size)
                    target_seq = target_seq.view(-1)
                    
                    # Calculate loss
                    loss = criterion(output, target_seq)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    optimizer.step()
                    
                    # Accumulate loss
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Print progress
                    if batch_idx % 10 == 0:
                        print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print('WARNING: out of memory, skipping batch')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}')
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path)
                print(f'Model saved to {save_path}')
            
            # Clear cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model on a validation set."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_seq, target_seq in dataloader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                output, _ = self.model(input_seq)
                output = output.view(-1, self.processor.vocab_size)
                target_seq = target_seq.view(-1)
                
                loss = criterion(output, target_seq)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    

def debug_attention_mask(model: HarmonicSequenceLSTM, batch_size: int = 32, seq_length: int = 16):
    """Debug function to check attention mask dimensions."""
    device = next(model.parameters()).device
    
    # Create sample input
    x = torch.randint(0, model.vocab_size, (batch_size, seq_length)).to(device)
    
    # Get mask
    mask = model.create_attention_mask(seq_length, device)
    
    print("Mask shape:", mask.shape)
    print("Sample mask values:\n", mask[0:2, 0:2])
    
    # Forward pass with dimension printing
    with torch.no_grad():
        output, hidden, attn_weights = model(x)
        
        print("\nDimensions:")
        print("Input shape:", x.shape)
        print("Output shape:", output.shape)
        print("Attention weights shape:", attn_weights.shape)
        
    return mask, attn_weights


def create_harmonic_datasets(processor: HarmonicEventProcessor, 
                           sequences: List[List[int]], 
                           sequence_length: int,
                           train_ratio: float = 0.8,
                           batch_size: int = 32):
    """Create training and validation datasets."""
    # Create dataset
    dataset = HarmonicSequenceDataset(sequences, sequence_length)
    
    # Split into train and validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


class InferenceManager:
    """Manages the inference process with flexible model loading."""
    def __init__(self, model_path: str, processor_path: str):
        self.model_path = model_path
        self.processor_path = processor_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load processor first
        self.processor = self.load_processor()
        
        # Then load model
        self.model = self.load_model()
    
    def load_processor(self):
        """Load the processor with vocabulary."""
        try:
            with open(self.processor_path, 'rb') as f:
                processor = pickle.load(f)
            print(f"Loaded processor with vocabulary size: {processor.vocab_size}")
            return processor
        except Exception as e:
            raise RuntimeError(f"Failed to load processor: {str(e)}")
    
    def load_model(self):
        """Load the model, inferring architecture from state dict if necessary."""
        if True: #try:
            # Load checkpoint
            # checkpoint = torch.load(self.model_path, map_location=self.device)

            checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True  # Safe loading
            )
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            else:
                state_dict = checkpoint
            
            # Infer model parameters from state dict
            model_params = self.infer_model_params(state_dict)
            print("Inferred model parameters:", model_params)
            
            # Initialize model with inferred parameters
            model = HarmonicSequenceLSTM(**model_params)
            
            # Load state dict
            # model.load_state_dict(state_dict)

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print("Warning: Missing keys in state dict:", missing_keys)
            if unexpected_keys:
                print("Warning: Unexpected keys in state dict:", unexpected_keys)

            model = model.to(self.device)
            model.eval()
            
            return model
            
        # except Exception as e:
        #     raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def infer_model_params(self, state_dict: Dict[str, torch.Tensor]) -> Dict:
        """Infer model parameters from state dictionary."""
        try:
            # Extract dimensions from embedding layer
            embedding_weight = state_dict['embedding.weight']
            vocab_size, embedding_dim = embedding_weight.shape
            
            # Extract hidden dimension from LSTM
            hidden_dim = state_dict['lstm.weight_hh_l0'].shape[1] # // 4
            
            # # Count number of LSTM layers
            # # num_layers = sum(1 for key in state_dict if 'lstm.weight_hh' in key)

            # Count LSTM layers by looking at the layer-specific weights
            max_layer_idx = -1
            for key in state_dict.keys():
                if 'lstm.weight_ih_l' in key and '_reverse' not in key:
                    layer_idx = int(key.split('lstm.weight_ih_l')[1][0])  # Extract the layer number
                    max_layer_idx = max(max_layer_idx, layer_idx)
            
            # Add 1 because layer indexing starts at 0
            num_layers = max_layer_idx + 1
            print('num_layers loaded is : ', num_layers)


            # Ensure vocab size matches processor
            if vocab_size != self.processor.vocab_size:
                print(f"Warning: Model vocab size ({vocab_size}) differs from processor vocab size ({self.processor.vocab_size})")
                vocab_size = self.processor.vocab_size
            
            model_params = {
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': 0.0  # Set to 0 for inference
            }
            
            return model_params
            
        except KeyError as e:
            raise ValueError(f"Could not infer model parameters from state dict: {str(e)}")
        


    def verify_model_loading(self, model, state_dict):
        """Verify that model loading was successful."""
        # Check that all parameters are loaded
        model_state = model.state_dict()
        loaded_params = state_dict.keys()
        model_params = model_state.keys()
        
        missing = [k for k in model_params if k not in loaded_params]
        unexpected = [k for k in loaded_params if k not in model_params]
        
        if missing or unexpected:
            print("\nModel loading verification:")
            if missing:
                print("Missing parameters:", missing)
            if unexpected:
                print("Unexpected parameters:", unexpected)
            
        return len(missing) == 0 and len(unexpected) == 0
    

    def safe_generate(self, sequence_length: int = 16, 
                     generation_length: int = 32,
                     temperature: float = 0.8) -> Tuple[List[int], List[ChordEvent]]:
        """Generate sequence with additional safety checks."""
        print(f"\nStarting safe generation with:")
        print(f"- Sequence length: {sequence_length}")
        print(f"- Generation length: {generation_length}")
        print(f"- Temperature: {temperature}")
        print(f"- Device: {self.device}")
        
        try:
            # Verify model is in eval mode
            self.model.eval()
            
            # Create seed sequence
            seed_sequence = [0] * sequence_length
            
            with torch.no_grad():
                current_seq = torch.LongTensor([seed_sequence]).to(self.device)
                generated = list(seed_sequence)
                hidden = None
                
                for i in range(generation_length):
                    try:
                        # Get model output
                        output, hidden, _ = self.model(current_seq, hidden)
                        
                        # Apply temperature
                        logits = output[0, -1, :self.processor.vocab_size] / temperature
                        
                        # Get probabilities
                        probs = torch.softmax(logits, dim=0)
                        
                        # Sample next token
                        next_event = torch.multinomial(probs, 1).item()
                        
                        # Verify valid index
                        if not 0 <= next_event < self.processor.vocab_size:
                            print(f"Warning: Invalid index {next_event}, using modulo")
                            next_event = next_event % self.processor.vocab_size
                        
                        generated.append(next_event)
                        
                        # Update input sequence
                        current_seq = torch.LongTensor([generated[-sequence_length:]]).to(self.device)
                        
                        if (i + 1) % 10 == 0:
                            print(f"Generated {i + 1}/{generation_length} events")
                            
                    except RuntimeError as e:
                        print(f"Error during generation step {i}: {str(e)}")
                        break
                
                # Convert to events
                try:
                    generated_events = self.processor.indices_to_events(generated)
                except Exception as e:
                    print(f"Error converting to events: {str(e)}")
                    return generated, []
                
                return generated, generated_events
                
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            return [], []

    
    def save_midi(self, events: List[ChordEvent], output_path: str):
        """Save generated events as MIDI file."""
        try:
            create_harmonic_midi(events, output_path)
            print(f"MIDI file saved to: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save MIDI: {str(e)}")


class MusicAttentionWrapper:
    """Wrapper for attention visualization and analysis."""
    def __init__(self, model: HarmonicSequenceLSTM):
        self.model = model
        self.attention_maps = []
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Get stored attention weights."""
        return self.attention_maps
    
    def clear_attention_maps(self):
        """Clear stored attention maps."""
        self.attention_maps = []
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor):
        """Analyze attention patterns in the generated sequence."""
        # Average attention weights across heads
        avg_attention = attention_weights.mean(dim=1)
        
        # Find strongest connections
        top_k = 3
        values, indices = torch.topk(avg_attention, top_k, dim=-1)
        
        return {
            'top_connections': indices.cpu().numpy(),
            'connection_strengths': values.cpu().numpy()
        }

def generate_with_attention_analysis(model: HarmonicSequenceLSTM,
                                   seed_sequence: List[int],
                                   length: int,
                                   temperature: float = 1.0) -> Tuple[List[int], List[Dict]]:
    """Generate sequence with attention analysis."""
    model.eval()
    attention_wrapper = MusicAttentionWrapper(model)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        current_seq = torch.LongTensor([seed_sequence]).to(device)
        generated = list(seed_sequence)
        hidden = None
        attention_analyses = []
        
        for _ in range(length):
            # Generate next token
            output, hidden, attn_weights = model(current_seq, hidden)
            
            # Apply temperature
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            
            # Sample next token
            next_event = torch.multinomial(probs, 1).item()
            generated.append(next_event)
            
            # Analyze attention patterns
            analysis = attention_wrapper.analyze_attention_patterns(attn_weights)
            attention_analyses.append(analysis)
            
            # Update sequence
            current_seq = torch.LongTensor([generated[-len(seed_sequence):]]).to(device)
    
    return generated, attention_analyses

def create_harmonic_midi(events: List[ChordEvent], output_file: str):
    """Create a MIDI file from chord events."""
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    for event in events:
        # Create a note for each pitch in the chord
        for pitch in event.notes:
            note = pretty_midi.Note(
                velocity=event.velocity,
                pitch=int(pitch),
                start=event.time,
                end=event.time + event.duration
            )
            piano.notes.append(note)
    
    midi.instruments.append(piano)
    midi.write(output_file)

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


def main():

    mode = 'train' #  'generate'
    # sequence_length =  50
    midi_folder = "./midi_dataset/piano_maestro-v1.0.0/2004/" #all_years/"
    model_file = 'harmonic_model_attention.pth'
    processor_file = 'processor.pkl'
    
    # Model parameters
    model_params = {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 3,
        'num_heads': 8,
        'dropout': 0.1
    }

    # Training parameters
    train_params = {
        'batch_size': 8,      # Reduced batch size
        'sequence_length': 16,
        'num_epochs': 10,
        'learning_rate': 0.001
    }
    
    # Initialize processor
    processor = HarmonicEventProcessor(time_threshold=0.1)


    if mode == 'train':

        # Process MIDI files
        midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + \
                    glob.glob(os.path.join(midi_folder, "*.midi"))
        print('len of midi_files', len(midi_files))
        
        # Extract events from all files
        all_sequences = []
        for midi_file in midi_files:
            try:
                events = processor.extract_chord_events(midi_file)
                # print(events)
                if events:
                    all_sequences.append(events)
                # Clear memory after processing each file
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing {midi_file}: {str(e)}")
                continue
        
        # Create vocabulary
        processor._create_chord_vocab(all_sequences)
        
        # Convert events to indices
        indexed_sequences = [processor.events_to_indices(seq) for seq in all_sequences]
        
        # # Create dataset
        # dataset = MIDISequenceDataset(indexed_sequences, sequence_length)

        # dataloader = DataLoader(
        #     dataset, 
        #     batch_size=train_params['batch_size'],
        #     shuffle=True,
        #     pin_memory=True  # Faster data transfer to GPU
        # )

        # Create datasets
        train_loader, val_loader = create_harmonic_datasets(
        processor,
        indexed_sequences,
        sequence_length=train_params['sequence_length'],
        batch_size=train_params['batch_size']
        )

        
        # Create trainer and train model
        # Train model
        trainer = HarmonicSequenceTrainer(processor, model_params)
        trainer.train(
            train_loader,
            num_epochs=train_params['num_epochs'],
            learning_rate=train_params['learning_rate'],
            save_path = model_file
        )
        
        # Save the model and processor
        # torch.save(trainer.model.state_dict(), model_file)
        
        with open(processor_file, 'wb') as f:
            pickle.dump(processor, f)
        
        print(f"Model saved to {model_file}")
        print(f"Processor saved to {processor_file}")


    elif mode == 'generate':

        """Example usage of inference manager."""
        # Paths to saved files
        model_path = model_file
        processor_path = processor_file
        output_path = 'generated_sequence.mid'
        
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(processor_path):
            raise FileNotFoundError(f"Processor file not found: {processor_path}")
        
        try:
            # Initialize inference manager
            print("Initializing inference manager...")
            inference_manager = InferenceManager(model_path, processor_path)
            
            generated_indices, generated_events = inference_manager.safe_generate(
            sequence_length=16,
            generation_length=32,
            temperature=0.8
            )

            # # Generate with attention analysis
            # generated_indices, attention_analyses = generate_with_attention_analysis(
            #     model=inference_manager.model,
            #     seed_sequence=[0] * 16,  # Example seed
            #     length=32,
            #     temperature=0.8
            # )
            
            # # Convert to events
            # generated_events = inference_manager.processor.indices_to_events(generated_indices)
            
            # # Print attention analysis
            # print("\nAttention Analysis:")
            # for i, analysis in enumerate(attention_analyses):
            #     print(f"\nStep {i+1}:")
            #     print("Top connections:", analysis['top_connections'])
            #     print("Connection strengths:", analysis['connection_strengths'])


            # Save to MIDI
            print("Saving to MIDI...")
            inference_manager.save_midi(generated_events, output_path)
            
            print("\nGeneration Statistics:")
            print(f"Total events generated: {len(generated_events)}")
            print(f"Unique chords used: {len(set([event.notes for event in generated_events]))}")

            play_midi(output_path)
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        print("\nInference completed successfully!")

if __name__ == "__main__":
    main()