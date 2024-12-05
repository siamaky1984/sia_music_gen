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
import argparse

from typing import List, Tuple, Set, Dict
import music21  # For music theory analysis

from typing import List, Tuple, Set, Dict
from collections import defaultdict


# from chord_aware_LSTM import ChordAwareAttention, ChordEvent, HarmonicEventProcessor, HarmonicSequenceDataset, HarmonicSequenceTrainer, prepare_training_data, play_midi

# from chord_aware_LSTM import ChordAwareAttention

class ChordEvent:
    """Represents a chord event with timing."""
    def __init__(self, notes: Set[int], time: float, duration: float, velocity: int):
        self.notes = frozenset(notes)
        self.time = time
        self.duration = duration
        self.velocity = velocity
        # self.bass_note = min(notes) if notes else 0

        # Separate into voices
        self.bass_notes = self._get_bass_notes(notes)
        self.upper_notes = self._get_upper_notes(notes)

    def _get_bass_notes(self, notes: Set[int]) -> Set[int]:
        """Extract bass notes (usually below middle C - MIDI note 60)."""
        return {note for note in notes if note < 60}
    
    def _get_upper_notes(self, notes: Set[int]) -> Set[int]:
        """Extract upper notes (usually above or at middle C)."""
        return {note for note in notes if note >= 60}
    
    @property
    def has_both_hands(self) -> bool:
        """Check if event has notes for both hands."""
        return bool(self.treble_notes and self.bass_notes)

class HarmonicEventProcessor:
    """Processes MIDI files into chord events."""
    def __init__(self, time_threshold: float = 0.25):
        self.time_threshold = time_threshold
        self.chord_to_idx = {}
        self.idx_to_chord = {}
        self.vocab_size = 0
    
    def extract_chord_events(self, midi_file: str) -> List[ChordEvent]:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            events = []
            
            # Separate notes by register
            bass_notes = defaultdict(list)
            upper_notes = defaultdict(list)

            # Group notes by time
            note_groups = defaultdict(set)
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        start_time = round(note.start / self.time_threshold) * self.time_threshold
                        # note_groups[start_time].add(note.pitch)

                        # Separate based on pitch
                        if note.pitch < 60:  # Below middle C
                            bass_notes[start_time].append(note)
                        else:
                            upper_notes[start_time].append(note)

            
            # Combine notes into chord events
            all_times = sorted(set(bass_notes.keys()) | set(upper_notes.keys()))
            
            for time in all_times:
                # Get current notes from both registers
                current_bass = {note.pitch for note in bass_notes[time]}
                current_upper = {note.pitch for note in upper_notes[time]}
                
                # Combine into chord event
                if current_bass or current_upper:
                    event = ChordEvent(
                        notes=current_bass | current_upper,
                        time=time,
                        duration=self.time_threshold,
                        velocity=80
                    )
                    events.append(event)


            # # Create chord events
            # for time, notes in sorted(note_groups.items()):
            #     if notes:
            #         event = ChordEvent(
            #             notes=notes,
            #             time=time,
            #             duration=self.time_threshold,
            #             velocity=80
            #         )
            #         events.append(event)
            
            return events
        except Exception as e:
            print(f"Error processing {midi_file}: {str(e)}")
            return []
    
    def create_vocabulary(self, all_sequences: List[List[ChordEvent]]):
        unique_chords = set()
        for sequence in all_sequences:
            for event in sequence:
                unique_chords.add(event.notes)
        
        self.chord_to_idx = {chord: idx for idx, chord in enumerate(sorted(unique_chords))}
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}
        self.vocab_size = len(self.chord_to_idx)
        print(f"Created vocabulary with {self.vocab_size} unique chords")
    
    def events_to_indices(self, events: List[ChordEvent]) -> List[int]:
        return [self.chord_to_idx[event.notes] for event in events]
    
    def indices_to_events(self, indices: List[int], base_duration: float = 0.25) -> List[ChordEvent]:
        events = []
        current_time = 0.0
        
        for idx in indices:
            if idx in self.idx_to_chord:
                notes = self.idx_to_chord[idx]
                events.append(ChordEvent(
                    notes=notes,
                    time=current_time,
                    duration=base_duration,
                    velocity=80
                ))
                current_time += base_duration
        
        return events

class HarmonicSequenceDataset(Dataset):
    """Dataset for harmonic sequences."""
    def __init__(self, sequences: List[List[int]], sequence_length: int):
        self.sequences = sequences
        self.sequence_length = sequence_length
        
        self.valid_indices = []
        for seq_idx, sequence in enumerate(sequences):
            if len(sequence) >= sequence_length + 1:
                for start_idx in range(len(sequence) - sequence_length):
                    self.valid_indices.append((seq_idx, start_idx))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        seq_idx, start_idx = self.valid_indices[idx]
        sequence = self.sequences[seq_idx]
        
        input_seq = sequence[start_idx:start_idx + self.sequence_length]
        target_seq = sequence[start_idx + 1:start_idx + self.sequence_length + 1]

        # Verify lengths
        assert len(input_seq) == self.sequence_length, f"Input sequence length {len(input_seq)} != {self.sequence_length}"
        assert len(target_seq) == self.sequence_length, f"Target sequence length {len(target_seq)} != {self.sequence_length}"
        
        return torch.LongTensor(input_seq), torch.LongTensor(target_seq)


class ChordAwareAttention(nn.Module):
    """Fixed multi-head attention with chord awareness."""
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query projections for different musical aspects
        self.pitch_query = nn.Linear(hidden_dim, hidden_dim)
        self.harmony_query = nn.Linear(hidden_dim, hidden_dim)
        self.voice_leading_query = nn.Linear(hidden_dim, hidden_dim)
        
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Feature projections
        self.chord_proj = nn.Linear(12, hidden_dim)
        self.bass_proj = nn.Linear(12, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def get_chord_features(self, event: ChordEvent, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract chord features for a single event."""
        # Pitch class vector
        pitch_classes = torch.zeros(12, device=device)
        for note in event.notes:
            pitch_classes[note % 12] = 1
        
        # Bass note vector
        bass_note = min(event.notes) if event.notes else 0
        bass_vector = torch.zeros(12, device=device)
        bass_vector[bass_note % 12] = 1
        
        return pitch_classes, bass_vector


    def forward(self, query, key, value, chord_events, mask=None,
            is_decoder_self_attn=False, is_cross_attention=False):
        """
        Forward pass with proper dimension handling.
        
        Args:
            query: [batch_size, query_len, hidden_dim]
            key: [batch_size, key_len, hidden_dim]
            value: [batch_size, key_len, hidden_dim]
            chord_events: List of ChordEvent
            mask: Optional attention mask
            is_decoder_self_attn: Flag for decoder self-attention
            is_cross_attention: Flag for cross-attention
        """
        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()
        device = query.device
        
        # Process queries
        pitch_q = self.pitch_query(query)
        harmony_q = self.harmony_query(query)
        voice_leading_q = self.voice_leading_query(query)
        
        k = self.key(key)
        v = self.value(value)
        
        # Determine feature length based on attention type
        feature_len = key_len if is_cross_attention else query_len
        
        # Process chord features
        chord_features = torch.zeros(feature_len, self.hidden_dim, device=device)
        bass_features = torch.zeros(feature_len, self.hidden_dim, device=device)
        
        # Ensure we have enough chord events
        if len(chord_events) < feature_len:
            chord_events = chord_events + [chord_events[-1]] * (feature_len - len(chord_events)) \
                if chord_events else [ChordEvent(set(), 0.0, 0.25, 80)] * feature_len
        else:
            chord_events = chord_events[:feature_len]
        
        # Extract features
        for i in range(feature_len):
            pitch_classes, bass_vector = self.get_chord_features(chord_events[i], device)
            chord_features[i] = self.chord_proj(pitch_classes)
            bass_features[i] = self.bass_proj(bass_vector)
        
        # Expand for batch dimension
        chord_features = chord_features.unsqueeze(0).expand(batch_size, -1, -1)
        bass_features = bass_features.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape for multi-head attention
        def reshape_for_heads(x, seq_length):
            return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape all tensors
        pitch_q = reshape_for_heads(pitch_q, query_len)
        harmony_q = reshape_for_heads(harmony_q, query_len)
        voice_q = reshape_for_heads(voice_leading_q, query_len)
        k = reshape_for_heads(k, key_len)
        v = reshape_for_heads(v, key_len)
        chord_features = reshape_for_heads(chord_features, feature_len)
        bass_features = reshape_for_heads(bass_features, feature_len)
        
        # Calculate attention scores
        scale = self.head_dim ** -0.5
        pitch_scores = torch.matmul(pitch_q, k.transpose(-2, -1))
        harmony_scores = torch.matmul(harmony_q, chord_features.transpose(-2, -1))
        voice_scores = torch.matmul(voice_q, bass_features.transpose(-2, -1))
        scores = (pitch_scores + harmony_scores + voice_scores) * scale / 3.0
        
        # Handle mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
            # scores = scores.masked_fill(mask == 0, float('-inf'))
            scores = scores.masked_fill(mask == 0, -1e9)  # Changed from float('-inf')
        
        # Calculate attention and apply to values
        
        # attention_weights = F.softmax(scores, dim=-1)

        # Use stable softmax
        attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attention_weights = attention_weights.type_as(scores)


        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        
        # Reshape output
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        output = self.output_proj(context)
        
        return output, attention_weights
    


    def shape_check(self, x: torch.Tensor, name: str):
        """Debug helper to check tensor shapes."""
        print(f"{name} shape: {x.shape}")

    
    # def _generate_subsequent_mask(self, sz: int) -> torch.Tensor:
    #     """Generate causal mask for self-attention."""
    #     mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    #     mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    #     return mask



class ChordAwareTransformer(nn.Module):
    """Transformer model with chord-aware attention for music generation."""
    def __init__(self, 
                 vocab_size: int,
                 max_seq_length: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        # Add checks for valid parameters
        assert vocab_size > 0, f"Invalid vocab_size: {vocab_size}"
        assert d_model > 0, f"Invalid d_model: {d_model}"
        
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)


        # Initialize embedding with proper scaling
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=None,  # Set this if you have a padding token
            max_norm=1.0  # Add max norm constraint
        )
        
        # Initialize embedding weights properly
        nn.init.xavier_uniform_(self.embedding.weight)



        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Create chord-aware encoder layer
        # Create encoder layers with chord awareness
        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_layer = ChordAwareEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            encoder_layers.append(encoder_layer)


        # Create decoder layers with chord awareness
        decoder_layers = []
        for _ in range(num_decoder_layers):
            decoder_layer = ChordAwareDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            decoder_layers.append(decoder_layer)


        # Create ModuleList for layers
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, chord_events, src_mask=None, tgt_mask=None):
        """
        Forward pass with chord-aware attention.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            chord_events: List of chord events
            src_mask: Source mask for attention
            tgt_mask: Target mask for attention
        """
        
        if torch.any(src >= self.embedding.num_embeddings):
            invalid_indices = torch.nonzero(src >= self.embedding.num_embeddings)
            raise ValueError(f"Source contains invalid indices: {invalid_indices}")
    


        # Source embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model) * 10
        src = self.pos_encoder(src)
        
        # Target embedding and positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model) * 10
        tgt = self.pos_encoder(tgt)
        
        # Create masks if not provided
        if src_mask is None:
            # src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
            src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
        if tgt_mask is None:
            # tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
           
        # Encoder
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask, chord_events)
        enc_output = self.norm(enc_output)
        
        # Decoder
        dec_output = tgt
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask, chord_events)
        dec_output = self.norm(dec_output)
        
        # Final linear layer
        output = self.output_layer(dec_output)
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for self-attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = (~mask).float()
        return mask


class ChordAwareEncoderLayer(nn.Module):
    """Transformer encoder layer with chord-aware self-attention."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.chord_attention = ChordAwareAttention(
            hidden_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        # Initialize feed forward layers properly
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.02)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.02)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
    def forward(self, src, src_mask, chord_events):
        # Chord-aware self-attention
        src2, attention_weights = self.chord_attention(
            query=src,
            key=src,
            value=src,
            chord_events=chord_events,
            mask=src_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class ChordAwareDecoderLayer(nn.Module):
    """Transformer decoder layer with chord-aware attention."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention with chord awareness
        self.self_attn = ChordAwareAttention(
            hidden_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Cross-attention with chord awareness
        self.multihead_attn = ChordAwareAttention(
            hidden_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
    def forward(self, tgt, memory, tgt_mask, chord_events):
        # Get sequence lengths
        batch_size, tgt_len, _ = tgt.size()
        src_len = memory.size(1)
        
        # # Print shapes for debugging
        # print(f"Target shape: {tgt.shape}")
        # print(f"Memory shape: {memory.shape}")
        # print(f"Target mask shape: {tgt_mask.shape}" )
              
        # Self-attention
        tgt2, self_attention_weights = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            chord_events=chord_events,
            mask=tgt_mask,
            is_decoder_self_attn=True  # Flag to indicate this is decoder self-attention
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2, cross_attention_weights = self.multihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            chord_events=chord_events,
            is_cross_attention=True  # Flag to indicate this is cross-attention
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class TransformerSystem:
    """Complete system for training and inference."""
    def __init__(self, model_params: Dict):
        self.model_params = model_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = HarmonicEventProcessor()
        self.model = None
    
    def train(self, midi_folder: str, num_epochs: int, batch_size: int, learning_rate: float):
        """Train the model on MIDI files."""
        print("Processing MIDI files...")
        
        # Process MIDI files
        midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + \
                glob.glob(os.path.join(midi_folder, "*.midi"))
        all_sequences = []
        
        for midi_file in midi_files:
            events = self.processor.extract_chord_events(midi_file)
            if events:
                all_sequences.append(events)
        
        # Create vocabulary
        self.processor.create_vocabulary(all_sequences)
        
        # Update model parameters with vocabulary size
        self.model_params['vocab_size'] = self.processor.vocab_size
        
        # Initialize model
        self.model = ChordAwareTransformer(**self.model_params).to(self.device)
        
        # Convert to indices
        indexed_sequences = [self.processor.events_to_indices(seq) for seq in all_sequences]
        
        # Create dataset and dataloader
        dataset = HarmonicSequenceDataset(indexed_sequences, self.model_params['max_seq_length'])
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True, 
                                drop_last=True  # Prevent partial batches 
                                ) 
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print("Starting training...")
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (src_seq, tgt_seq) in enumerate(dataloader):
                
                batch_size, seq_len = src_seq.size()
                assert seq_len == self.model_params['max_seq_length'], \
                    f"Input sequence length {seq_len} != {self.model_params['max_seq_length']}"

                src_seq = src_seq.to(self.device)
                tgt_seq = tgt_seq.to(self.device)
                
                # Create input and target sequences
                tgt_input = tgt_seq[:, :-1]
                tgt_output = tgt_seq[:, 1:]
                
                # Convert indices to chord events
                chord_events = self.processor.indices_to_events(src_seq[0].cpu().tolist())
                
                # Forward pass
                output = self.model(
                    src=src_seq,
                    tgt=tgt_input,
                    chord_events=chord_events
                )
                
                # Calculate loss
                loss = criterion(
                    output.view(-1, self.processor.vocab_size),
                    tgt_output.contiguous().view(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}')
            
            # Save checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, optimizer, loss)
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer: optim.Optimizer, loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_params': self.model_params,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)
        
        # Save processor separately
        with open('processor.pkl', 'wb') as f:
            pickle.dump(self.processor, f)
    
    def load_checkpoint(self, model_path: str, processor_path: str):
        """Load model checkpoint."""
        # Load processor
        print(processor_path)
        with open(processor_path, 'rb') as f:
            self.processor = pickle.load(f)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_params = checkpoint['model_params']
        
        # Initialize model
        self.model = ChordAwareTransformer(**self.model_params).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


    def generate(self, seed_sequence = None,
                length: int = 32,
                temperature: float = 1.0,
                # min_bass_note: int = 36,  # C2
                # max_bass_note: int = 59,  # B3
                # min_upper_note: int = 60,  # C4
                # max_upper_note: int = 84,  # C6
                ) -> List[ChordEvent]:
        """Generate with voice range constraints."""
        if self.model is None:
            raise ValueError("Model not initialized. Please load a checkpoint first.")
        
        # self.model.eval()
        
        with torch.no_grad():
            # Initialize seed sequence
            if seed_sequence is None:
                seed_sequence = [0] * 4
            
            src = torch.LongTensor([seed_sequence]).to(self.device)

            tgt = torch.LongTensor([[seed_sequence[0]]]).to(self.device)

            generated = list(seed_sequence)
            
            for i in range(length):
                # Convert to chord events
                chord_events = self.processor.indices_to_events(generated[-len(seed_sequence):])
                
                # Get model output
                output = self.model(src, tgt, chord_events)
                
                # Apply temperature
                logits = output[0, -1, :] / temperature
                
                # Split logits based on register
                bass_indices = []
                upper_indices = []
                
                # Find valid indices for each register
                for idx, chord in enumerate(self.processor.idx_to_chord.items()):
                    notes = chord[1]
                    # if any(min_bass_note <= note <= max_bass_note for note in notes):
                    #     bass_indices.append(idx)
                    # if any(min_upper_note <= note <= max_upper_note for note in notes):
                    #     upper_indices.append(idx)
                    if any(note < 60 for note in notes):
                        bass_indices.append(idx)
                    else:
                        upper_indices.append(idx)
                
                # Sample from appropriate register based on position
                if i % 2 == 0:  # Alternate between bass and upper voices
                    valid_indices = bass_indices
                else:
                    valid_indices = upper_indices
                
                # Mask invalid indices
                mask = torch.ones_like(logits) * float('-1e9')
                mask[valid_indices] = 0
                logits = logits + mask
                
                # Sample
                probs = F.softmax(logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                src = torch.LongTensor([generated[-len(seed_sequence):]]).to(self.device)
            
            return self.processor.indices_to_events(generated)
        
    # def generate(self, 
    #             seed_sequence: List[int] = None,
    #             length: int = 32,
    #             temperature: float = 1.0,
    #             top_k: int = 50,
    #             top_p: float = 0.9) -> List[ChordEvent]:
    #     """Generate new music sequence."""
    #     if self.model is None:
    #         raise ValueError("Model not initialized. Please load a checkpoint first.")
        
    #     self.model.eval()
    #     sequence_length = self.model_params['max_seq_length']
        
    #     with torch.no_grad():
    #         # Create seed sequence if not provided
    #         if seed_sequence is None:
    #             seed_sequence = [0] * sequence_length
    #         else:
    #             # Pad or trim seed sequence to match sequence length
    #             if len(seed_sequence) < sequence_length:
    #                 seed_sequence = seed_sequence + [0] * (sequence_length - len(seed_sequence))
    #             else:
    #                 seed_sequence = seed_sequence[:sequence_length]
            
    #         src = torch.LongTensor([seed_sequence]).to(self.device)
    #         tgt = torch.LongTensor([[seed_sequence[0]]]).to(self.device)
            
    #         generated = list(seed_sequence)
    #         print(f"Initial sequence length: {len(generated)}")
            
    #         for i in range(length):
    #             context_sequence = generated[-sequence_length:]
    #             if len(context_sequence) < sequence_length:
    #                 context_sequence = [0] * (sequence_length - len(context_sequence)) + context_sequence
                
    #             # Convert to chord events
    #             chord_events = self.processor.indices_to_events(generated)
                
    #             # Get model output
    #             output = self.model(src, tgt, chord_events)
                
    #             # Sample next token
    #             logits = output[0, -1, :] / temperature
                
    #             # Apply top-k filtering
    #             if top_k > 0:
    #                 indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    #                 logits[indices_to_remove] = float('-inf')
                
    #             # Apply top-p filtering
    #             if top_p < 1.0:
    #                 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #                 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
    #                 sorted_indices_to_remove = cumulative_probs > top_p
    #                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #                 sorted_indices_to_remove[..., 0] = 0
                    
    #                 indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #                 logits[indices_to_remove] = float('-inf')
                
    #             # Sample
    #             probs = F.softmax(logits, dim=-1)
    #             next_token = torch.multinomial(probs, 1).item()
                
    #             generated.append(next_token)
    #             tgt = torch.cat([tgt, torch.LongTensor([[next_token]]).to(self.device)], dim=1)

    #             # Optional: Keep tgt sequence length manageable
    #             if tgt.size(1) > sequence_length:
    #                 tgt = tgt[:, -sequence_length:]
                
    #             if (i + 1) % 10 == 0:
    #                 print(f"Generated {i + 1}/{length} events")
            
    #         return self.processor.indices_to_events(generated)
        

    
    # def save_midi(self, events: List[ChordEvent], output_path: str):
    #     """Save generated events as MIDI file."""
    #     midi = pretty_midi.PrettyMIDI()
    #     piano = pretty_midi.Instrument(program=0)
        
    #     for event in events:
    #         for pitch in event.notes:
    #             note = pretty_midi.Note(
    #                 velocity=event.velocity,
    #                 pitch=int(pitch),
    #                 start=event.time,
    #                 end=event.time + event.duration
    #             )
    #             piano.notes.append(note)
        
    #     midi.instruments.append(piano)
    #     midi.write(output_path)
    #     print(f"MIDI file saved to: {output_path}")



    # def create_harmonic_midi(self,events: List[ChordEvent], output_file: str):
    #     """Enhanced MIDI creation with proper voice separation."""
    #     midi = pretty_midi.PrettyMIDI()
        
    #     # Create two tracks for different registers
    #     bass_track = pretty_midi.Instrument(program=0)  # Piano
    #     upper_track = pretty_midi.Instrument(program=0)  # Piano
        
    #     # Process events
    #     for event in events:
    #         # Add bass notes
    #         for pitch in event.bass_notes:
    #             note = pretty_midi.Note(
    #                 velocity=event.velocity,
    #                 pitch=int(pitch),
    #                 start=event.time,
    #                 end=event.time + event.duration
    #             )
    #             bass_track.notes.append(note)
            
    #         # Add upper notes
    #         for pitch in event.upper_notes:
    #             note = pretty_midi.Note(
    #                 velocity=event.velocity,
    #                 pitch=int(pitch),
    #                 start=event.time,
    #                 end=event.time + event.duration
    #             )
    #             upper_track.notes.append(note)
        
    #     # Add tracks to MIDI
    #     midi.instruments.append(bass_track)
    #     midi.instruments.append(upper_track)
        
    #     # Sort notes in each track
    #     for track in midi.instruments:
    #         track.notes.sort(key=lambda x: (x.start, x.pitch))
        
    #     midi.write(output_file)
    #     print(f"Harmonic MIDI file saved to: {output_file}")

    def create_harmonic_midi(self, events: List[ChordEvent], output_file: str):
        """Create MIDI file with proper piano staff layout."""
        midi = pretty_midi.PrettyMIDI()
        
        # Create single piano instrument
        piano = pretty_midi.Instrument(
            program=0,  # Acoustic Grand Piano
            is_drum=False,
            name='Piano'  # Single piano instrument
        )
        
        # Process events
        for event in events:
            # Add all notes to the same piano track
            for pitch in event.notes:
                note = pretty_midi.Note(
                    velocity=event.velocity if event.velocity else 80,
                    pitch=int(pitch),
                    start=event.time,
                    end=event.time + event.duration
                )
                piano.notes.append(note)
        
        # Sort notes by time and pitch
        piano.notes.sort(key=lambda x: (x.start, x.pitch))
        
        # Add single piano track
        midi.instruments.append(piano)
        
        # # Add time signature and tempo
        # ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0)
        # midi.time_signatures.append(ts)
        
        # tempo = pretty_midi.Tempo(tempo=120, time=0)
        # midi.tempo_changes.append(tempo)
        
        # Write MIDI file
        midi.write(output_file)

    # def create_harmonic_midi(self, events: List[ChordEvent], output_file: str):
    #     """Create MIDI with proper staff separation for MuseScore."""
    #     midi = pretty_midi.PrettyMIDI()
        
    #     # Create two separate piano tracks for left and right hand
    #     # Track 0: Right hand (treble clef)
    #     right_hand = pretty_midi.Instrument(
    #         program=0,  # Acoustic Grand Piano
    #         name='Right Hand',
    #         is_drum=False
    #     )
        
    #     # Track 1: Left hand (bass clef)
    #     left_hand = pretty_midi.Instrument(
    #         program=0,  # Acoustic Grand Piano
    #         name='Left Hand',
    #         is_drum=False
    #     )
        
    #     # Process events with clear staff separation
    #     for event in events:
    #         # Add lower register notes to left hand (bass clef)
    #         bass_notes = sorted(n for n in event.notes if n < 60)  # Notes below middle C
    #         for pitch in bass_notes:
    #             note = pretty_midi.Note(
    #                 velocity=event.velocity if event.velocity else 80,
    #                 pitch=int(pitch),
    #                 start=event.time,
    #                 end=event.time + event.duration,
    #             )
    #             left_hand.notes.append(note)
            
    #         # Add higher register notes to right hand (treble clef)
    #         treble_notes = sorted(n for n in event.notes if n >= 60)  # Notes from middle C up
    #         for pitch in treble_notes:
    #             note = pretty_midi.Note(
    #                 velocity=event.velocity if event.velocity else 80,
    #                 pitch=int(pitch),
    #                 start=event.time,
    #                 end=event.time + event.duration,
    #             )
    #             right_hand.notes.append(note)
        
    #     # Ensure each track has at least one note (to show both staves)
    #     if not right_hand.notes:
    #         # Add a silent note to show treble clef
    #         right_hand.notes.append(
    #             pretty_midi.Note(velocity=0, pitch=72, start=0, end=0.1)
    #         )
    #     if not left_hand.notes:
    #         # Add a silent note to show bass clef
    #         left_hand.notes.append(
    #             pretty_midi.Note(velocity=0, pitch=48, start=0, end=0.1)
    #         )
        
    #     # Sort notes in each track by time and pitch
    #     right_hand.notes.sort(key=lambda x: (x.start, x.pitch))
    #     left_hand.notes.sort(key=lambda x: (x.start, x.pitch))
        
    #     # Add tracks to MIDI file in correct order
        
    #     midi.instruments.append(left_hand)   # Track 1
    #     midi.instruments.append(right_hand)  # Track 0

    #     # # Add time signature and tempo
    #     # ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=0)
    #     # # midi.time_signatures.append(ts)
        
    #     # tempo = pretty_midi.Tempo(tempo=120, time=0)  # 120 BPM
    #     # # midi.tempo_changes.append(tempo)
        
    #     # Write the MIDI file
    #     midi.write(output_file)



def validate_voice_separation(events: List[ChordEvent]):
    """Helper function to validate voice separation in generated sequence."""
    print("\nVoice Separation Analysis:")
    
    bass_notes = []
    upper_notes = []
    
    for event in events:
        if event.bass_notes:
            bass_notes.extend(event.bass_notes)
        if event.upper_notes:
            upper_notes.extend(event.upper_notes)
    
    print(f"Bass notes range: [{min(bass_notes) if bass_notes else 'N/A'}, "
          f"{max(bass_notes) if bass_notes else 'N/A'}]")
    print(f"Upper notes range: [{min(upper_notes) if upper_notes else 'N/A'}, "
          f"{max(upper_notes) if upper_notes else 'N/A'}]")
    
    # Check voice crossing
    crossings = 0
    for i in range(len(events)-1):
        if events[i].bass_notes and events[i+1].bass_notes:
            if max(events[i].bass_notes) > min(events[i+1].upper_notes):
                crossings += 1
    
    print(f"Voice crossings detected: {crossings}")


def main():
    """Example usage of the complete system."""

    # Model parameters
    model_params = {
        'max_seq_length': 64,
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 1024,
        'dropout': 0.1
    }
    
    # Initialize system
    system = TransformerSystem(model_params)

    print(args.midi_folder)
    
    # Training
    if args.mode == 'train':
        system.train(
            midi_folder= args.midi_folder,
            num_epochs=5,
            batch_size=32,
            learning_rate=0.001
        )
    
    # Inference
    elif args.mode == 'generate':
        # Load checkpoint
        system.load_checkpoint(args.model_path, args.processor_path)
        
        # Generate
        # generated_events = system.generate(
        #     length=32,
        #     temperature=0.8,
        #     top_k=50,
        #     top_p=0.9
        # )
        generated_events = system.generate(
            seed_sequence=[0] * 4,
            length=32,
            temperature=0.7,
            # min_bass_note=36,
            # max_bass_note=59,
            # min_upper_note=60,
            # max_upper_note=84
        )

        # validate_voice_separation(generated_events)

        # Save to MIDI
        # system.save_midi(generated_events, "generated_music.mid")
        system.create_harmonic_midi(generated_events, args.output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Music Transformer System')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train',
                      help='Mode of operation: train or generate')
    parser.add_argument('--midi_folder', type=str, help='Path to MIDI files for training')
    parser.add_argument('--model_path', type=str, help='Path to saved model for generation')
    parser.add_argument('--processor_path', type=str, help='Path to saved processor for generation')
    parser.add_argument('--output_path', type=str, default='generated_music.mid',
                      help='Path for generated MIDI output')
    
    args = parser.parse_args()

    args.mode = 'generate' # 'train' # 'generate'

    args.midi_folder = "../midi_dataset/piano_maestro-v1.0.0/all_years/"
    args.model_path = './models_chord_transformer/checkpoint_epoch_5.pth'
    args.processor_path = './models_chord_transformer/processor.pkl'

    

    main( )
