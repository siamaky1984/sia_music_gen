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
import music21  # For music theory analysis

from typing import List, Tuple, Set, Dict
from collections import defaultdict


from sia_LSTM_harmonic import InferenceManager, play_midi




class ChordEvent:
    """
    Represents a musical chord event with timing information.
    """
    def __init__(self, notes: Set[int], time: float, duration: float, velocity: int):
        self.notes = frozenset(notes)  # Immutable set of MIDI notes in the chord
        self.time = time              # Start time of the chord
        self.duration = duration      # How long the chord lasts
        self.velocity = velocity      # Volume/intensity of the chord
        self.bass_note = min(notes) if notes else 0  # Lowest note (bass)

class ChordAwareAttention(nn.Module):
    """
    Attention mechanism that considers three musical aspects:
    1. Pitch relationships
    2. Harmonic context
    3. Voice leading (bass movement)
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Separate query projections for different musical aspects
        self.pitch_query = nn.Linear(hidden_dim, hidden_dim)      # For individual note relationships
        self.harmony_query = nn.Linear(hidden_dim, hidden_dim)    # For chord progressions
        self.voice_leading_query = nn.Linear(hidden_dim, hidden_dim)  # For bass line movement
        self.melody_query = nn.Linear(hidden_dim, hidden_dim)
        self.rhythm_query = nn.Linear(hidden_dim, hidden_dim)
        
        # Standard key and value projections
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Projections for musical features
        self.chord_proj = nn.Linear(12, self.hidden_dim)  # Projects chord (12 pitch classes) to attention space
        self.bass_proj = nn.Linear(12, self.hidden_dim)   # Projects bass note to attention space
        self.melody_proj = nn.Linear(24, hidden_dim)        # Melody contour (2 octaves)
        self.chord_quality_proj = nn.Linear(7, hidden_dim)  # Major, minor, dim, aug, sus2, sus4, 7th
        self.interval_proj = nn.Linear(12, hidden_dim)      # Intervals within chord
        self.rhythm_proj = nn.Linear(4, hidden_dim)         # Duration features


        # # Additional feature projections for harmonic analysis
        # self.function_proj = nn.Linear(7, hidden_dim)      # Harmonic function (T/S/D)
        # self.roman_proj = nn.Linear(12, hidden_dim)        # Roman numeral analysis
        # self.cadence_proj = nn.Linear(4, hidden_dim)       # Cadence patterns
        # self.tension_proj = nn.Linear(3, hidden_dim)       # Harmonic tension
        # self.progression_proj = nn.Linear(8, hidden_dim)   # Common progressions
        
        # self.harmonic_analyzer = HarmonicAnalyzer()
        
        # # Additional queries for harmonic aspects
        # self.function_query = nn.Linear(hidden_dim, hidden_dim)
        # self.progression_query = nn.Linear(hidden_dim, hidden_dim)
        # self.tension_query = nn.Linear(hidden_dim, hidden_dim)


        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

        
        # Define common chord qualities for recognition
        self.chord_qualities = {
            'major': {0, 4, 7},        # Root, major third, fifth
            'minor': {0, 3, 7},        # Root, minor third, fifth
            'diminished': {0, 3, 6},   # Root, minor third, diminished fifth
            'augmented': {0, 4, 8},    # Root, major third, augmented fifth
            'sus2': {0, 2, 7},         # Root, second, fifth
            'sus4': {0, 5, 7},         # Root, fourth, fifth
            'dominant7': {0, 4, 7, 10}, # Root, major third, fifth, minor seventh
        }
    
    # def get_chord_features(self, notes: Set[int], device) -> torch.Tensor:
    #     """
    #     Convert a chord into musical features:
    #     1. Pitch class vector (which of the 12 notes are present)
    #     2. Bass note (lowest note)
    #     3. Chord characteristics (size, intervals, etc.)
    #     """
    #     # Create 12-dimensional pitch class vector (C, C#, D, etc.)
    #     pitch_classes = torch.zeros(12, device = device)
    #     for note in notes:
    #         pitch_classes[note % 12] = 1  # Map MIDI note to pitch class
        
    #     # Get bass note (lowest note)
    #     bass = min(notes) % 12
    #     bass_vector = torch.zeros(12, device = device)
    #     bass_vector[bass] = 1
        
    #     # Analyze chord structure
    #     notes = sorted(list(notes))
    #     intervals = [notes[i+1] - notes[i] for i in range(len(notes)-1)]
        
    #     # # Features about chord structure
    #     # tension_features = torch.tensor([
    #     #     len(notes),        # Chord size
    #     #     max(intervals) if intervals else 0,  # Largest interval
    #     #     min(intervals) if intervals else 0,  # Smallest interval
    #     #     sum(intervals) / len(intervals) if intervals else 0,  # Average interval
    #     # ])
        
    #     return pitch_classes, bass_vector #, tension_features
    

    def get_musical_features(self, notes: Set[int], prev_notes: Set[int], time: float, 
                           duration: float, device: torch.device) -> Dict[str, torch.Tensor]:
        """Extract comprehensive musical features from a chord event."""
        # Sort notes for consistent processing
        sorted_notes = sorted(list(notes))
        root = min(notes)  # Assume lowest note is root
        
        # 1. Pitch Class Vector (which notes are present)
        pitch_classes = torch.zeros(12, device=device)
        for note in notes:
            pitch_classes[note % 12] = 1
        
        # 2. Bass Note (lowest note)
        bass_vector = torch.zeros(12, device=device)
        bass_vector[root % 12] = 1
        
        # 3. Melody Contour (top two octaves of notes)
        melody_vector = torch.zeros(24, device=device)
        if sorted_notes:
            top_notes = [n % 24 for n in sorted_notes[-2:]]  # Top 2 notes within 2 octaves
            for note in top_notes:
                melody_vector[note] = 1
        
        # 4. Chord Quality Recognition
        quality_vector = torch.zeros(7, device=device)  # One-hot for 7 chord qualities
        normalized_notes = {(n - root) % 12 for n in notes}  # Normalize to root
        
        for idx, (quality, pattern) in enumerate(self.chord_qualities.items()):
            if normalized_notes == pattern:
                quality_vector[idx] = 1
                break
        
        # 5. Interval Content
        interval_vector = torch.zeros(12, device=device)
        for i, note1 in enumerate(sorted_notes):
            for note2 in sorted_notes[i+1:]:
                interval = (note2 - note1) % 12
                interval_vector[interval] = 1
        
        # 6. Rhythm Features
        rhythm_vector = torch.zeros(4, device=device)
        rhythm_vector[0] = duration                    # Duration
        rhythm_vector[1] = time % 4                    # Position in bar (assuming 4/4)
        rhythm_vector[2] = 1 if time.is_integer() else 0  # On-beat vs off-beat
        rhythm_vector[3] = len(notes) / 8              # Normalized density
        
        # 7. Voice Leading (movement from previous chord)
        if prev_notes:
            prev_sorted = sorted(list(prev_notes))
            voice_leading = [min(abs(n1 - n2) for n2 in sorted_notes) 
                           for n1 in prev_sorted]
            avg_movement = sum(voice_leading) / len(voice_leading)
        else:
            avg_movement = 0
        
        return {
            'pitch_classes': pitch_classes,
            'bass': bass_vector,
            'melody': melody_vector,
            'quality': quality_vector,
            'intervals': interval_vector,
            'rhythm': rhythm_vector,
            'voice_leading': torch.tensor([avg_movement], device=device)
        }

    
    # def forward(self, query, key, value, chord_events, mask=None):
    #     """
    #     Forward pass combining three types of attention:
    #     1. Pitch attention: Note-to-note relationships
    #     2. Harmony attention: Chord progression patterns
    #     3. Voice leading attention: Bass line movement
    #     """
    #     batch_size = query.size(0)
    #     seq_len = query.size(1)
    #     device = query.device
        
    #     # Process different musical aspects
    #     harmony_q = self.harmony_query(query)
    #     pitch_q = self.pitch_query(query)
    #     voice_leading_q = self.voice_leading_query(query)
        
    #     k = self.key(key)
    #     v = self.value(value)
        
    #     # Get musical features for each chord
    #      # Initialize empty tensors on GPU
    #     chord_features = torch.empty(
    #         # (seq_len, self.head_dim),
    #         (seq_len, self.hidden_dim),
    #         device=device
    #     )
    #     bass_features = torch.empty(
    #         # (seq_len, self.head_dim),
    #         (seq_len, self.hidden_dim),
    #         device=device
    #     )

    #     # Process chord events
    #     for i, event in enumerate(chord_events):
    #         pitch_classes, bass_vector = self.get_chord_features(event.notes, device)
            
    #         # Project features
    #         chord_features[i] = self.chord_proj(pitch_classes)
    #         bass_features[i] = self.bass_proj(bass_vector)
        
    #     # Expand features for batch size
    #     chord_features = chord_features.unsqueeze(0).expand(batch_size, -1, -1)
    #     bass_features = bass_features.unsqueeze(0).expand(batch_size, -1, -1)


    #     # for event in chord_events:
    #     #     pitch_classes, bass, tension = self.get_chord_features(event.notes)
    #     #     # Project features to attention space
    #     #     chord_features.append(self.chord_proj(pitch_classes))
    #     #     bass_features.append(self.bass_proj(bass))
        
    #     # # Stack features
    #     # chord_features = torch.stack(chord_features).to(query.device)
    #     # bass_features = torch.stack(bass_features).to(query.device)

    #     # Process queries
    #     pitch_q = self.pitch_query(query)
    #     harmony_q = self.harmony_query(query)
    #     voice_leading_q = self.voice_leading_query(query)
    #     k = self.key(key)
    #     v = self.value(value)


        
    #     # Reshape for multi-head attention
    #     def reshape_for_heads(x):
    #         return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    #     # Prepare queries for different aspects
    #     harmony_q = reshape_for_heads(harmony_q)
    #     pitch_q = reshape_for_heads(pitch_q)
    #     voice_leading_q = reshape_for_heads(voice_leading_q)
    #     k = reshape_for_heads(k)
    #     v = reshape_for_heads(v)
        

    #     # Reshape features with same shape as queries
    #     chord_features = reshape_for_heads(chord_features)  # (32, 4, 16, 128)
    #     bass_features = reshape_for_heads(bass_features)    # (32, 4, 16, 128)

    #     # Calculate attention scores for different musical aspects
    #     scale = self.head_dim ** -0.5
        
    #     # # Harmonic attention (chord progressions)
    #     # harmony_scores = torch.matmul(harmony_q, chord_features.unsqueeze(1)) * scale
        
    #     # # Pitch attention (note relationships)
    #     # pitch_scores = torch.matmul(pitch_q, k.transpose(-2, -1)) * scale
        
    #     # # Voice leading attention (bass line)
    #     # voice_leading_scores = torch.matmul(voice_leading_q, bass_features.unsqueeze(1)) * scale

    #      # Now all matmuls will have matching dimensions
    #     pitch_scores = torch.matmul(pitch_q, k.transpose(-2, -1))         # (32, 4, 16, 16)
    #     harmony_scores = torch.matmul(harmony_q, chord_features.transpose(-2, -1))  # (32, 4, 16, 16)
    #     voice_leading_scores = torch.matmul(voice_leading_q, bass_features.transpose(-2, -1))      # (32, 4, 16, 16)
        
    #     # Combine all attention aspects
    #     attention_scores = (harmony_scores + pitch_scores + voice_leading_scores) / 3.0
        
    #     # Apply mask if provided
    #     if mask is not None:
    #         attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
    #     # Calculate attention weights and apply to values
    #     attention_weights = F.softmax(attention_scores, dim=-1)
    #     attention_weights = self.dropout(attention_weights)
        
    #     # Get weighted combination of values
    #     context = torch.matmul(attention_weights, v)
        
    #     # Reshape and project output
    #     context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
    #     output = self.output_proj(context)
        
    #     return output, attention_weights


    #### TODO additional Harmony features
    # def analyze_harmony(self, current_notes: Set[int], prev_notes: Set[int], 
    #                    next_notes: Set[int], estimated_key: int) -> Dict[str, torch.Tensor]:
    #     """Comprehensive harmonic analysis of a chord in context."""
    #     device = next(self.parameters()).device
        
    #     # Normalize notes to key
    #     normalized_notes = {(n - estimated_key) % 12 for n in current_notes}
    #     root = min(normalized_notes)
        
    #     # 1. Roman Numeral Analysis
    #     roman_vector = torch.zeros(12, device=device)
    #     roman_vector[root] = 1
        
    #     # 2. Harmonic Function (T/S/D)
    #     function_vector = torch.zeros(7, device=device)
    #     if root in self.harmonic_analyzer.chord_functions['major']:
    #         function = self.harmonic_analyzer.chord_functions['major'][root]
    #         function_idx = {'T': 0, 'S': 1, 'D': 2}[function]
    #         function_vector[function_idx] = 1
            
    #         # Secondary dominants
    #         if self.is_dominant_seventh(normalized_notes):
    #             function_vector[3] = 1  # Secondary dominant flag
        
    #     # 3. Cadence Analysis
    #     cadence_vector = torch.zeros(4, device=device)
    #     if prev_notes and next_notes:
    #         prev_function = self.get_harmonic_function(prev_notes, estimated_key)
    #         next_function = self.get_harmonic_function(next_notes, estimated_key)
            
    #         cadence_type = self.analyze_cadence(prev_function, function, next_function)
    #         cadence_vector[{
    #             'authentic': 0,
    #             'plagal': 1,
    #             'deceptive': 2,
    #             'half': 3
    #         }.get(cadence_type, -1)] = 1
        
    #     # 4. Harmonic Tension Analysis
    #     tension_vector = torch.zeros(3, device=device)
    #     tension_vector[0] = self.calculate_harmonic_tension(normalized_notes)
    #     if prev_notes:
    #         tension_vector[1] = self.calculate_voice_leading_tension(prev_notes, current_notes)
    #     tension_vector[2] = self.calculate_dissonance(normalized_notes)
        
    #     # 5. Progression Pattern Analysis
    #     progression_vector = torch.zeros(8, device=device)
    #     if prev_notes:
    #         prev_root = min({(n - estimated_key) % 12 for n in prev_notes})
    #         root_movement = (root - prev_root) % 12
            
    #         # Common root movements
    #         progression_vector[{
    #             5: 0,  # Up perfect fourth
    #             7: 1,  # Up perfect fifth
    #             2: 2,  # Up major second
    #             10: 3, # Down major second
    #             9: 4,  # Up major sixth (relative major/minor)
    #             3: 5,  # Up minor third
    #             8: 6   # Up minor sixth
    #         }.get(root_movement, 7)] = 1
        
    #     return {
    #         'roman': roman_vector,
    #         'function': function_vector,
    #         'cadence': cadence_vector,
    #         'tension': tension_vector,
    #         'progression': progression_vector
    #     }
    
    # def calculate_harmonic_tension(self, notes: Set[int]) -> float:
    #     """Calculate harmonic tension based on chord quality and intervals."""
    #     # Get chord quality
    #     chord_type = self.identify_chord_type(notes)
    #     base_tension = self.harmonic_analyzer.tension_levels.get(chord_type, 0.5)
        
    #     # Add tension for dissonant intervals
    #     intervals = {(n2 - n1) % 12 for n1 in notes for n2 in notes if n1 < n2}
    #     dissonant_intervals = {1, 2, 6, 10, 11}  # Minor 2nd, Major 2nd, Tritone, etc.
    #     dissonance_count = len(intervals & dissonant_intervals)
        
    #     return base_tension + (dissonance_count * 0.1)
    
    # def calculate_voice_leading_tension(self, prev_notes: Set[int], 
    #                                   current_notes: Set[int]) -> float:
    #     """Calculate tension from voice leading."""
    #     # Find minimal movement between voices
    #     movements = []
    #     for p1 in prev_notes:
    #         movement = min(abs(p1 - p2) for p2 in current_notes)
    #         movements.append(movement)
        
    #     # Larger movements create more tension
    #     return sum(mov * 0.1 for mov in movements) / len(movements)
    
    # def calculate_dissonance(self, notes: Set[int]) -> float:
    #     """Calculate dissonance level based on interval content."""
    #     intervals = {(n2 - n1) % 12 for n1 in notes for n2 in notes if n1 < n2}
        
    #     # Weight different intervals by dissonance
    #     interval_weights = {
    #         0: 0.0,  # Unison
    #         7: 0.1,  # Perfect fifth
    #         4: 0.2,  # Major third
    #         3: 0.3,  # Minor third
    #         8: 0.4,  # Minor sixth
    #         9: 0.4,  # Major sixth
    #         5: 0.5,  # Perfect fourth
    #         2: 0.7,  # Major second
    #         1: 0.8,  # Minor second
    #         6: 0.9,  # Tritone
    #         10: 0.7, # Minor seventh
    #         11: 0.8  # Major seventh
    #     }
        
    #     return sum(interval_weights.get(i, 0.5) for i in intervals) / len(intervals)
    
    # def get_harmonic_function(self, notes: Set[int], key: int) -> str:
    #     """Get primary harmonic function of a chord."""
    #     root = min((n - key) % 12 for n in notes)
    #     return self.harmonic_analyzer.chord_functions['major'].get(root, 'X')
    
    # def analyze_cadence(self, prev_func: str, curr_func: str, next_func: str) -> str:
    #     """Analyze cadence type based on function progression."""
    #     pattern = (prev_func, curr_func)
    #     return self.harmonic_analyzer.cadence_patterns.get(pattern, 'none')
    



    def forward(self, query, key, value, chord_events, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        device = query.device
        
        # Initialize feature tensors
        features = {
            'chord': torch.empty((seq_len, self.hidden_dim), device=device),
            'bass': torch.empty((seq_len, self.hidden_dim), device=device),
            'melody': torch.empty((seq_len, self.hidden_dim), device=device),
            'quality': torch.empty((seq_len, self.hidden_dim), device=device),
            'intervals': torch.empty((seq_len, self.hidden_dim), device=device),
            'rhythm': torch.empty((seq_len, self.hidden_dim), device=device)
        }
        
        # Process chord events and extract features
        prev_notes = None
        for i, event in enumerate(chord_events):
            musical_features = self.get_musical_features(
                event.notes, prev_notes, event.time, event.duration, device
            )
            
            # Project each feature type
            features['chord'][i] = self.chord_proj(musical_features['pitch_classes'])
            features['bass'][i] = self.bass_proj(musical_features['bass'])
            features['melody'][i] = self.melody_proj(musical_features['melody'])
            features['quality'][i] = self.chord_quality_proj(musical_features['quality'])
            features['intervals'][i] = self.interval_proj(musical_features['intervals'])
            features['rhythm'][i] = self.rhythm_proj(musical_features['rhythm'])
            
            prev_notes = event.notes
        
        # Expand features for batch dimension
        for kk in features:
            features[kk] = features[kk].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process queries
        queries = {
            'pitch': self.pitch_query(query),
            'harmony': self.harmony_query(query),
            'voice': self.voice_leading_query(query),
            'melody': self.melody_query(query),
            'rhythm': self.rhythm_query(query)
        }
        
        k = self.key(key)
        v = self.value(value)
        
        # Reshape for multi-head attention
        def reshape_for_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape all queries and features
        queries = {k: reshape_for_heads(q) for k, q in queries.items()}
        features = {k: reshape_for_heads(f) for k, f in features.items()}
        k = reshape_for_heads(k)
        v = reshape_for_heads(v)
        
        # Calculate attention scores for each aspect
        scale = self.head_dim ** -0.5
        scores = {}
        
        scores['pitch'] = torch.matmul(queries['pitch'], k.transpose(-2, -1))
        scores['harmony'] = torch.matmul(queries['harmony'], features['chord'].transpose(-2, -1))
        scores['melody'] = torch.matmul(queries['melody'], features['melody'].transpose(-2, -1))
        scores['voice'] = torch.matmul(queries['voice'], features['bass'].transpose(-2, -1))
        scores['rhythm'] = torch.matmul(queries['rhythm'], features['rhythm'].transpose(-2, -1))
        
        # Combine all scores
        combined_scores = sum(scores.values()) * scale / len(scores)
        
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        #     mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        #     combined_scores = combined_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(combined_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_linear(context)
        
        return output, attention_weights
    

class HarmonicAnalyzer:
    """Analyzes harmonic functions and relationships."""
    def __init__(self):
        # Define scale degrees for different modes
        self.scale_degrees = {
            'major': [0, 2, 4, 5, 7, 9, 11],  # C major scale degrees
            'minor': [0, 2, 3, 5, 7, 8, 10],  # Natural minor scale degrees
        }
        
        # Define chord functions in major and minor keys
        self.chord_functions = {
            'major': {
                0: 'T',   # Tonic (I)
                4: 'S',   # Subdominant (IV)
                7: 'D',   # Dominant (V)
                9: 'T',   # Relative minor (vi)
                2: 'S',   # Supertonic (ii)
                11: 'D',  # Leading tone (vii)
                5: 'D'    # Dominant preparation (iii)
            },
            'minor': {
                0: 't',   # Tonic (i)
                3: 'S',   # Mediant (III)
                7: 'd',   # Dominant (v)
                5: 's',   # Subdominant (iv)
                2: 'd',   # Supertonic (ii°)
                10: 'S',  # Subtonic (VII)
                8: 'D'    # Dominant (V) in harmonic minor
            }
        }
        
        # Define common cadence patterns
        self.cadence_patterns = {
            ('D', 'T'): 'authentic',
            ('S', 'T'): 'plagal',
            ('D', 't'): 'deceptive',
            ('D', 'S'): 'half'
        }
        
        # Define tension levels for different chord types
        self.tension_levels = {
            'major': 0.2,
            'minor': 0.3,
            'dominant7': 0.6,
            'diminished': 0.7,
            'augmented': 0.8,
            'half_diminished': 0.65,
            'french6': 0.75
        }


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


class HarmonicSequenceLSTM(nn.Module):
    """LSTM with chord-aware attention."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = ChordAwareAttention(
            hidden_dim,
            num_heads,
            dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, chord_events=None, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Attention
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            chord_events=chord_events
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(lstm_out + attended_out)
        
        # Final output
        output = self.fc(self.dropout(output))
        
        return output, hidden, attention_weights
    


class HarmonicEventProcessor:
    """Processes and manages chord events and vocabulary."""
    def __init__(self, time_threshold: float = 0.25):
        self.time_threshold = time_threshold
        self.chord_to_idx = {}
        self.idx_to_chord = {}
        self.vocab_size = 0
    
    def extract_chord_events(self, midi_file: str) -> List[ChordEvent]:
        """Extract chord events from MIDI file."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            events = []
            
            # Group notes by time
            note_groups = defaultdict(set)
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        start_time = round(note.start / self.time_threshold) * self.time_threshold
                        note_groups[start_time].add(note.pitch)
            
            # Create chord events
            for time, notes in sorted(note_groups.items()):
                if notes:  # Skip empty sets
                    event = ChordEvent(
                        notes=notes,
                        time=time,
                        duration=self.time_threshold,
                        velocity=80
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            print(f"Error processing {midi_file}: {str(e)}")
            return []

    def create_vocabulary(self, all_sequences: List[List[ChordEvent]]):
        """Create vocabulary from chord events."""
        unique_chords = set()
        for sequence in all_sequences:
            for event in sequence:
                unique_chords.add(event.notes)
        
        self.chord_to_idx = {chord: idx for idx, chord in enumerate(sorted(unique_chords))}
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}
        self.vocab_size = len(self.chord_to_idx)
        print(f"Created vocabulary with {self.vocab_size} unique chords")

    def events_to_indices(self, events: List[ChordEvent]) -> List[int]:
        """Convert chord events to indices."""
        return [self.chord_to_idx[event.notes] for event in events]

    def indices_to_events(self, indices: List[int], base_duration: float = 0.25) -> List[ChordEvent]:
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
                    velocity=80
                ))
                current_time += base_duration
        
        return events


class HarmonicSequenceTrainer:
    """Trainer class for the harmonic sequence model."""
    def __init__(self, processor: HarmonicEventProcessor, model_params: Dict):
        self.processor = processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = HarmonicSequenceLSTM(
            vocab_size=processor.vocab_size,
            embedding_dim=model_params['embedding_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            num_heads=model_params.get('num_heads', 4),
            dropout=model_params.get('dropout', 0.1)
        ).to(self.device)
    
    def train(self, dataloader: DataLoader, num_epochs: int, learning_rate: float):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                optimizer.zero_grad()
                
                # Convert indices to chord events for attention
                # chord_events = self.processor.indices_to_events(input_seq[0].tolist())

                chord_events = self.processor.indices_to_events(input_seq[0].cpu().tolist())
                # chord_events.to(self.device)
                
                # Forward pass
                output, _, _ = self.model(input_seq, chord_events)
                
                # Calculate loss
                output = output.view(-1, self.processor.vocab_size)
                target_seq = target_seq.view(-1)
                
                loss = criterion(output, target_seq)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}')
    
    def save_model(self, path: str):
        """Save model and processor."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'processor': self.processor
        }, path)
    
    def load_model(self, path: str):
        """Load model and processor."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.processor = checkpoint['processor']

        return self.model, self.processor


# Example usage
def prepare_training_data(midi_files: List[str]) -> Tuple[HarmonicEventProcessor, DataLoader]:
    """Prepare data for training with chord events."""
    processor = HarmonicEventProcessor()
    all_sequences = []
    
    # Process MIDI files
    for midi_file in midi_files:
        # Extract chord events from MIDI
        # events = extract_chord_events_from_midi(midi_file)
        events = processor.extract_chord_events(midi_file)
        if events:
            all_sequences.append(events)
    
    # Create vocabulary
    processor.create_vocabulary(all_sequences)
    
    # Convert to indices
    indexed_sequences = []
    for sequence in all_sequences:
        indices = [processor.chord_to_idx[frozenset(event.notes)] for event in sequence]
        indexed_sequences.append(indices)
    
    # Create dataset
    dataset = HarmonicSequenceDataset(indexed_sequences, sequence_length=16)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return processor, dataloader

def extract_chord_events_from_midi(midi_file: str) -> List[ChordEvent]:
    """Extract chord events from MIDI file."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        events = []
        current_time = 0.0
        
        # Group notes by time
        note_groups = defaultdict(set)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    # Round time to nearest grid point
                    start_time = round(note.start / 0.25) * 0.25
                    note_groups[start_time].add(note.pitch)
        
        # Convert to chord events
        for time, notes in sorted(note_groups.items()):
            if notes:  # If we have notes at this time
                event = ChordEvent(
                    notes=notes,
                    time=time,
                    duration=0.25,  # Fixed duration for simplicity
                    velocity=80     # Default velocity
                )
                events.append(event)
        
        return events
    
    except Exception as e:
        print(f"Error processing {midi_file}: {str(e)}")
        return []

# Main training script
def main():

    mode =  'train'

    # Setup
    midi_folder = "./midi_dataset/piano_maestro-v1.0.0/2004/" # all_years/"
    midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + \
                    glob.glob(os.path.join(midi_folder, "*.midi"))
    print( len(midi_files) )
    processor, dataloader = prepare_training_data(midi_files)
    
    if mode == 'train':
        
        # Create model
        model_params = {
            'vocab_size': processor.vocab_size,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 3,
            'num_heads': 4,
            'dropout': 0.1
        }
        
        # # Initialize trainer
        trainer = HarmonicSequenceTrainer(processor, model_params)
        
        # Train
        trainer.train(dataloader, num_epochs=10, learning_rate=0.001)

        
        # Save model
        trainer.save_model('harmonic_model_attention.pth')
    

    elif mode =='inference':
        # Paths to saved files
        model_path = './models/harmonic_model_attention.pth'

        processor_path = './models/processor.pkl'
        
        output_path = 'generated_sequence.mid'

        # Create model
        # model_params = {
        #     'vocab_size': processor.vocab_size,
        #     'embedding_dim': 256,
        #     'hidden_dim': 512,
        #     'num_layers': 3,
        #     'num_heads': 4,
        #     'dropout': 0.1
        # }

        # trainer = HarmonicSequenceTrainer(processor, model_params)

        # model, processor_pkl = trainer.load_model( model_path )



        print("Initializing inference manager...")
        inference_manager = InferenceManager(model_path, processor_path)
        
        generated_indices, generated_events = inference_manager.safe_generate(
        sequence_length=16,
        generation_length=32,
        temperature=0.8
        )


        # Save to MIDI
        print("Saving to MIDI...")
        inference_manager.save_midi(generated_events, output_path)
        
        print("\nGeneration Statistics:")
        print(f"Total events generated: {len(generated_events)}")
        print(f"Unique chords used: {len(set([event.notes for event in generated_events]))}")

        play_midi(output_path)

    



if __name__ == "__main__":
    main()