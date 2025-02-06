import mido
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import math

class RhythmDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RhythmModel(nn.Module):
    def __init__(self, sequence_length=16, hidden_dim=64):
        super(RhythmModel, self).__init__()
        # self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True)
        # self.dropout = nn.Dropout(0.2)
        # self.dense1 = nn.Linear(hidden_dim, 32)
        # self.dense2 = nn.Linear(32, 1)

        self.embedding = nn.Linear(1, 32)
        self.pos_encoder = PositionalEncoding(32, max_len=sequence_length)
        
        self.lstm1 = nn.LSTM(32, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True)
        
        self.attention = nn.MultiheadAttention(hidden_dim//2, num_heads=4)
        
        self.dense1 = nn.Linear(hidden_dim//2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dense2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dense3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # lstm_out, _ = self.lstm(x)
        # lstm_out = lstm_out[:, -1, :]  # Take last output
        # x = self.dropout(lstm_out)
        # x = torch.relu(self.dense1(x))
        # x = torch.sigmoid(self.dense2(x))
        
        # Input embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # LSTM layers with residual connections
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = lstm2_out + lstm1_out[:, :, :lstm2_out.size(2)]
        
        # Self-attention
        attended, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        x = attended[:, -1, :]
        
        # Dense layers with batch normalization
        x = torch.relu(self.bn1(self.dense1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.dense2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.dense3(x))

        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RhythmLearner:
    def __init__(self, device=None):
        self.model = None
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def extract_rhythm_from_midi(self, midi_file):
        """Extract rhythm patterns from a MIDI file"""
        mid = mido.MidiFile(midi_file)
        total_ticks = 0
        
        # Calculate total ticks
        for track in mid.tracks:
            track_ticks = 0
            for msg in track:
                track_ticks += msg.time
            total_ticks = max(total_ticks, track_ticks)
        
        timeline = np.zeros(total_ticks)
        
        for track in mid.tracks:
            current_time = 0
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    if current_time < len(timeline):
                        timeline[current_time] = 1
                current_time += msg.time
        
        # Quantize timeline
        ticks_per_bar = mid.ticks_per_beat * 4
        quantization_factor = ticks_per_bar // 16  # 16 steps per bar
        
        quantized_length = len(timeline) // quantization_factor
        quantized_timeline = np.zeros(quantized_length)
        
        for i in range(quantized_length):
            start = i * quantization_factor
            end = start + quantization_factor
            if start < len(timeline) and any(timeline[start:end]):
                quantized_timeline[i] = 1
        
        # Split into bars
        bars = []
        steps_per_bar = 16
        for i in range(0, len(quantized_timeline), steps_per_bar):
            if i + steps_per_bar <= len(quantized_timeline):
                bars.append(quantized_timeline[i:i + steps_per_bar])
        
        return bars
    
    def prepare_training_data(self, midi_folder):
        """Prepare training data for rhythm prediction"""
        all_bars = []
        
        for filename in os.listdir(midi_folder):
            if filename.endswith(('.mid', '.midi')):
                try:
                    bars = self.extract_rhythm_from_midi(os.path.join(midi_folder, filename))
                    all_bars.extend(bars)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        if not all_bars:
            raise ValueError("No rhythm patterns extracted from MIDI files")
            
        # Create sequences
        sequences = []
        next_steps = []
        sequence_length = 16
        
        # Create continuous sequence
        full_sequence = np.concatenate(all_bars)
        
        for i in range(len(full_sequence) - sequence_length):
            sequences.append(full_sequence[i:i + sequence_length])
            next_steps.append(full_sequence[i + sequence_length])
        
        X = np.array(sequences).reshape(-1, sequence_length, 1)
        y = np.array(next_steps)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        
        return X, y
    
    def train(self, midi_folder, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the rhythm model"""
        X, y = self.prepare_training_data(midi_folder)
        
        dataset = RhythmDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if self.model is None:
            self.model = RhythmModel().to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    def generate_rhythm(self, bars=1, seed_pattern=None):
        """Generate rhythm patterns"""
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        self.model.eval()
        steps_per_bar = 16
        total_steps = steps_per_bar * bars
        
        if seed_pattern is None:
            seed_pattern = np.zeros(steps_per_bar)
            seed_pattern[0] = 1  # Start with a hit
        
        # Extend seed pattern to cover the required number of bars
        seed_pattern = np.tile(seed_pattern, bars)
        
        sequence = torch.FloatTensor(seed_pattern).reshape(1, -1, 1).to(self.device)
        generated_rhythm = list(seed_pattern)
        
        with torch.no_grad():
            while len(generated_rhythm) < total_steps:
                pred = self.model(sequence)
                next_step = (pred.item() > 0.5) * 1
                
                generated_rhythm.append(next_step)
                
                # Update sequence
                sequence = torch.FloatTensor(generated_rhythm[-steps_per_bar:]).reshape(1, -1, 1).to(self.device)
        
        return generated_rhythm
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load a trained model"""
        if self.model is None:
            self.model = RhythmModel().to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()




class MelodyDataset(Dataset):
    def __init__(self, X, y_pitch, y_duration):
        self.X = torch.FloatTensor(X)
        self.y_pitch = torch.LongTensor(y_pitch)
        self.y_duration = torch.FloatTensor(y_duration)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_pitch[idx], self.y_duration[idx]

class MelodyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MelodyModel, self).__init__()
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        # self.dropout = nn.Dropout(0.2)
        
        # # Pitch prediction branch
        # self.pitch_dense = nn.Linear(hidden_dim, 32)
        # self.pitch_out = nn.Linear(32, 12)
        
        # # Duration prediction branch
        # self.duration_dense = nn.Linear(hidden_dim, 32)
        # self.duration_out = nn.Linear(32, 1)

        self.input_bn = nn.BatchNorm1d(input_dim)
        self.embedding = nn.Linear(input_dim, 64)
        self.pos_encoder = PositionalEncoding(64, max_len=8)
        
        self.lstm1 = nn.LSTM(64, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Pitch prediction branch
        self.pitch_dense1 = nn.Linear(hidden_dim, 64)
        self.pitch_bn1 = nn.BatchNorm1d(64)
        self.pitch_dense2 = nn.Linear(64, 32)
        self.pitch_bn2 = nn.BatchNorm1d(32)
        self.pitch_out = nn.Linear(32, 12)
        
        # Duration prediction branch
        self.duration_dense1 = nn.Linear(hidden_dim, 64)
        self.duration_bn1 = nn.BatchNorm1d(64)
        self.duration_dense2 = nn.Linear(64, 32)
        self.duration_bn2 = nn.BatchNorm1d(32)
        self.duration_out = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        # lstm_out, _ = self.lstm(x)
        # lstm_out = lstm_out[:, -1, :]  # Take last output
        # lstm_out = self.dropout(lstm_out)
        
        # # Pitch prediction
        # pitch = torch.relu(self.pitch_dense(lstm_out))
        # pitch = self.pitch_out(pitch)
        
        # # Duration prediction
        # duration = torch.relu(self.duration_dense(lstm_out))
        # duration = torch.sigmoid(self.duration_out(duration))
        
        
        batch_size = x.size(0)
        
        # Input normalization and embedding
        x = x.view(batch_size * x.size(1), -1)
        x = self.input_bn(x)
        x = x.view(batch_size, -1, x.size(-1))
        
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # LSTM layers with residual connections
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = lstm2_out + lstm1_out
        
        # Self-attention
        attended, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        x = attended[:, -1, :]
        
        # Pitch prediction
        pitch = torch.relu(self.pitch_bn1(self.pitch_dense1(x)))
        pitch = self.dropout(pitch)
        pitch = torch.relu(self.pitch_bn2(self.pitch_dense2(pitch)))
        pitch = self.pitch_out(pitch)
        
        # Duration prediction
        duration = torch.relu(self.duration_bn1(self.duration_dense1(x)))
        duration = self.dropout(duration)
        duration = torch.relu(self.duration_bn2(self.duration_dense2(duration)))
        duration = torch.sigmoid(self.duration_out(duration))
        
        return pitch, duration
    


# Melody extraction heuristics
def analyze_track(notes):
    if not notes:
        return -float('inf')
        
    # Calculate track features
    pitches = [note['pitch'] for note in notes]
    avg_pitch = sum(pitches) / len(pitches)
    max_pitch = max(pitches)
    min_pitch = min(pitches)
    pitch_range = max_pitch - min_pitch
    
    # Time between consecutive notes
    intervals = []
    sorted_notes = sorted(notes, key=lambda x: x['start_time'])
    for i in range(1, len(sorted_notes)):
        interval = sorted_notes[i]['start_time'] - sorted_notes[i-1]['end_time']
        intervals.append(interval)
    avg_interval = sum(intervals) / len(intervals) if intervals else 0
    
    # Score the track based on melodic characteristics
    score = 0
    
    # Prefer tracks in typical melody range (60-84)
    if 60 <= avg_pitch <= 84:
        score += 10
    else:
        score -= abs(72 - avg_pitch) / 12
        
    # Prefer reasonable pitch range (not too static, not too wild)
    if 12 <= pitch_range <= 24:  # One to two octaves
        score += 5
    else:
        score -= abs(18 - pitch_range) / 6
        
    # Prefer tracks that mostly play one note at a time
    simultaneous_notes = 0
    for i, note in enumerate(notes):
        overlap = sum(1 for other in notes if 
                    i != notes.index(other) and
                    note['start_time'] < other['end_time'] and
                    other['start_time'] < note['end_time'])
        simultaneous_notes += overlap
    polyphony_ratio = simultaneous_notes / len(notes)
    score -= polyphony_ratio * 10  # Penalize polyphonic sections
    
    # Prefer moderate note intervals
    if 0.1 <= avg_interval <= 1.0:  # Reasonable time between notes
        score += 5
    
    return score
    

class MelodyLearner:
    def __init__(self, device=None):
        self.model = None
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


    def extract_melody_and_rhythm(self, midi_file):
        """Extract melody and rhythm from MIDI file"""
        mid = mido.MidiFile(midi_file)
        events = []

        C4 = 60
        
        for track in mid.tracks:
            current_time = 0
            active_notes = {}
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0 and msg.note >= C4: #### only include right hand side of piano
                    active_notes[msg.note] = current_time
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time = active_notes[msg.note]
                        duration = current_time - start_time
                        pitch_class = msg.note % 12
                        
                        events.append({
                            'time': start_time,
                            'duration': duration,
                            'pitch_class': pitch_class,
                            'midi_note': msg.note
                        })
                        del active_notes[msg.note]
        
        return sorted(events, key=lambda x: x['time'])

        
    # def extract_melody_and_rhythm(self, midi_file):
    #     """Extract melody and rhythm from MIDI file"""
    #     mid = mido.MidiFile(midi_file)
    #     tracks_notes = []
        
    #     for track in mid.tracks:
    #         current_time = 0
    #         track_notes  = []
    #         active_notes = {}
            
    #         for msg in track:
    #             current_time += msg.time
                
    #             if msg.type == 'note_on' and msg.velocity > 0:
    #                 # Note starts
    #                 pitch = msg.note
    #                 active_notes[pitch] ={
    #                 'start_time': current_time,
    #                 'velocity': msg.velocity
    #                 }
    #             elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
    #                 # Note ends
    #                 pitch = msg.note
    #                 if pitch in active_notes:
    #                     note_data  = active_notes[msg.note]
    #                     duration = current_time - note_data['start_time']
    #                     pitch_class = msg.note % 12
    #                     velocity = active_notes[pitch]['velocity']
                        
    #                     track_notes.append({
    #                         # 'time': start_time,
    #                         # 'duration': duration,
    #                         # 'pitch_class': pitch_class,
    #                         # 'midi_note': msg.note
    #                         'pitch': msg.note,
    #                         'start_time': note_data['start_time'],
    #                         'duration': duration,
    #                         'velocity': note_data['velocity'],
    #                         'end_time': current_time
    #                     })
    #                     del active_notes[msg.note]
        
    #         # # Sort notes by start time
    #         # track_notes.sort(key=lambda x: x['start_time'])

    #         if track_notes:
    #             tracks_notes.append(track_notes)
                
    #     if not tracks_notes:
    #         return []

        
    #     # Score each track and select the one most likely to be melody
    #     track_scores = [analyze_track(track) for track in tracks_notes]
    #     melody_track = tracks_notes[track_scores.index(max(track_scores))]

    #     # Convert to our standard format
    #     melody_events = []
    #     for note in sorted(melody_track, key=lambda x: x['start_time']): # melody_track:
    #         melody_events.append({
    #             'time': note['start_time'],
    #             'duration': note['duration'],
    #             'pitch_class': note['pitch'] % 12,
    #             'midi_note': note['pitch']
    #         })
        
        
    #     # return sorted(events, key=lambda x: x['time'])

    #     return melody_events



    def prepare_training_data(self, midi_folder):
        """Prepare training data for PyTorch model"""
        all_events = []
        
        for filename in os.listdir(midi_folder):
            if filename.endswith(('.mid', '.midi')):
                try:
                    events = self.extract_melody_and_rhythm(os.path.join(midi_folder, filename))
                    all_events.extend(events)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        if not all_events:
            raise ValueError("No events extracted from MIDI files")
        
        max_duration = max(event['duration'] for event in all_events)
        sequence_length = 8
        
        input_sequences = []
        target_pitches = []
        target_durations = []
        
        for i in range(len(all_events) - sequence_length - 1):
            event_sequence = all_events[i:i + sequence_length]
            target_event = all_events[i + sequence_length]
            
            sequence_features = np.zeros((sequence_length, 14))  # Pre-allocate array for efficiency
            for j in range(sequence_length):
                current_event = event_sequence[j]
                next_event = event_sequence[j + 1] if j < sequence_length - 1 else target_event
                
                pitch_one_hot = np.zeros(12)
                pitch_one_hot[current_event['pitch_class']] = 1
                
                norm_duration = current_event['duration'] / max_duration
                gap = next_event['time'] - (current_event['time'] + current_event['duration'])
                norm_gap = min(gap / max_duration, 1.0)
                
                sequence_features[j, :12] = pitch_one_hot
                sequence_features[j, 12] = norm_duration
                sequence_features[j, 13] = norm_gap
            
            input_sequences.append(sequence_features)
            target_pitches.append(target_event['pitch_class'])
            target_durations.append(target_event['duration'] / max_duration)
        
        return np.array(input_sequences), np.array(target_pitches), np.array(target_durations)
    
    def train(self, midi_folder, epochs=50, batch_size=16, learning_rate=0.001):
        """Train the model using PyTorch"""
        X, y_pitch, y_duration = self.prepare_training_data(midi_folder)
        
        dataset = MelodyDataset(X, y_pitch, y_duration)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if self.model is None:
            self.model = MelodyModel(X.shape[2]).to(self.device)
        
        pitch_criterion = nn.CrossEntropyLoss()
        duration_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y_pitch, batch_y_duration in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y_pitch = batch_y_pitch.to(self.device)
                batch_y_duration = batch_y_duration.to(self.device)
                
                optimizer.zero_grad()
                pitch_pred, duration_pred = self.model(batch_X)
                
                pitch_loss = pitch_criterion(pitch_pred, batch_y_pitch)
                duration_loss = duration_criterion(duration_pred.squeeze(), batch_y_duration)
                
                loss = pitch_loss + duration_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
        
    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load a trained model"""
        if self.model is None:
            self.model = MelodyModel().to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()



    def generate_melody(self, rhythm_pattern, seed_sequence=None):
        """Generate melody following the rhythm pattern"""
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        self.model.eval()
        feature_dim = 14  # 12 pitch classes + duration + gap
        
        if seed_sequence is None:
            seed_sequence = np.zeros((8, feature_dim))
            seed_sequence[:, 0] = 1  # Start with C pitch class
            seed_sequence[:, -2:] = 0.25  # Default duration and gap
        
        sequence = torch.FloatTensor(seed_sequence).unsqueeze(0).to(self.device)
        melody = []
        durations = []
        
        with torch.no_grad():
            for is_active in rhythm_pattern:
                if is_active:
                    pitch_pred, duration_pred = self.model(sequence)
                    
                    pitch_probs = torch.softmax(pitch_pred, dim=1)
                    pitch_class = torch.multinomial(pitch_probs, 1).item()
                    duration = duration_pred.item()
                    
                    melody.append(pitch_class)
                    durations.append(duration)
                    
                    # Update sequence
                    new_event = torch.zeros(feature_dim)
                    new_event[pitch_class] = 1
                    new_event[-2:] = torch.tensor([duration, 0.25])
                    
                    sequence = sequence.roll(-1, dims=1)
                    sequence[0, -1] = new_event
                else:
                    melody.append(-1)
                    durations.append(0)
        
        return melody, durations


def create_rhythm_midi(rhythm_pattern, output_file="rhythm.mid", tempo=120):
    """Convert rhythm pattern to MIDI file"""
    from midiutil import MIDIFile
    
    midi = MIDIFile(1)
    track = 0
    channel = 9  # Standard MIDI drum channel
    time = 0
    volume = 100
    
    midi.addTempo(track, time, tempo)
    
    for i, hit in enumerate(rhythm_pattern):
        if hit == 1:
            time = i * 0.25  # 16th note duration
            duration = 0.25
            midi.addNote(track, channel, 36, time, duration, volume)  # 36 = Bass Drum
    
    with open(output_file, 'wb') as f:
        midi.writeFile(f) 


def save_midi(melody, durations, output_file="generated_melody.mid", base_note=60):
    """Save generated melody to MIDI file"""
    from midiutil import MIDIFile
    
    midi = MIDIFile(1)
    track, channel, time = 0, 0, 0
    tempo = 120
    volume = 100
    
    midi.addTempo(track, time, tempo)
    current_time = 0
    
    for pitch, duration in zip(melody, durations):
        if pitch >= 0:
            beat_duration = max(0.125, duration * 4)
            midi.addNote(track, channel, base_note + pitch, current_time, beat_duration, volume)
        current_time += 0.25
    
    with open(output_file, 'wb') as f:
        midi.writeFile(f)


def main():


    midi_folder = '../midi_dataset/piano_maestro/piano_maestro-v1.0.0/all_years/'

    rythm_model_path = './models/rhythm_model.pth'
    melody_model_path = './models/melody_model.pth'

    mode = 'train'
    what = 'melody'

    num_epochs = 20

    if mode == 'train':

        if what == 'rhythm':
            rhythm_learner = RhythmLearner()
            rhythm_learner.train(midi_folder, epochs=num_epochs)
            rhythm_learner.save_model(rythm_model_path)

        elif what == 'melody':
            melody_learner = MelodyLearner()
            melody_learner.train(midi_folder, epochs=num_epochs)
            melody_learner.save_model(melody_model_path)

    elif mode == 'generate':
        
        if what == 'rhythm':
            rhythm_learner = RhythmLearner()
            rhythm_learner.load_model(rythm_model_path)
            rhythm_pattern = rhythm_learner.generate_rhythm(bars=16)
            create_rhythm_midi(rhythm_pattern, "generated_rhythm.mid")
        
        elif what == 'melody':
           
            rhythm_learner = RhythmLearner()
            rhythm_learner.load_model(rythm_model_path)

            melody_learner = MelodyLearner()
            melody_learner.load_model(melody_model_path)

            rhythm_pattern = rhythm_learner.generate_rhythm(bars=4)
            melody, durations = melody_learner.generate_melody(rhythm_pattern)
            save_midi(melody, durations, "generated_melody.mid")
        

if __name__ == '__main__':
    main()