# %%
from lib2to3.pgen2 import token
import os 
import pickle 
import sys
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pygame 
import time
from tqdm import tqdm  # for progress bar

import pretty_midi
import librosa

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from midi_transform import MIDIDataset, create_dataloaders,  MusicTransformer, train_transformer , generate_music


# %%
## load 
def load_midi(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    return midi_data



def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))



'''
def encode_midi(midi_data, time_step=0.025):
    events = []
    for track_idx, instrument in enumerate(midi_data.instruments):
        print(instrument)
        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            pitch = note.pitch
            velocity = note.velocity
            
            # Encode note-on event with track information
            events.append((start_time, 'note_on', track_idx, pitch, velocity))
            # Encode note-off event with track information
            events.append((end_time, 'note_off', track_idx, pitch, velocity))
    
    # Sort events by time
    events.sort(key=lambda x: x[0])
    
    # Quantize events to fixed time steps
    quantized_events = []
    current_time = 0
    for event in events:
        time_diff = int((event[0] - current_time) / time_step)
        if time_diff > 0:
            quantized_events.append(('time_shift', time_diff))
        quantized_events.append(event[1:])
        current_time = event[0]
    
    return quantized_events

def create_vocabulary(encoded_data):
    vocab = set()
    for event in encoded_data:
        if event[0] == 'time_shift':
            vocab.add(f"time_shift_{event[1]}")
        else:
            vocab.add(f"{event[0]}_{event[1]}_{event[2]}_{event[3]}")
    return {token: idx for idx, token in enumerate(sorted(vocab))}

def tokenize_events(encoded_data, vocab):
    tokens = []
    for event in encoded_data:
        if event[0] == 'time_shift':
            token = f"time_shift_{event[1]}"
        else:
            token = f"{event[0]}_{event[1]}_{event[2]}_{event[3]}"
        tokens.append(vocab[token])
    return tokens
''' 

# %%
def encode_midi(midi_data, time_step=0.025):
    events = []
    for track_idx, instrument in enumerate(midi_data.instruments):
        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            pitch = note.pitch
            velocity = note.velocity
            
            # Encode note-on event with absolute time
            events.append((start_time, 'note_on', track_idx, pitch, velocity))
            # Encode note-off event with absolute time
            events.append((end_time, 'note_off', track_idx, pitch, velocity))
    
    # Sort events by time
    events.sort(key=lambda x: x[0])
    
    return events

def events_to_sequence(events, time_step=0.025):
    sequence = []
    current_time = 0
    
    for event in events:
        event_time, event_type, track_idx, pitch, velocity = event
        
        # Add time shift if necessary
        time_diff = int((event_time - current_time) / time_step)
        if time_diff > 0:
            sequence.append(('time_shift', time_diff))
            current_time += time_diff * time_step
        
        # Add the event
        sequence.append((event_type, track_idx, pitch, velocity))
    
    return sequence


def create_vocabulary(sequence):
    vocab = set()
    print(len(sequence))
    for event in sequence:
        if event[0] == 'time_shift':
            vocab.add(f"time_shift_{event[1]}")
        else:
            # print('event', len(event))
            event_type, track_idx, pitch, velocity = event
            vocab.add(f"{event_type}_{track_idx}_{pitch}_{velocity}")
    return {token: idx for idx, token in enumerate(sorted(vocab))}

def tokenize_sequence(sequence, vocab):
    tokens = []
    for event in sequence:
        # print('len event', len(event))
        if event[0] == 'time_shift':
            token = f"time_shift_{event[1]}"
        else:
            event_type, track_idx, pitch, velocity = event
            token = f"{event_type}_{track_idx}_{pitch}_{velocity}"
        tokens.append(vocab[token])
    return tokens


# %% Separate melody 

def extract_melody(midi_data):

    # Combine all notes from all instruments
    all_notes = []
    for instrument in midi_data.instruments:
        all_notes.extend(instrument.notes)

    # Sort notes by start time
    all_notes.sort(key=lambda x: x.start)

    # Heuristics for melody extraction
    melody_notes = []
    current_time = 0
    for note in all_notes:
        # Skip very short notes (adjust threshold as needed)
        if note.end - note.start < 0.1:
            continue
        
        # If this note starts after the previous note ended, add it to melody
        if note.start >= current_time:
            melody_notes.append(note)
            current_time = note.end
        # If this note is higher pitched than the current melody note, replace it
        elif note.pitch > melody_notes[-1].pitch and note.start < melody_notes[-1].end:
            melody_notes[-1].end = note.start
            melody_notes.append(note)
            current_time = note.end

    # Create a new MIDI file with just the melody
    melody_midi = pretty_midi.PrettyMIDI()
    melody_instrument = pretty_midi.Instrument(program=0)  # Piano
    melody_instrument.notes = melody_notes
    melody_midi.instruments.append(melody_instrument)

    print(melody_midi)

    return melody_midi


# from collections import defaultdict

# def is_monophonic(instrument):
#     """Check if an instrument is mostly monophonic."""
#     notes = sorted(instrument.notes, key=lambda x: x.start)
#     overlaps = 0
#     total_notes = len(notes)
#     for i in range(1, total_notes):
#         if notes[i].start < notes[i-1].end:
#             overlaps += 1
#     return overlaps / total_notes < 0.1  # Allow 10% overlap for some grace

# def get_pitch_range(notes):
#     """Calculate the pitch range of a set of notes."""
#     pitches = [note.pitch for note in notes]
#     return max(pitches) - min(pitches)

# def calculate_melody_score(notes):
#     """Calculate a melody score based on various metrics."""
#     pitch_range = get_pitch_range(notes)
#     avg_duration = np.mean([note.end - note.start for note in notes])
#     pitch_variance = np.var([note.pitch for note in notes])
    
#     # Favor instruments with a good pitch range, moderate note duration, and some pitch variance
#     return pitch_range * 0.5 + (1 / avg_duration) * 0.3 + pitch_variance * 0.2

# def extract_melody_advanced(midi_data):
 
#     # Filter for monophonic instruments
#     monophonic_instruments = [inst for inst in midi_data.instruments if is_monophonic(inst)]

#     if not monophonic_instruments:
#         print("No suitable monophonic instruments found.")
#         return

#     # Calculate melody scores for each monophonic instrument
#     melody_scores = [(calculate_melody_score(inst.notes), inst) for inst in monophonic_instruments]

#     # Select the instrument with the highest melody score
#     best_instrument = max(melody_scores, key=lambda x: x[0])[1]

#     # Create a new MIDI file with just the selected melody
#     melody_midi = pretty_midi.PrettyMIDI()
#     melody_instrument = pretty_midi.Instrument(program=best_instrument.program)
#     melody_instrument.notes = best_instrument.notes
#     melody_midi.instruments.append(melody_instrument)


#     return melody_midi


# %% Detokenize 
def detokenize_sequence(tokens, vocab_reverse):
    sequence = []
    for token in tokens:
        if token not in vocab_reverse:
            raise KeyError(f"Unknown token: {token}")
        event = vocab_reverse[token]
        if event.startswith('time_shift'):
            # print(event.split('_')[2])
            sequence.append(('time_shift', int(event.split('_')[2]) ) )
        else:
            event_parts = event.split('_')
            # print('event_parts', event_parts)
            note_on_off = event_parts[0] + '_' + event_parts[1]
            sequence.append((note_on_off, int(event_parts[2]), int(event_parts[3]), int(event_parts[4]) ))
    return sequence


def sequence_to_midi(sequence, time_step=0.025):
    midi = pretty_midi.PrettyMIDI()
    current_time = 0
    tracks = {}

    for event in sequence:
        if event[0] == 'time_shift':
            current_time += event[1] * time_step
        else:
            event_type, track_idx, pitch, velocity = event
            if track_idx not in tracks:
                # Use a different program number for each track
                program = track_idx % 128  # Cycle through available MIDI programs
                instrument = pretty_midi.Instrument(program=program)
                tracks[track_idx] = instrument
                midi.instruments.append(instrument)
                print('instrument', instrument)

            if event_type == 'note_on':
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=current_time,
                    end=current_time  # We'll set the end time when we find the note_off event
                )
                tracks[track_idx].notes.append(note)
            elif event_type == 'note_off':
                # Find the corresponding note_on and set its end time
                for note in reversed(tracks[track_idx].notes):
                    if note.pitch == pitch and note.end == note.start:
                        note.end = current_time
                        break

    return midi

def play_midi(midi_data):
    pygame.mixer.init()
    pygame.mixer.music.load(midi_data)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.mixer.quit()

def generate_and_play_midi(generated_tokens, vocab, time_step=0.025):
    # Create reverse vocabulary for detokenization
    vocab_reverse = {idx: token for token, idx in vocab.items()}
    # print('>>>>>', vocab_reverse)

    # Detokenize the sequence
    try:
        sequence = detokenize_sequence(generated_tokens, vocab_reverse)
    except KeyError as e:
        print(f"Error during detokenization: {e}")
        print("This could be due to an unknown token. Check if all generated tokens are in the vocabulary.")
        return

    # Print first few events for debugging
    print("First few events after detokenization:")
    print(sequence[:10])

    # Convert sequence to MIDI
    try:
        midi_data = sequence_to_midi(sequence, time_step)
    except Exception as e:
        print(f"Error during MIDI conversion: {e}")
        return

    return midi_data

# Example usage
# Assuming you have generated_tokens from your model and the original vocab
# generated_tokens = [...] # Your model's output
# vocab = {...} # The vocabulary used for tokenization

# generate_and_play_midi(generated_tokens, vocab)   


def process_midi_folder(input_folder, output_folder, time_step=0.025):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_sequences = []
    file_paths = []

    # First pass: load and encode all MIDI files
    for filename in tqdm(os.listdir(input_folder), desc="Loading MIDI files"):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(input_folder, filename)
            midi_data = load_midi(file_path)
            if midi_data is not None:
                events = encode_midi(midi_data, time_step)
                sequence = events_to_sequence(events, time_step)
                print('sequence:>>>>>>>', type(sequence), len(sequence))
                all_sequences.extend(sequence)
                file_paths.append( (file_path, len(sequence)) )
    
    print('....', len(all_sequences))
    # Create vocabulary from all sequences
    vocab = create_vocabulary(all_sequences)

    # Save vocabulary
    with open(os.path.join(output_folder, 'vocabulary.pkl'), 'wb') as f:
        pickle.dump(vocab, f)


     # Tokenize all events
    all_tokens = tokenize_sequence(all_sequences, vocab)

    # Second pass: save tokenized sequences
    start_idx = 0
    for file_path, seq_length in tqdm(file_paths, desc="Saving tokenized sequences"):
        end_idx = start_idx + seq_length
        tokens = all_tokens[start_idx:end_idx]
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + '.npy'
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, np.array(tokens))
        start_idx = end_idx

    # # Second pass: tokenize and save each sequence
    # for sequence, file_path in tqdm(zip(all_sequences, file_paths), desc="Tokenizing and saving", total=len(all_sequences)):
    #     tokens = tokenize_sequence(sequence, vocab)
    #     output_filename = os.path.splitext(os.path.basename(file_path))[0] + '.npy'
    #     output_path = os.path.join(output_folder, output_filename)
    #     np.save(output_path, np.array(tokens))

    print(f"Processed {len(all_sequences)} MIDI files. Tokenized data saved in {output_folder}")
    print(f"Vocabulary size: {len(vocab)}")

# %%

if __name__=='__main__':

    mode = 'train' #  'preprocess' #'train' #'play' #'train'

    input_folder = '../piano_maestro-v2.0.0/2004/' # './midi_dataset/' #
    output_folder = './gen_midi_dataset' # './processed_tokens/'


    if mode == 'play':
        # Example usage
        input_midi_file ='midi_dataset/amadbahra.mid' #bavarkon.mid'
        midi_data = load_midi(input_midi_file)
        # midi_data = load_midi('midi_dataset/Lady_In_Red__Wrubel__Milne_1935_AB.mid')

        # plt.figure(figsize=(8, 4))
        # plot_piano_roll(midi_data, 56, 70)
        # plt.show()

        time_step=0.025
        events = encode_midi(midi_data, time_step)
        sequence = events_to_sequence(events, time_step)
        vocab = create_vocabulary(sequence)
        tokens = tokenize_sequence(sequence, vocab)

        print('sequence >>>>' , sequence[:30] ) 
        # print(vocab)
        print('tokens >>>>', tokens[:30] )
        ### generate midi from tokens
        gen_midi_data = generate_and_play_midi(tokens, vocab, time_step=time_step)

        # Save MIDI to a temporary file
        temp_midi_file = 'gen_midi_dataset/'+ input_midi_file.split('/')[-1].split('.')[0]+'_gen.mid'
        gen_midi_data.write(temp_midi_file)

        # ### extract the melody 
        # # # Write the melody MIDI file
        # melody_midi = extract_melody(midi_data)
        # sep_melody = 'melody_separated.mid'
        # melody_midi.write(sep_melody)

        # # Play the MIDI
        try:
            play_midi(temp_midi_file)
            # play_midi(sep_melody)
        except Exception as e:
            print(f"Error during MIDI playback: {e}")

    elif mode == 'preprocess':
        process_midi_folder(input_folder, output_folder, time_step=0.025)


    elif mode == 'train' :
        ### first load tokens 
        ## TODO 

        ### 
        token_folder = './gen_midi_dataset'

        with open(os.path.join(token_folder, 'vocabulary.pkl'), 'rb') as f:
            vocab = pickle.load(f)
        
        # Create dataloaders
        seq_length=512
        batch_size=32

        train_loader, val_loader = create_dataloaders(token_folder, seq_length, batch_size)


        #### Transformer train 
        # # Assuming your tokenized_sequences is a list of token IDs
        # dataset = MIDIDataset(tokens, seq_length=512)
        # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize the model
        vocab_size = len(vocab)  # vocab is your vocabulary from tokenization
        model = MusicTransformer(vocab_size, d_model=512, nhead=8, num_layers=6)    

        train_transformer(model, train_loader, vocab_size)


    elif mode == 'inference' :
        ### load model 
        model = torch.load('music_transformer.pth')
        # Generate a new sequence
        start_seq = tokens[0][:10]  # Use the first 10 tokens of a sequence as a start
        generated_sequence = generate_music(model, start_seq)

        # Convert the generated sequence back to MIDI
        # Use your detokenization and MIDI conversion functions here

# %%
