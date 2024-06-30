# %%
import pretty_midi

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import librosa

import pygame 
import time

import sys

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
    for event in sequence:
        if event[0] == 'time_shift':
            vocab.add(f"time_shift_{event[1]}")
        else:
            event_type, track_idx, pitch, velocity = event
            vocab.add(f"{event_type}_{track_idx}_{pitch}_{velocity}")
    return {token: idx for idx, token in enumerate(sorted(vocab))}

def tokenize_sequence(sequence, vocab):
    tokens = []
    for event in sequence:
        if event[0] == 'time_shift':
            token = f"time_shift_{event[1]}"
        else:
            event_type, track_idx, pitch, velocity = event
            token = f"{event_type}_{track_idx}_{pitch}_{velocity}"
        tokens.append(vocab[token])
    return tokens




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

# def sequence_to_midi(sequence, time_step=0.025):
#     midi = pretty_midi.PrettyMIDI()
#     current_time = 0
#     tracks = {}

#     for event in sequence:
#         if event[0] == 'time_shift':
#             current_time += event[1] * time_step
#         else:
#             event_type, track_idx, pitch, velocity = event
#             if track_idx not in tracks:
#                 instrument = pretty_midi.Instrument(program=0)  # Default to piano
#                 tracks[track_idx] = instrument
#                 midi.instruments.append(instrument)

#             if event_type == 'note_on':
#                 note = pretty_midi.Note(
#                     velocity=velocity,
#                     pitch=pitch,
#                     start=current_time,
#                     end=current_time  # We'll set the end time when we find the note_off event
#                 )
#                 tracks[track_idx].notes.append(note)
#             elif event_type == 'note_off':
#                 # Find the corresponding note_on and set its end time
#                 for note in reversed(tracks[track_idx].notes):
#                     if note.pitch == pitch and note.end == note.start:
#                         note.end = current_time
#                         break

#     return midi

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


# %%

if __name__=='__main__':

    # Example usage
    midi_data = load_midi('midi_dataset/bavarkon.mid')

    # plt.figure(figsize=(8, 4))
    # plot_piano_roll(midi_data, 56, 70)
    # plt.show()

    # encoded_data = encode_midi(midi_data)
    # print('encoded_Data', encoded_data)

    # # Example usage
    # vocab = create_vocabulary(encoded_data)
    # tokens = tokenize_events(encoded_data, vocab)
    # print(tokens)

    time_step=0.025
    events = encode_midi(midi_data, time_step)
    sequence = events_to_sequence(events, time_step)
    vocab = create_vocabulary(sequence)
    tokens = tokenize_sequence(sequence, vocab)

    print('sequence >>>>' , sequence[:10] ) 
    # print(vocab[:,10])
    print('tokens >>>>', tokens[:10] )
    

    ### generate midi from tokens
    gen_midi_data = generate_and_play_midi(tokens, vocab, time_step=time_step)


    # Save MIDI to a temporary file
    temp_midi_file = 'bavar_kon_gen.mid'
    midi_data.write(temp_midi_file)

    # # Play the MIDI
    try:
        play_midi(temp_midi_file)
    except Exception as e:
        print(f"Error during MIDI playback: {e}")
# %%
