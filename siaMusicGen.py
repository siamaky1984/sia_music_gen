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

from load_midi import generate_and_play_midi, extract_melody, load_midi, play_midi, tokenize_sequence,\
      events_to_sequence, create_vocabulary, encode_midi, sequence_to_midi ,  plot_piano_roll
from midi_transform import MIDIDataset, create_dataloaders,  MusicTransformer, train_transformer 
import argparse


# %%

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

    # Create parser
    parser = argparse.ArgumentParser(description='Music Generation')

    # Add arguments
    parser.add_argument('--mode', type=str, default='play', help='Mode of operation: play, inference, train, preprocess')
    parser.add_argument('--input_folder', type=str, default='../MIDI_dataset/maestro-v2.0.0/2004/', help='Input folder for MIDI files')
    parser.add_argument('--output_folder', type=str, default='./gen_midi_dataset', help='Output folder for processed MIDI files')

    parser.add_argument('--target_seq_length', type=int, default=1000, help='Target sequence length for inference')

    # Parse arguments
    args = parser.parse_args()

    # Assign arguments to variables
    mode = args.mode
    input_folder = args.input_folder
    output_folder = args.output_folder

    # mode = 'play' # 'inference' # 'train' #  'preprocess' #'train' #'play' #'train'

    # input_folder = '../MIDI_dataset/maestro-v2.0.0/2004/' # './midi_dataset/' #
    # output_folder = './gen_midi_dataset' # './processed_tokens/'


    if mode == 'play':
        # Example usage
        input_midi_file ='midi_dataset/appenzel.mid'  #'midi_dataset/amadbahra.mid' #bavarkon.mid'
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

        input_midi_file ='MIDI-Unprocessed_XP_08_R1_2004_01-02_ORIG_MID--AUDIO_08_R1_2004_02_Track02_wav.npy'
        token_folder = './gen_midi_dataset/'
        tokens = np.load( token_folder + input_midi_file )

        # Convert numpy array to torch tensor
        tokens_tensor = torch.from_numpy(tokens).unsqueeze(0).cuda()
        print('tokens_tensor', tokens_tensor.shape)

        with open(os.path.join(token_folder, 'vocabulary.pkl'), 'rb') as f:
            vocab = pickle.load(f)
        # Initialize the model
        vocab_size = len(vocab)  # vocab is your vocabulary from tokenization
        print('vocab_size', vocab_size)
        
        ### load model 
        model = MusicTransformer(vocab_size, d_model=512, nhead=8, num_layers=6).cuda()   
        model.load_state_dict( torch.load('./gen_models/music_transformer.pth') )
        print('model', type(model) )
        # Generate a new sequence
        ### 

        print('tokens', tokens_tensor[:,0:10])
        start_seq = tokens_tensor[:,:10]  # Use the first 10 tokens of a sequence as a start
        # generated_sequence = generate_seq(model, start_seq)

        generated_sequence = model.generate(start_seq, target_seq_length= args.target_seq_length ) 

        print(';en', generated_sequence.size )

        # Convert the generated sequence to a numpy array
        generated_sequence_np = generated_sequence.cpu().numpy().squeeze() ### this is NECESSARY to numpy and squeeze

        # Convert the generated sequence back to MIDI
        # Use your detokenization and MIDI conversion functions here
        gen_midi_data = generate_and_play_midi( generated_sequence_np, vocab, time_step=0.025)
        # Save MIDI to a temporary file
        temp_midi_file = input_midi_file +'_gen.mid'
        gen_midi_data.write(temp_midi_file)
        
        # # Play the MIDI
        try:
            play_midi(temp_midi_file)
            # play_midi(sep_melody)
        except Exception as e:
            print(f"Error during MIDI playback: {e}")

# %%
