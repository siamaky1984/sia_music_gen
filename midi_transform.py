#### transofrmer 

# %%
import numpy as np
import os 
from tqdm import tqdm  # for progress bar


import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim



# %%
# class MIDIDataset(Dataset):
#     def __init__(self, tokenized_sequences, seq_length):
#         self.data = tokenized_sequences
#         self.seq_length = seq_length

#     def __len__(self):
#         return len(self.data) - self.seq_length

#     def __getitem__(self, idx):
#         return (torch.tensor(self.data[idx:idx+self.seq_length]),
#                 torch.tensor(self.data[idx+1:idx+self.seq_length+1]))


class MIDIDataset(Dataset):
    def __init__(self, folder_path, seq_length):
        self.folder_path = folder_path
        self.seq_length = seq_length
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        self.data = self.load_all_data()

    def load_all_data(self):
        all_data = []
        for file in tqdm(self.file_list, desc="Loading tokenized files"):
            data = np.load(os.path.join(self.folder_path, file))
            all_data.extend(data)
        return all_data

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # return torch.tensor(self.data[idx:idx+self.seq_length]), torch.tensor(self.data[idx+1:idx+self.seq_length+1])
        return (torch.tensor(self.data[idx:idx+self.seq_length]), torch.tensor(self.data[idx+1:idx+self.seq_length+1]) )


def create_dataloaders(folder_path, seq_length, batch_size, train_split=0.8):
    dataset = MIDIDataset(folder_path, seq_length)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
# %%

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)  # Adjust 1000 if you have longer sequences
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.pos_encoder(torch.arange(src.size(1), device=src.device))
        tgt = self.embedding(tgt) + self.pos_encoder(torch.arange(tgt.size(1), device=tgt.device))
        output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1)).transpose(0, 1)
        return self.fc_out(output)


    def generate(self, start_seq, target_seq_length=1000):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gen_seq = torch.full((1,target_seq_length), -1, dtype=torch.long , device= device) 
        # gen_seq =  torch.tensor(start_seq).unsqueeze(0).to(device)

        print('gen size', gen_seq.size())
        print('start_sq', start_seq.size())

        num_primer = start_seq.size()[1]
        print(num_primer)
        gen_seq[..., :num_primer] = start_seq

        cur_i = num_primer

        while cur_i < target_seq_length:
            with torch.no_grad():
                pred = self.forward(gen_seq[..., :cur_i], gen_seq[..., :cur_i])
            
            # print('>>>', pred.size())

            pred = pred[0, cur_i-1, :].argmax().unsqueeze(0)
            # print('next token', pred)
            gen_seq[..., cur_i] = pred
            cur_i += 1

        return gen_seq[:, :cur_i]

# %% 

def train_transformer(model, dataloader, vocab_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)

            print('>>>', tgt.size())
            print('>>>>', src.size())
            
            optimizer.zero_grad()
            output = model(src,tgt) # model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, vocab_size), tgt.reshape(-1))
            # loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        
        print(f'Epoch {epoch}, Average Loss: {total_loss / len(dataloader)}')

    # Save the model
    torch.save(model.state_dict(), 'music_transformer.pth')


# %%
# def generate_music(model, start_seq, max_length=1000):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model.eval()
#     with torch.no_grad():
#         current_seq = torch.tensor(start_seq).unsqueeze(0).to(device)
#         for _ in range(max_length - len(start_seq)):
#             output = model(current_seq, current_seq)
#             next_token = output[:, -1, :].argmax(dim=-1)
#             current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
#         return current_seq.squeeze().cpu().numpy()

