import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import argparse

# -------- Dataset -------- #
class EngToFrenchDataset(Dataset):
    def __init__(self, data_path, max_len=20):
        self.df = pd.read_csv(data_path)
        self.max_len = max_len

        self.src_vocab = self.build_vocab(self.df['English words/sentences'])
        self.tgt_vocab = self.build_vocab(self.df['French words/sentences'])
        self.rev_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}

        self.data = [
            (self.encode(src, self.src_vocab), self.encode(tgt, self.tgt_vocab))
            for src, tgt in zip(self.df['English words/sentences'], self.df['French words/sentences'])
        ]

    def build_vocab(self, sentences):
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        idx = 4
        for sentence in sentences:
            for word in sentence.strip().split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

    def encode(self, sentence, vocab):
        tokens = sentence.strip().split()
        ids = [vocab.get(w, vocab['<unk>']) for w in tokens]
        return torch.tensor([vocab['<sos>']] + ids + [vocab['<eos>']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

# -------- Positional Encoding -------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# -------- Transformer Model -------- #
class TransformerEngToFrench(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=100, nhead=4, num_layers=2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.src_embedding(src))  # (batch, src_len, d_model)
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt))  # (batch, tgt_len, d_model)

        tgt_mask = generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)

        memory = self.transformer.encoder(src_emb)

        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )   
        return self.output_layer(output)

# -------- Training Function -------- #
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# -------- Translation Function -------- #
def translate(model, sentence, dataset, device, max_len=20):
    model.eval()
    tokens = sentence.strip().split()
    src = torch.tensor([dataset.src_vocab.get(w, dataset.src_vocab['<unk>']) for w in tokens])
    src = torch.cat([torch.tensor([dataset.src_vocab['<sos>']]), src, torch.tensor([dataset.src_vocab['<eos>']])])
    src = src.unsqueeze(0).to(device)  # (1, src_len)

    tgt = torch.tensor([[dataset.tgt_vocab['<sos>']]]).to(device)  # (1, 1)

    for _ in range(max_len):
        output = model(src, tgt)
        next_token = output[:, -1, :].argmax(dim=-1).item()
        if next_token == dataset.tgt_vocab['<eos>']:
            break
        tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)

    translated = [dataset.rev_tgt_vocab[idx.item()] for idx in tgt[0, 1:]]
    return ' '.join(translated)

# -------- Main -------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(os.path.dirname(__file__), 'Input/eng_-french.csv')

    parser = argparse.ArgumentParser(description="Train or load Transformer model")
    parser.add_argument('--retrain', action='store_true', help='Retrain the model even if a checkpoint exists')
    args = parser.parse_args()

    dataset = EngToFrenchDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = TransformerEngToFrench(
        src_vocab_size=len(dataset.src_vocab),
        tgt_vocab_size=len(dataset.tgt_vocab),
        d_model=100
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model_path = "Outputs/transformer_epoch_10.pt"

    if os.path.exists(model_path) and not args.retrain:
        print("Loading saved model...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    else:
        print("Training model...")
        for epoch in range(10):
            loss = train(model, dataloader, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'Outputs/transformer_epoch_{epoch+1}.pt')

    while True:
        user_input = input("\nEnter a sentence to translate (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        translation = translate(model, user_input, dataset, device)
        print("Translation:", translation)