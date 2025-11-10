import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
from pathlib import Path

class Preprocess:
    def __init__(self):
        self.path = Path(__file__).resolve().parent.parent / "Data" / "TdcMaskPost.csv"
        self.df = self.load_csv()

    def load_csv(self):
        df = pd.read_csv(self.path)
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
        return df

    def padding(self):
        sequences = []
        lengths = []

        for _, row in self.df.iterrows():
            seq_id = row['ID']
            seq_values = row.drop('ID').values.astype(float)
            seq_tensor = torch.tensor(seq_values, dtype=torch.float).unsqueeze(-1)
            sequences.append(seq_tensor)
            lengths.append(seq_tensor.size(0))
        padded = pad_sequence(sequences, batch_first=True)

        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        return packed


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
from pathlib import Path



class Preprocess:
    def __init__(self, csv_path):
        self.path = Path(csv_path)
        self.df = self.load_csv()

    def load_csv(self):
        df = pd.read_csv(self.path)
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
        return df

    def padding(self):
        sequences = []
        lengths = []

        for i in range(len(self.df)):
            seq_values = self.df.iloc[i, 1:].values.astype(float)  # skip ID
            seq_tensor = torch.tensor(seq_values, dtype=torch.float).unsqueeze(-1)  # (seq_len, 1)
            sequences.append(seq_tensor)
            lengths.append(seq_tensor.size(0))

        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        return packed, lengths, padded  # padded needed for teacher forcing

class EncoderRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, packed_seq):
        packed_output, h_n = self.rnn(packed_seq)
        return h_n  # RNN has only hidden state


class DecoderRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, hidden):
        out, hidden = self.rnn(decoder_input, hidden)
        out = self.fc(out)  # raw logits for binary classification
        return out, hidden



class Seq2SeqBinaryRNN(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing_ratio=0.5):
        super(Seq2SeqBinaryRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, packed_input, target=None):
        h_n = self.encoder(packed_input)  # encoder hidden
        hidden = h_n

        batch_size = h_n.size(1)
        if target is not None:
            target_len = target.size(1)
        else:
            target_len = 10

        decoder_input = torch.zeros(batch_size, 1, 1)
        outputs = []

        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            outputs.append(out)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = out

        outputs = torch.cat(outputs, dim=1)
        return outputs



csv_path = "Data/TdcMaskPost.csv"


preprocessor = Preprocess(csv_path)
packed_seq, lengths, padded_targets = preprocessor.padding()


padded_targets = (padded_targets > 0.5).float()  # convert to 0/1 if necessary


encoder_rnn = EncoderRNN(input_size=1, hidden_size=16)
decoder_rnn = DecoderRNN(input_size=1, hidden_size=16, output_size=1)
seq2seq_model = Seq2SeqBinaryRNN(encoder_rnn, decoder_rnn, teacher_forcing_ratio=0.5)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=0.001)


seq2seq_model.train()
optimizer.zero_grad()

output_logits = seq2seq_model(packed_seq, target=padded_targets)  # raw logits

mask = torch.zeros_like(padded_targets)
for i, l in enumerate(lengths):
    mask[i, :l, :] = 1

loss = criterion(output_logits * mask, padded_targets * mask)
loss.backward()
optimizer.step()

print("Training step done. Loss:", loss.item())


seq2seq_model.eval()
with torch.no_grad():
    logits = seq2seq_model(packed_seq)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

print("Predicted binary sequences shape:", preds.shape)



