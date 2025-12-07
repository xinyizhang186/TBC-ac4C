import torch
import numpy as np
from model import TBC_ac4C
from util.data_loader import load_data, transform_token2index
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
predict.py is designed to use the trained TBC-ac4C model for predicting ac4C modification sites in RNA sequences, and outputs prediction labels
and probabilities.
"""

def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model

def predict(model, data_loader, device):
    model.eval()
    probabilities = []
    label_pred = []

    with torch.no_grad():
        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)

            outputs = model(x)
            label_pred = label_pred + outputs.argmax(dim=1).tolist()
            probabilities.extend(outputs)
    return np.array(label_pred), np.array(probabilities)

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


if __name__ == '__main__':
    file = 'dataset/iRNA-ac4C/ac4c-testset.txt'
    path_pretrain_model = "model/TBC-ac4C.pt"

    seqs = load_data(file)
    seqs = list(seqs)
    token_list, max_len = transform_token2index(seqs)
    seqs_data = rnn_utils.pad_sequence(token_list, batch_first=True)  # Fill the sequence to the same length
    data_loader = Data.DataLoader(MyDataSet(seqs_data), batch_size=64, shuffle=False, drop_last=False)

    print(f"Loading model from {path_pretrain_model}")
    model = TBC_ac4C().to(device)
    model = load_model(model, path_pretrain_model)

    print("Starting prediction...")
    label_pred, probabilities = predict(model, data_loader, device)
    print(f'label_pred: {label_pred[:10]}')
    print(f'probabilities: {probabilities[:10]}')


