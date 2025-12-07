import torch
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

"""
data_loader.py provides a complete set of RNA sequence data processing and loading functions, capable of converting RNA sequences into 
numerical representations and constructing PyTorch data loaders.

Main functions:
  1. Load RNA sequence data in the iRNA-ac4C data: load_data(file).
  2. Load RNA sequence data in the Meta-ac4C data: load_Meta_data(file).
  3. Convert RNA sequences to numerical representation: transform_token2index(sequences).
  4. Construct batched data loaders: construct_dataset(seqs, labels, train=True, batch_size=64).
  5. Load benchmark datasets and independent test sets: load_bench_data(fasta_file, nums),  load_ind_data(fasta_file, nums), 
     load_bench_Metadata(pos_file,neg_file, pos_nums, neg_nums), load_ind_Metadata(pos_file,neg_file, pos_nums, neg_nums).
"""

def load_data(file):
    file = open(file, 'r')
    lines = file.readlines()
    sequences = []

    for line in lines:
        if line[0] != '>' and line[0] != '#':
            each_line = line.strip()  
            sequences.append(each_line)
    return sequences

    
def load_Meta_data(file):  
    data = np.loadtxt(file, dtype=list)
    sequences = []

    for seq in data:
        seq = seq.upper()  
        seq = str(seq.strip('\n'))
        sequences.append(seq)
    return sequences
    
def transform_token2index(sequences):
    """
    Convert RNA sequences to numerical representation
    Args:
        sequences: list of RNA sequences
    Returns:
        token_list: list of numerical sequence tensors
        max_len: maximum sequence length
    """
    
    token2index ={'A':1,'U':2,'T':2,'C':3,'G':4,'-':0,'X':0}  # Suppose "X" represents an unknown nucleotide, which is mapped to 0
    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    max_len = len(max(sequences,key=len))
    token_list = list()
    for seq in sequences:
        seq_id = [token2index[base] for base in seq]
        token_list.append(torch.tensor(seq_id))

    return token_list, max_len
    

def construct_dataset(seqs, labels, train=True, batch_size=64):
    """
    Construct dataset and return data loader
    Args:
        seqs: sequence data
        labels: label data
        train: whether it is training set
        batch_size: batch size
    Returns:
        data_loader: PyTorch data loader
    """
    seqs, labels = list(seqs), list(labels)
    token_list, max_len = transform_token2index(seqs)
    seqs_data = rnn_utils.pad_sequence(token_list, batch_first=True)  # Fill the sequence to the same length

    data_loader = Data.DataLoader(Data.TensorDataset(seqs_data, torch.LongTensor(labels)),
                                  batch_size=batch_size,
                                  shuffle=train, 
                                  drop_last=False)
    return data_loader
    
def load_bench_data(fasta_file,nums=4412): 
    seqs = load_data(fasta_file)
    labels = np.vstack((np.ones((int(nums / 2), 1), dtype=int), np.zeros((int(nums / 2), 1), dtype=int))).flatten()

    train_seqs, test_seqs, train_labels, test_labels = train_test_split(seqs, labels, test_size=0.2, random_state=42)
    train_iter = construct_dataset(train_seqs, train_labels, train=True)
    valid_iter = construct_dataset(test_seqs, test_labels, train=False)

    return train_iter, valid_iter

def load_ind_data(fasta_file, nums=1104):
    seqs = load_data(fasta_file)
    labels = np.vstack((np.ones((int(nums / 2), 1), dtype=int), np.zeros((int(nums / 2), 1), dtype=int))).flatten()
    seqs, labels = np.array(seqs), np.array(labels)
    data_iter = construct_dataset(seqs, labels, train=False)
    return data_iter
    
def load_bench_Metadata(pos_file,neg_file, pos_nums, neg_nums):
    pos_seqs = load_Meta_data(pos_file)
    neg_seqs = load_Meta_data(neg_file)
    seqs = np.concatenate((pos_seqs,neg_seqs),axis=0)
    labels = np.vstack((np.ones((pos_nums, 1), dtype=int), np.zeros((neg_nums, 1), dtype=int))).flatten()

    train_seqs, test_seqs, train_labels, test_labels = train_test_split(seqs, labels, test_size=0.2, random_state=42)
    train_iter = construct_dataset(train_seqs, train_labels, train=True)
    valid_iter = construct_dataset(test_seqs, test_labels, train=False)
    
    return train_iter, valid_iter

def load_ind_Metadata(pos_file,neg_file, pos_nums, neg_nums):
    pos_seqs = load_Meta_data(pos_file)
    neg_seqs = load_Meta_data(neg_file)
    seqs = np.concatenate((pos_seqs,neg_seqs),axis=0)

    labels = np.vstack((np.ones((pos_nums, 1), dtype=int), np.zeros((neg_nums, 1), dtype=int))).flatten()
    seqs, labels = np.array(seqs), np.array(labels)
    data_iter = construct_dataset(seqs, labels, train=False)
    return data_iter

if __name__ == '__main__':
    # Load benchmark dataset and independent test set for iRNA-ac4C data
    path_trainset = '../dataset/iRNA-ac4C/ac4c-trainset.txt'
    train_iter, valid_iter = load_bench_data(path_trainset, nums=4412)

    path_testset = '../dataset/iRNA-ac4C/ac4c-testset.txt'
    ind_iter = load_ind_data(path_testset, nums=1104)

    print(f'len(train_iter): {len(train_iter)}')
    print(f'len(valid_iter): {len(valid_iter)}')
    print(f'len(ind_iter): {len(ind_iter)}')

    # Load benchmark dataset and independent test set for Meta-ac4C balanced data
    path_balance_Meta_pos_train = '../dataset/Meta-ac4C/ac4c_balance_train_test/ac4c_positive_train.fa'
    path_balance_Meta_neg_train = '../dataset/Meta-ac4C/ac4c_balance_train_test/ac4c_negative_train.fa'
    train_Meta_iter, valid_Meta_iter = load_bench_Metadata(path_balance_Meta_pos_train, path_balance_Meta_neg_train, pos_nums = 1148, neg_nums = 1148)

    path_balance_Meta_pos = '../dataset/Meta-ac4C/ac4c_balance_train_test/ac4c_positive_test.fa'
    path_balance_Meta_neg = '../dataset/Meta-ac4C/ac4c_balance_train_test/ac4c_negative_test.fa'
    ind_Meta_iter = load_ind_Metadata(path_balance_Meta_pos, path_balance_Meta_neg, pos_nums = 467, neg_nums = 467)

    print(f'len(train_Meta_iter): {len(train_Meta_iter)}')
    print(f'len(valid_Meta_iter): {len(valid_Meta_iter)}')
    print(f'len(ind_Meta_iter): {len(ind_Meta_iter)}')


    # Load benchmark dataset and independent test set for Meta-ac4C unbalanced data
    path_unbalance_Meta_pos_train = '../dataset/Meta-ac4C/ac4c_unbalance_train_test/ac4c_unbalance_positive_train.fa'
    path_unbalance_Meta_neg_train = '../dataset/Meta-ac4C/ac4c_unbalance_train_test/ac4c_unbalance_negative_train.fa'
    train_Meta_iter, valid_Meta_iter = load_bench_Metadata(path_unbalance_Meta_pos_train, path_unbalance_Meta_neg_train, pos_nums = 1148, neg_nums = 5439)

    path_unbalance_Meta_pos_test = '../dataset/Meta-ac4C/ac4c_balance_train_test/ac4c_positive_test.fa'
    path_unbalance_Meta_neg_test = '../dataset/Meta-ac4C/ac4c_balance_train_test/ac4c_negative_test.fa'
    ind_Meta_iter = load_ind_Metadata(path_unbalance_Meta_pos_test, path_unbalance_Meta_neg_test, pos_nums = 467, neg_nums = 467)

    print(f'len(train_Meta_unbalance_iter): {len(train_Meta_iter)}')
    print(f'len(valid_Meta_unbalance_iter): {len(valid_Meta_iter)}')
    print(f'len(ind_Meta_unbalance_iter): {len(ind_Meta_iter)}')