import os
import numpy as np
import torch
import time
from termcolor import colored
from util.data_loader import construct_dataset
from util.util_metric import evaluate, reg_loss
from sklearn.model_selection import StratifiedKFold
from model import TBC_ac4C
from util.data_loader import load_data, load_bench_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
train.py provides complete model training, validation, and evaluation functions, including early stopping mechanism, learning rate scheduling, and k-fold cross-validation.

Main functions:
  1. Early stopping mechanism.
  2. Model training and validation.
  3. k-fold cross-validation.
"""

class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = None

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        self.best_acc = val_acc
        path = 'best_network.pt'
        torch.save(model.state_dict(), path)



def train_test(train_iter, valid_iter, iter_k=1):
    net = TBC_ac4C().to(device)
    lr = 0.00003
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15, delta=0.0001)

    best_acc = 0
    EPOCH = 100
    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()

        net.train()
        for x, label in train_iter:
            if device:
                x, label = x.to(device), label.to(device)

            output, attn = net(x)
            print(f'attn: {attn}')
            loss = reg_loss(net, output, label).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())

        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data, _ = evaluate(train_iter, net)
            valid_performance, valid_roc_data, valid_prc_data, label_real = evaluate(valid_iter, net)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4]) + '\n' + '=' * 60
        print(results)

        valid_acc = valid_performance[0]  # valid_performance: [ACC, Sensitivity, Specificity, AUC, MCC]

        scheduler.step(valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_performance = valid_performance
            best_ROC = valid_roc_data
            best_PRC = valid_prc_data

            if best_acc > 0.85:
                filename = '{}, {}[{:.4f}].pt'.format(
                    'mRNA_Model' + ', {}折'.format(iter_k + 1) + ', epoch[{}]'.format(epoch + 1), 'ACC', best_acc)
                save_path_pt = os.path.join('./Result', filename)  
                torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)

                best_ROC = np.array(best_ROC, dtype=object)  
                best_PRC = np.array(best_PRC, dtype=object)
                # np.save("./Result/{}fold-valid_best_ROC.npy".format(iter_k + 1), best_ROC)
                # np.save("./Result/{}fold-valid_best_PRC.npy".format(iter_k + 1), best_PRC)

            best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                best_performance[4]) + '\n' + '=' * 60

        early_stopping(valid_acc, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return best_performance, best_results, best_ROC, best_PRC


def K_CV(file, k=10):
    seqs = load_data(file)
    labels = np.vstack((np.ones((int(4412 / 2), 1), dtype=int), np.zeros((int(4412 / 2), 1), dtype=int))).flatten()
    seqs, labels = np.array(seqs), np.array(labels)
    CV_perform = []

    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=2025)
    for iter_k, (train_index, test_index) in enumerate(kfold.split(seqs, labels)):
        print("\n" + "=" * 16 + "k = " + str(iter_k + 1) + "=" * 16)

        train_seqs, test_seqs = seqs[train_index], seqs[test_index]
        train_lables, test_labels = labels[train_index], labels[test_index]
        train_iter = construct_dataset(train_seqs, train_lables, train=True)
        test_iter = construct_dataset(test_seqs, test_labels, train=False)

        performance, _, ROC, PRC = train_test(train_iter, test_iter, iter_k)
        print('交叉验证: best_performance', performance)
        CV_perform.append(performance)


    print('\n' + '=' * 16 + colored(' Cross-Validation Performance ',
                                    'red') + '=' * 16 + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n')
    for k, out in enumerate(CV_perform):
        print( '第{}折: {:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(k + 1, out[0],
                                          out[1], out[2], out[3], out[4]))

    mean_out = np.array(CV_perform).mean(axis=0)  
    print('\n' + '=' * 16 + "Mean out" + '=' * 16)  # [ACC, Sensitivity, Specificity, AUC, MCC]

    print('ACC: {:.4f},\tSE: {:.4f},\tSP: {:.4f},\tAUC: {:.4f},\tMCC: {:.4f}'.format( \
        mean_out[0], mean_out[1], mean_out[2], mean_out[3], mean_out[4]))
    print('\n' + '=' * 60)


if __name__ == '__main__':
    net = TBC_ac4C().to(device)
    # print(net)

    path_trainset = 'dataset/iRNA-ac4C/ac4c-trainset.txt'
    train_iter, valid_iter = load_bench_data(path_trainset)
    best_performance, best_results, best_ROC, best_PRC = train_test(train_iter, valid_iter, iter_k=1)
    
    # k-fold cross-validation
    # K_CV('dataset/iRNA-ac4C/ac4c-trainset.txt', k=10)

