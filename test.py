import torch
from termcolor import colored
from model import  TBC_ac4C
from util.data_loader import load_ind_data, load_ind_Metadata
from util.util_metric import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model


if __name__ == '__main__':
    # Evaluating the Results on the Independent Test Set of iRNA-ac4C
    file = 'dataset/iRNA-ac4C/ac4c-testset.txt'
    path_pretrain_model = "model/TBC-ac4C.pt"
    ind_iter = load_ind_data(file)
    model = TBC_ac4C().to(device)
    model = load_model(model, path_pretrain_model)
    model.eval()
    with torch.no_grad():
        for x, label in ind_iter:
            if device:
                x, label = x.to(device), label.to(device)

        ind_performance, ind_roc_data, ind_prc_data, _ = evaluate(ind_iter, model)
    ind_results = '\n' + '=' * 16 + colored(' Independent Test Set Performance', 'red') + '=' * 16 \
                   + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(ind_performance[0],
                             ind_performance[1], ind_performance[2], ind_performance[3], ind_performance[4]) + '\n' + '=' * 60
    print(ind_results)
    
