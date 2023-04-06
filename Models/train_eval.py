import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score
from torch.optim import Adam, AdamW, SGD

def client_update(args, loader, model, criterion, optimizer, device, scheduler = None):
    client_loss = []
    for ep in range(args.epochs):
        loss = train_fn(dataloader=loader, model=model, criterion=criterion, optimizer=optimizer, device=device)
        client_loss.append(loss)
    return model.state_dict(), client_loss

def train_fn(dataloader, model, criterion, optimizer, device, scheduler = None):
    model.to(device)
    model.train()
    train_loss = 0.0
    num_pt = 0
    for bi, d in enumerate(dataloader):
        # img_tensor, protected_att, target
        features, _, target = d
        num_pt += features.size(dim=0)
        features = features.to(device, dtype=torch.float)
        target = torch.squeeze(target, dim=-1).to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(features)
        output = torch.squeeze(output, dim=-1)
        loss = criterion(output, target)
        train_loss += loss.item()*features.size(dim=0)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
    return train_loss/num_pt

def eval_fn(data_loader, model, criterion, device):
    model.to(device)
    fin_targets = []
    fin_outputs = []
    loss = 0
    num_data_point = 0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            features, _, target = d
            features = features.to(device, dtype=torch.float)
            target = torch.squeeze(target, dim=-1).to(device, dtype=torch.float)
            num_data_point += features.size(dim=0)
            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            loss_eval = criterion(outputs, target)
            loss += loss_eval.item()*features.size(dim=0)
            outputs = outputs.cpu().detach().numpy()

            fin_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            fin_outputs.extend(outputs)

    return loss/num_data_point, fin_outputs, fin_targets

def performace_eval(args, y_true, y_pred):
    if args.performance_metric == 'acc':
        return accuracy_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif args.performance_metric == 'f1':
        return f1_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif args.performance_metric == 'auc':
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    elif args.performance_metric == 'pre':
        return precision_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))

def compute_grad(local_dict, global_dict):
    temp_par = {}
    for key in global_dict.keys():
        temp_par[key] = deepcopy(local_dict[key] - global_dict[key])
    return temp_par

def compute_glob_grad(client_grads):
    temp_grad = {}
    num_client = len(client_grads.keys())
    for i, client in enumerate(client_grads.keys()):
        if i == 0:
            for key in client_grads[client].keys(): temp_grad[key] = deepcopy(client_grads[client][key])/num_client
        else:
            for key in client_grads[client].keys():
                temp_grad[key] = temp_grad[key] + deepcopy(client_grads[client][key])
                temp_grad[key] = temp_grad[key]/num_client
    return temp_grad

def FedAvg(w_b, w_c):
    w_avg = deepcopy(w_b)
    for k in w_avg.keys():
        w_avg[k] = w_b[k] + w_c[k]
    return w_avg

def get_grad(args, loader, model, device):
    criterion = torch.nn.BCELoss(reduction='none') if args.num_class <= 2 else torch.nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)
    model.to(device)
    model.train()
    model.zero_grad()
    tar_att = []
    grad_ls = []
    for bi, d in enumerate(loader):
        features, att, target = d
        features = features.to(device, dtype=torch.float)
        target = torch.squeeze(target).to(device, dtype=torch.float)
        output = model(features)
        output = torch.squeeze(output)
        loss = criterion(output, target)
        for pos, j in enumerate(loss):
            j.backward(retain_graph=True)
            grad_vec = get_grad_vec(model)
            att_val = att[pos].item()
            grad_ls.append(grad_vec)
            tar_att.append(att_val)
            model.zero_grad()
    return np.array(grad_ls), np.array(tar_att)

def get_grad_vec(model):
    grad_vec = []
    for name, layer in model.named_parameters():
        layer_grad = (layer.grad).view(-1).tolist()
        grad_vec.extend(layer_grad)
    return grad_vec

def get_client_grad(glob_dict, loc_dict):
    grad_vec = []
    for key in glob_dict.keys():
        layer_grad = (loc_dict[key] - glob_dict[key]).view(-1).tolist()
        grad_vec.extend(layer_grad)
    return grad_vec

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False, run_mode=None, skip_ep=100):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.run_mode = run_mode
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        if self.run_mode == 'func' and epoch < self.skip_ep:
            return
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            if self.run_mode != 'func':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        self.val_score = epoch_score

