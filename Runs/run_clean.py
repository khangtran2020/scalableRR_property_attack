import gc
import torch
from tqdm import tqdm
from Models.train_eval import *
from Models.models import init_model, init_optimizer
from Utils.utils import save_res, timeit


def run(args, client_dict, client_ids, name, device, eval_data, logger):

    num_client = len(client_ids)
    # set up criterion
    criterion = torch.nn.BCELoss() if args.num_class <= 2 else torch.nn.CrossEntropyLoss()
    criterion.to(device)
    model_name = f'{name}.pt'
    history = {
        'round': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
    }

    va_loader, te_loader = eval_data

    # build global model
    global_model = init_model(args=args, model_type='normal')
    global_model.to(device)
    es = EarlyStopping(patience=args.patience, verbose=False)

    for round in tqdm(range(args.rounds)):
        glob_dict = global_model.state_dict()
        if (round > 0) and (round % args.attack_round == 0):
            local_grad = []
            local_attr = []
        chosen_clients = np.random.choice(client_ids, args.client_bs, replace=False).tolist()
        for i, client in enumerate(chosen_clients):
            client_loader = client_dict[client]['client_loader']
            model = deepcopy(global_model)
            optimizer = init_optimizer(optimizer_name=args.optimizer, model=model,
                                       lr=args.lr, weight_decay=args.weight_decay)
            client_model_dict, client_loss = client_update(args=args, loader=client_loader, model=model, criterion=criterion,
                                              optimizer=optimizer, device=device)
            print(f'For client {client}: {client_loss}')
            local_update = deepcopy(client_model_dict) if i == 0 else FedAvg(local_update, client_model_dict)

        w_glob = deepcopy(local_update)
        for t in w_glob.keys():
            w_glob[t] = torch.div(w_glob[t], num_client)
        global_model.load_state_dict(w_glob)

        if round % args.eval_round == 0:
            va_loss, va_out, va_tar = eval_fn(data_loader=va_loader, model=global_model,
                                              criterion=criterion, device=device)
            va_acc = performace_eval(args, va_tar, va_out)

            te_loss, te_out, te_tar = eval_fn(data_loader=te_loader, model=global_model,
                                              criterion=criterion, device=device)
            te_acc = performace_eval(args, te_tar, te_out)

            es(epoch=round, epoch_score=va_acc, model=global_model, model_path=args.save_path + model_name)
            history['round'].append(round)
            history['val_history_loss'].append(va_loss)
            history['val_history_acc'].append(va_acc)
            history['test_history_loss'].append(te_loss)
            history['test_history_acc'].append(te_acc)
            print(f'Rounds {round}: val_loss = {va_loss}, val_{args.performance_metric} = {va_acc}, test_loss = {te_loss}, test_{args.performance_metric} = {te_acc}')
    save_res(name=name, args=args, dct=history)