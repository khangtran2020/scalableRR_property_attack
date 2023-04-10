import argparse


def add_general_group(group):
    group.add_argument("--save_path", type=str, default="results/models/", help="dir path for saving model file")
    group.add_argument("--res_path", type=str, default="results/dict/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--mode", type=str, default='clean', help="Mode of running ['clean', 'dp']")
    group.add_argument("--submode", type=str, default='clean', help="Submode of running ['clean', 'attack']")
    group.add_argument("--debug", type=bool, default=True)
    group.add_argument("--performance_metric", type=str, default='acc', help="Metrics of performance")


def add_data_group(group):
    group.add_argument('--data_path', type=str, default='Data/', help="dir path to dataset")
    group.add_argument('--dataset', type=str, default='celeba', help="name of dataset")
    group.add_argument('--data_type', type=str, default='embedding', help="type of data ['embedding', 'image']")
    group.add_argument('--att', type=str, default='Male', help="protected attribute")
    group.add_argument('--target', type=str, default='Eyeglasses', help="label")
    group.add_argument('--ratio', type=float, default=0.2, help="train/test split ratio")
    group.add_argument("--num_client", type=int, default=2, help="# of clients")
    group.add_argument("--client_lb", type=int, default=10, help="# of images per client")
    group.add_argument("--aux_data_size", type=str, default='med', help="Size of auxiliary dataset")
    group.add_argument("--aux_type", type=str, default='sort', help="Type to divide the train data")

def add_model_group(group):
    group.add_argument("--model_type", type=str, default='lr', help="Model type")
    group.add_argument("--lr", type=float, default=0.1, help="learning rate")
    group.add_argument("--weight_decay", type=float, default=0.9, help="weight decay SGD")
    group.add_argument("--momentum", type=float, default=0.9, help="momentum SGD")
    group.add_argument('--batch_size', type=int, default=10, help="batch size for training process")
    group.add_argument('--batch_size_val', type=int, default=256, help="batch size for training process")
    group.add_argument('--client_bs', type=int, default=32, help="# clients per round")
    group.add_argument('--n_hid', type=int, default=2, help='number hidden layer')
    group.add_argument('--hid_dim', type=int, default=32, help='hidden embedding dim')
    group.add_argument("--optimizer", type=str, default='sgd')
    group.add_argument("--dropout", type=float, default=0.2)
    group.add_argument("--patience", type=int, default=20)
    group.add_argument("--epochs", type=int, default=1, help='# of training step')
    group.add_argument("--rounds", type=int, default=100, help='# of FL rounds')
    group.add_argument("--adv_model_type", type=str, default='rf', help='type of attack model')
    group.add_argument("--eval_round", type=int, default=10, help='evaluate after this number of round')
    group.add_argument("--attack_round", type=int, default=50, help='evaluate after this number of round')
    group.add_argument('--aux_bs', type=int, default=512, help="batch size for training process")

def add_dp_group(group):
    group.add_argument("--tar_eps", type=float, default=1.0, help="targeted epsilon")
    group.add_argument("--num_bit", type=int, default=10, help='# of bits to use')
    group.add_argument("--num_int", type=int, default=3, help='# of exponent bits to use')


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    general_group = parser.add_argument_group(title="General configuration")
    dp_group = parser.add_argument_group(title="DP configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    add_dp_group(dp_group)
    return parser.parse_args()