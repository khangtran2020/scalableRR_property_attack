import datetime
import warnings
import logging
from config import parse_args
from sklearn.ensemble import RandomForestClassifier
from Utils.utils import *
from Dataset.read_data import *
from Models.models import init_model, Logit
from Models.train_eval import init_optimizer
from Runs.run_clean import run as run_clean

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)


def run(args, current_time, device):
    name = get_name(args=args, current_date=current_time)
    args.num_feat = 512 if args.data_type == 'embedding' else (64, 64)
    args.num_class = 2
    args.num_att = 2

    with timeit(logger, 'init-data'):
        # read data
        train_df = pd.read_csv(args.data_path + 'train.csv')
        val_df = pd.read_csv(args.data_path + 'val.csv')
        test_df = pd.read_csv(args.data_path + 'test.csv')

        # build aux data
        train_df, aux_df = build_aux(args=args, df=train_df)

    # build client dictionary
    with timeit(logger, 'create-client-dict'):
        print("Training with model {}".format(args.model_type))
        print("Optimizing with optimizer {}".format(args.optimizer))
        client_dict = {}
        client_ids = train_df['client_id'].unique().tolist()
        for i, client in enumerate(client_ids):
            client_df = read_celeba_csv_by_client_id(args=args, client_id=client, df=train_df)
            client_loader = init_loader(args=args, df=client_df, mode='train')
            client_dict[client] = {
                'client_df': client_df,
                'client_loader': client_loader
            }

    va_loader = init_loader(args=args, df=val_df, mode='val')
    te_loader = init_loader(args=args, df=test_df, mode='test')
    aux_loader = init_loader(args=args, df=aux_df, mode='aux')
    attack_model = RandomForestClassifier(max_depth=2, random_state=args.seed)
    eval_data = (va_loader, te_loader)
    attack_info = (aux_loader, attack_model)

    # run experiments
    if args.mode == 'clean':
        run_clean(args=args, client_dict=client_dict, client_ids=client_ids, name=name,
                  device=device, eval_data=eval_data, attack_info=attack_info, logger=logger)


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args=args, current_time=current_time, device=device)