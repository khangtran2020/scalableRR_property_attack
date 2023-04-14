import datetime
import warnings
import logging
from config import parse_args
from sklearn.ensemble import RandomForestClassifier
from Utils.utils import *
from Dataset.read_data import *
from Runs.run_normal import run as run_normal
from Runs.run_attack import run as run_attack

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)


def run(args, current_time, device):
    print(f'Running on {device}')
    name = get_name(args=args, current_date=current_time)
    args.num_feat = 512 if args.data_type == 'embedding' else (64, 64)
    args.num_class = 2
    args.num_att = 2

    if args.mode == 'dp':
        feature_matrix = np.load(f'Data/CelebA/celebA_eps_{args.tar_eps}.npz')['emb_all']
    elif args.mode == 'relax':
        feature_matrix = np.load(f'Data/CelebA/celebA_eps_{args.tar_eps}_relaxed.npz')['emb_all']

    with timeit(logger, 'init-data'):
        # read data
        train_df = pd.read_csv(args.data_path + 'train.csv')
        val_df = pd.read_csv(args.data_path + 'val.csv')
        test_df = pd.read_csv(args.data_path + 'test.csv')

        # build aux data
        if args.submode == 'attack':
            train_df, aux_df = build_aux(args=args, df=train_df)
            args.client_bs = len(list(train_df.groupby('client_id').count().index))
        else:
            temp = train_df.groupby('client_id').count()
            client_idx = list(temp[temp['image_id'] >= args.client_lb].index)
            args.client_bs = len(client_idx)
            train_df = train_df[train_df['client_id'].isin(client_idx)].copy().reset_index(drop=True)
    print(f'Training with {len(train_df["client_id"].unique())} users and each batch has {args.client_bs} clients')

    # build client dictionary
    with timeit(logger, 'create-client-dict'):
        print("Training with model {}".format(args.model_type))
        print("Optimizing with optimizer {}".format(args.optimizer))
        client_dict = {}
        client_ids = train_df['client_id'].unique().tolist()
        for i, client in enumerate(client_ids):
            client_df = read_celeba_csv_by_client_id(args=args, client_id=client, df=train_df)
            if args.mode in ['dp', 'relax']:
                client_id = client_df['image_name'].values.astype(int).tolist()
                client_matrix = feature_matrix[client_id]
                client_loader = init_loader(args=args, df=client_df, feat_matrix=client_matrix, mode='train')
            else:
                client_loader = init_loader(args=args, df=client_df, feat_matrix=None, mode='train')
            att = int(client_df['gender'].mean())
            client_dict[client] = {
                'client_df': client_df,
                'client_loader': client_loader,
                'att': att
            }

    va_loader = init_loader(args=args, df=val_df, mode='val', feat_matrix=None)
    te_loader = init_loader(args=args, df=test_df, mode='test', feat_matrix=None)
    if args.submode == 'attack':
        aux_loader = init_loader(args=args, df=aux_df, mode='aux', feat_matrix=None)
        attack_model = RandomForestClassifier(n_estimators=100, n_jobs=5, min_samples_leaf=5,
                                              min_samples_split=5, random_state=args.seed)
        attack_info = (aux_loader, attack_model)
    eval_data = (va_loader, te_loader)

    if args.submode == 'clean':
        run_normal(args=args, client_dict=client_dict, client_ids=client_ids, name=name,
                   device=device, eval_data=eval_data, logger=logger)
    else:
        run_attack(args=args, client_dict=client_dict, client_ids=client_ids, name=name,
                   device=device, eval_data=eval_data, attack_info=attack_info, logger=logger)


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args=args, current_time=current_time, device=device)
