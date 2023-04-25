# Property inference attack for ScalableRR

1. Data processing step:

+ Run the following command: `bash shell/init.sh`
+ To run the attack on `noiseless` model, put the extracted embeddings features of CelebA images into the directory `Data/embedding`.
+ To run the attack under `ScalableRR` protection model, put the randomized embedding features `.npz` files into  the directory `Data/CelebA`. The format of file name is as follows:
  + For ScalableRR: `celebA_eps_<value of epsilon_X>.npz`
  + For ScalableRR-Relaxed: `celebA_eps_<value of epsilon_X>_relaxed.npz`

2. Running the attack: 

+ Run `noiseless` model:

```angular2html
for RUN in 1 2 3 4 5
do
    python main.py --seed $RUN --mode clean --submode attack --performance_metric auc --eval_round 10 \
        --attack_round 10  --rounds 100 --lr 0.1 --client_lb 30 --optimizer sgd --model_type lr \
        --n_hid 1 --hid_dim 16 --target Smiling --epochs 1 --batch_size 10 --batch_size_val 128 \
        --momentum 0.0 --weight_decay 0.0 --aux_type sort --att Male
done
```

+ Run `ScalableRR` model:

```angular2html
for RUN in 1 2 3 4 5
do
    python main.py --seed $RUN --mode dp --submode attack --performance_metric auc --eval_round 10 \
        --attack_round 10  --rounds 100 --lr 0.1 --client_lb 30 --optimizer sgd --model_type lr \
        --n_hid 1 --hid_dim 16 --target Smiling --epochs 1 --batch_size 10 --batch_size_val 128 \
        --momentum 0.0 --weight_decay 0.0 --aux_type sort --tar_eps <value of epsilon_X> --att Male
done
```

+ Run `ScalableRR-Relaxed` model:

```angular2html
for RUN in 1 2 3 4 5
do
    python main.py --seed $RUN --mode relax --submode attack --performance_metric auc --eval_round 10 \
        --attack_round 10  --rounds 100 --lr 0.1 --client_lb 30 --optimizer sgd --model_type lr \
        --n_hid 1 --hid_dim 16 --target Smiling --epochs 1 --batch_size 10 --batch_size_val 128 \
        --momentum 0.0 --weight_decay 0.0 --aux_type sort --tar_eps <value of epsilon_X> --att Male
done
```
