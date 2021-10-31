
nohup python train.py -m hydra.sweep.dir=exp/ \
hydra/launcher=joblib hydra.launcher.n_jobs=9 &

python join_result.py --exp_dir=exp/
