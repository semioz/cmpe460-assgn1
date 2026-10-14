# CMPE 460 / 2026-2027 Fall -- Assignment 01

Student ID: ...
Student Name: ...

W&B Report Link: https://wandb.ai/semioz/cmpe460-assignment01/reports/CMPE-460-Assignment-01---Feedforward-Neural-Network-Experiments--VmlldzoxODExMDI0NQ==?accessToken=emchnbul5zewrvwkaxwy8wv58k87eudiz0db2gg27136qq4ndxc002t68ytwq3xb

Install: `pip install torch pandas numpy scikit-learn wandb`
Train: `python main.py` (requires a W&B login; saves the best checkpoint to `saved_objects/model.pt`)
Validate on a CUDA machine: `python run_test.py --valid`

Validation evaluator F1: 0.926465 (three 30-epoch hyperparameter experiments). Test-set score is unknown.
