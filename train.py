import json
from pytorch_lightning.callbacks import EarlyStopping
from neumf import *
from GMF import *
from Dataset import  *
import argparse

import pandas as pd
import gc
gc.collect()

from utils import *
from pytorch_lightning.callbacks import Callback



class MyPrintingCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        if pl_module.current_epoch % 5 ==0:
            print(pl_module._metron.subjects.head())


"""
Main training script. you can run from the command line by:
train.py --model GMF --data movielens1m --loss bpr 
For model specific hyperparameters, check the config files.
TODO: add the option to override model specific hyperparameters as cmd args 
TODO: create a hyperparameter tuning script
"""


def main(args):
    root = get_project_root()
    random.seed(42)
    if args.model == "GMF":
        config_path =  '/Config/GMFconfig.json'
    elif args.model == "neumf":
        config_path =  '/Config/neumfconfig.json'

    with open(str(root) + config_path) as json_file:
        config = json.load(json_file)


    if args.data == "movielens100":
        ml1m_rating = pd.read_csv(str(root) + "/Data/ml-100k/u.data", sep='\t', names="uid,mid,rating,timestamp".split(","))
    elif args.data == "movielens1m":
        ml1m_rating = pd.read_csv(str(root) + "/Data/ml-1m/ratings.dat", sep='::',names="uid,mid,rating,timestamp".split(","))

    if args.loss == "bpr":
        config['BPR_loss'] = 1
    elif args.loss == "CE":
        config['BPR_loss'] = 0

    sample_generator = preprocess_data(ml1m_rating, config)
    config['num_users'] = len(sample_generator.ratings['userId'].unique())
    config['num_items'] = len(sample_generator.ratings['itemId'].unique())

    if args.model == "GMF":
        model = GMF(config)

    elif args.model == "neumf":
        model = NeuMF(config)

    if args.loss == "bpr":
        train_loader = sample_generator.instance_a_train_loader_bpr(config['batch_size'])

    elif args.loss == "CE":
        train_loader = sample_generator.instance_a_train_loader(config['batch_size'])

    val_loader = sample_generator.instance_val_loader


    early_stop_callback = EarlyStopping(
        monitor='HR',
        min_delta=0.01,
        patience=10,
        verbose=True,
        mode='max'
    )

    for p in model.parameters():
        p.data.normal_(mean=0, std=0.01)

    trainer = pl.Trainer(gpus=1, max_epochs=config["num_epoch"],checkpoint_callback=False,
                         logger = False,early_stop_callback=early_stop_callback)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument('--data',
                        type=str,
                        default="movielens1m",
                        help="data name:movielens100 or movielens1m ")
    # model
    parser.add_argument('--model',
                        type=str,
                        default="neumf",
                        help="Model to train: GMF or neumf")

    # loss
    parser.add_argument('--loss',
                        type=str,
                        default="bpr",
                        help="bpr or CE // Use the pairwise loss or crossentropy")

    args = parser.parse_args()
    main(args)

