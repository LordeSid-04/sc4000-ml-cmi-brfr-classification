"""Train lgbm on the summary features:  python -m models.train.lgbm [--seeds 0 1 2] [--trees 500] [--save]"""
from models.train.tree_trainer import cli

if __name__ == '__main__':
    cli('lgbm')
