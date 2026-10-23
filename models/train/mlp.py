"""Train mlp on the summary features:  python -m models.train.mlp [--seeds 0 1 2] [--trees 500] [--save]"""
from models.train.tree_trainer import cli

if __name__ == '__main__':
    cli('mlp')
