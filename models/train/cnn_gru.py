"""Train the cnn_gru network:  python -m models.train.cnn_gru --variant full|imu [--folds 0 1] [--seeds 0 1 2] [--save]"""
from models.train.nn_trainer import cli

if __name__ == '__main__':
    cli('cnn_gru')
