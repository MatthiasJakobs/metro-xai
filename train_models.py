import pickle
import torch
import torch.nn as nn
import numpy as np
import json
import sys
from tsx.models import NeuralNetRegressor, TSValidSplit
from skorch.callbacks import EarlyStopping, LRScheduler, GradientNormClipping
from seedpy import fixedseed
from os.path import exists
from os import makedirs
from generate_chunks import load_data

from models import AE_MAD, DeepAE, LSTMAE

model_map = {
    'AE_MAD': AE_MAD,
    'DeepAE': DeepAE,
    'LSTMAE': LSTMAE
}

class ModelTrainer:
    def __init__(self, config_name):
        self.load_config(config_name)
        self.model = None

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.model_seed = config.get('model_seed', 91721)
        self.model_name = config.get('model_name')

        available_channels = np.arange(8) if config['version'] == 2 else np.arange(7)
        self.use_channels = np.array(config.get('use_channels', available_channels))

        model_params = {}
        model_params.update(config['model_params'])
        hyperparameters = {}
        hyperparameters.update(config['hyperparameters'])

        # Replace string representations with actual objects
        if 'train_split' in hyperparameters:
            if hyperparameters['train_split'] is None:
                hyperparameters['train_split'] = None
            elif hyperparameters['train_split'] == 'TSValidSplit':
                hyperparameters['train_split'] = TSValidSplit()
            else:
                raise RuntimeError('Unknown train_split entry', hyperparameters['train_split'])

        # Replace callbacks with actual objects
        callbacks = []
        if 'callbacks' in hyperparameters:
            for callback in hyperparameters['callbacks']:
                if callback['name'] == 'EarlyStopping':
                    callbacks.append(EarlyStopping(**callback['parameters']))
                if callback['name'] == 'LRScheduler':
                    callbacks.append(LRScheduler(**callback['parameters']))
                if callback['name'] == 'GradientNormClipping':
                    callbacks.append(GradientNormClipping(**callback['parameters']))
            hyperparameters['callbacks'] = callbacks

        if 'optimizer' in hyperparameters:
            if hyperparameters['optimizer'] == 'SGD':
                hyperparameters['optimizer'] = torch.optim.SGD
            if hyperparameters['optimizer'] == 'Adam':
                hyperparameters['optimizer'] = torch.optim.Adam
        
        self.model_class = model_map[config['model_class']]
        self.model_params = model_params
        self.hyperparameters = hyperparameters
        self.autoencoder = config['autoencoder']
        self.version = config['version']
        makedirs('models', exist_ok=True)
        self.path = f'models/pt{self.version}_{self.model_name}.pickle'

    def preprocess(self, X):
        X = X[..., self.use_channels]
        return X

    def fit(self, X=None, y=None, refit=False):

        is_torch = issubclass(self.model_class, nn.Module)

        if X is not None:
            X = self.preprocess(X)
        if y is not None:
            y = self.preprocess(y)

        if is_torch:
            with fixedseed(torch, self.model_seed):
                model = self.model_class(**self.model_params)
            self.model = NeuralNetRegressor(model, **self.hyperparameters)
            
            if exists(self.path) and not refit:
                self.model.initialize()
                self.model.load_params(f_params=self.path, f_history=self.path.replace('pickle', 'history'))
            else:
                if self.autoencoder:
                    self.model.fit(X, X)
                else:
                    self.model.fit(X, y.squeeze())
                self.model.save_params(f_params=self.path, f_history=self.path.replace('pickle', 'history'))
        else:
            # Assume scikit-learn-style pipeline
            if exists(self.path):
                with open(self.path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                self.model = self.model_class(**self.model_params)
                self.model.fit(X, y)
                with open(self.path, 'wb') as f:
                    pickle.dump(self.model, f)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        X = self.preprocess(X)
        return self.model.predict(X)

    def calc_loss(self, X, y, average=True):
        preds = self.predict(X)
        y = self.preprocess(y)
        errors = (preds-y)**2
        if not average:
            return errors
        return errors.mean()

def train_cv(config):
    train_data, _, test_data, _ = load_data()
    n_splits = 5
    val_percents = np.linspace(0.1, 0.5, n_splits)
    val_losses = np.zeros((n_splits))
    print('===', config, '===')
    for i in range(n_splits):
        print(f'Split {i+1}/{n_splits} - {1-val_percents[i]:.2f} train {val_percents[i]:.2f} val')
        trainer = ModelTrainer(f'configs/{config}.json')
        trainer.hyperparameters['train_split'] = TSValidSplit(val_percents[i])
        trainer.hyperparameters['verbose'] = False
        trainer.fit(train_data, refit=True)
        val_data = train_data[-int(val_percents[i]*len(train_data)):]
        val_loss = trainer.calc_loss(val_data, val_data)
        val_losses[i] = val_loss

    print(val_losses)
    print('average', np.mean(val_losses))

if __name__ == '__main__':
    model_name = sys.argv[1]
    trainer = ModelTrainer(f'configs/{model_name}.json')
    train_data, _, test_data, _ = load_data(version=trainer.version)
    trainer.fit(train_data)
    print(model_name, 'train_loss', trainer.calc_loss(train_data, train_data))
    print(model_name, 'test_loss', trainer.calc_loss(test_data, test_data))
