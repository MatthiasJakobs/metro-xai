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

from models import AE_MAD, DeepAE

model_map = {
    'AE_MAD': AE_MAD,
    'DeepAE': DeepAE,
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
        
        self.model_class = model_map[config['model_class']]
        self.model_params = model_params
        self.hyperparameters = hyperparameters
        self.autoencoder = config['autoencoder']
        makedirs('models', exist_ok=True)
        self.path = f'models/pt2_{self.model_name}.pickle'

    def fit(self, X=None, y=None):

        is_torch = issubclass(self.model_class, nn.Module)

        if is_torch:
            with fixedseed(torch, self.model_seed):
                model = self.model_class(**self.model_params)
            self.model = NeuralNetRegressor(model, **self.hyperparameters)
            
            if exists(self.path):
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
        return self.model.predict(X)

    def calc_loss(self, X, y, average=True):
        preds = self.predict(X)
        errors = (preds-y)**2
        if not average:
            return errors
        return errors.mean()



if __name__ == '__main__':
    train_data, _, test_data, _ = load_data()

    model_name = sys.argv[1]
    trainer = ModelTrainer(f'configs/{model_name}.json')
    trainer.fit(train_data)
    print(model_name, 'train_loss', trainer.calc_loss(train_data, train_data))
    print(model_name, 'test_loss', trainer.calc_loss(test_data, test_data))
