import torch
import torch.nn as nn

class AE_MAD(nn.Module):

    def __init__(self, L, n_channels, embedding_dim=4, n_kernel=6, kernel_size=7):
        super().__init__()
        self.L = L 

        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, n_kernel, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(n_kernel),
            nn.ReLU(),
            nn.Conv1d(n_kernel, n_kernel, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(n_kernel),
            nn.ReLU(),
        )
        self.output_layer1 = nn.Linear(n_kernel, embedding_dim)
        self.output_layer2 = nn.Linear(n_kernel, n_channels)
        self.decoder = nn.Sequential(
            nn.Conv1d(embedding_dim, n_kernel, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(n_kernel),
            nn.ReLU(),
            nn.Conv1d(n_kernel, n_kernel, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(n_kernel),
            nn.ReLU(),
        )

    # X.shape=(batch_size, n_channels, L)
    def forward(self, X):
        embedding = self.encoder(X.permute(0, 2, 1))
        embedding = self.output_layer1(embedding.permute(0,2,1)).permute(0,2,1)
        reconstruction = self.decoder(embedding)
        reconstruction = self.output_layer2(reconstruction.permute(0,2,1))
        return reconstruction

class TemporalBlock(nn.Module):

    def __init__(self, input_filters, n_kernel, kernel_size, skip_connections=True, dropout=0.2, dilation=1):
        super().__init__()
        self.block = []
        self.block.append(nn.Conv1d(input_filters, n_kernel, kernel_size, dilation=dilation, padding='same'))
        self.block.append(nn.BatchNorm1d(n_kernel))
        self.block.append(nn.ReLU())
        if dropout != 0:
            self.block.append(nn.Dropout(dropout))
        self.block.append(nn.Conv1d(n_kernel, n_kernel, kernel_size, dilation=dilation, padding='same'))
        self.block.append(nn.BatchNorm1d(n_kernel))
        self.block = nn.Sequential(*self.block)
        self.downsample = nn.Conv1d(input_filters, n_kernel, 1) if input_filters != n_kernel else None

        self.skip_connections = skip_connections

    def forward(self, X):
        res = X if self.downsample is None else self.downsample(X)
        out = self.block(X)
        if self.skip_connections:
            out = out + res

        return out

class DeepAE(nn.Module):

    def __init__(self, L, n_channels, embedding_dim=16, n_kernel=6, kernel_size=7, n_blocks=5, skip_connections=True, dropout=0.2):
        super().__init__()

        self.encoder = []
        self.decoder = []
        for i in range(n_blocks):
            self.encoder.append(TemporalBlock(n_channels if i == 0 else n_kernel, n_kernel, kernel_size, skip_connections=skip_connections, dilation=2**i, dropout=dropout))
            self.decoder.append(TemporalBlock(embedding_dim if i == 0 else n_kernel, n_kernel, kernel_size, skip_connections=skip_connections, dilation=2**i, dropout=dropout))
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        self.project_to_latent = nn.Linear(n_kernel, embedding_dim)
        self.project_to_output = nn.Linear(n_kernel, n_channels)

    # X.shape=(batch_size, n_channels, L)
    def forward(self, X):
        embedding = self.encoder(X.permute(0, 2, 1))
        embedding = self.project_to_latent(embedding.permute(0,2,1)).permute(0,2,1)
        reconstruction = self.decoder(embedding)
        reconstruction = self.project_to_output(reconstruction.permute(0,2,1))
        return reconstruction

class LSTMAE(nn.Module):

    def __init__(self, L, n_channels, embedding_dim=16, num_layers=4, dropout=0):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_channels, hidden_size=embedding_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.project_to_output = nn.Linear(2*embedding_dim, n_channels)
        self.embedding_dim = embedding_dim

    def forward(self, X):
        n_examples = X.shape[1]
        _, (hidden, _) = self.encoder(X)
        latent_vector = hidden[-1]
        stacked_LV = torch.repeat_interleave(latent_vector, n_examples, dim=1).reshape(-1, n_examples, self.embedding_dim).to(X.device)
        out, (_, _) = self.decoder(stacked_LV)
        out = self.project_to_output(out)
        return out
        

class SimpleAE(nn.Module):

    def __init__(self, L, n_channels, n_kernel=8, embedding_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, kernel_size=5, out_channels=n_kernel, padding='same'),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_kernel, kernel_size=5, out_channels=n_kernel, padding='same'),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_kernel, kernel_size=5, out_channels=n_kernel, padding='same'),
            nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, kernel_size=5, out_channels=n_kernel, padding='same'),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_kernel, kernel_size=5, out_channels=n_kernel, padding='same'),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_kernel, kernel_size=5, out_channels=n_kernel, padding='same'),
            nn.Upsample(scale_factor=2),
        )
        self.project_to_latent = nn.Linear(n_kernel, embedding_dim)
        self.project_to_output = nn.Linear(n_kernel, n_channels)

    def forward(self, X):
        embedding = self.encoder(X.permute(0, 2, 1))
        embedding = self.project_to_latent(embedding.permute(0,2,1)).permute(0,2,1)
        reconstruction = self.decoder(embedding)
        reconstruction = self.project_to_output(reconstruction.permute(0,2,1))
        return reconstruction