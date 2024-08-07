import torch.nn as nn

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