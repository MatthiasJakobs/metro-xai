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