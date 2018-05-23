import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 32)
        self.logits = nn.Linear(32, 1)

    def forward(self, x):
        y = F.elu(self.layer1(x))
        y = F.elu(self.layer2(y))
        y = self.logits(y)

        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 32)
        self.logits = nn.Linear(32, 1)

    def forward(self, x):
        y = F.elu(self.layer1(x))
        y = F.elu(self.layer2(y))
        y = F.sigmoid(self.logits(y))

        return y


def generate_real_data(n):
    data = np.random.normal(0, 1, (n, 1))  # generate data
    return data


def main():
    # define networks and move them to the GPU (if available)
    g = Generator()
    g.to(device)
    d = Discriminator()
    d.to(device)

    # generate real dataset
    n = 1024
    data = generate_real_data(n)
    batched_data = np.split(data, n / 8)

    # define optimizers for Generator and Discriminator
    g_optim = optim.Adam(g.parameters(), lr=2e-3)
    d_optim = optim.Adam(d.parameters(), lr=2e-3)

    # defines losses
    criterion = nn.BCELoss()

    # pretrain D
    for i, batch in enumerate(batched_data):
            d_optim.zero_grad()

            # Train D on real data
            labels_real = torch.ones((batch.shape[0], 1))
            d_prediction_real = d(torch.from_numpy(batch).float())
            d_loss_real = criterion(d_prediction_real, labels_real)
            d_loss_real.backward()

            # Train D on fake data
            labels_fake = torch.zeros((batch.shape[0], 1))
            inputs_fake = torch.rand((batch.shape[0], 1))
            batch_fake = g(inputs_fake)
            d_prediction_fake = d(batch_fake)
            d_loss_fake = criterion(d_prediction_fake, labels_fake)
            d_loss_fake.backward()

            d_optim.step()

    # training
    num_epochs = 10000
    num_samples = 1000
    dists = np.zeros((num_epochs, num_samples))
    for epoch in range(num_epochs):
        g_loss_mean = 0.
        d_loss_mean = 0.
        for i, batch in enumerate(batched_data):
            for _ in range(3):
                d_optim.zero_grad()

                # Train D on real data
                labels_real = torch.ones((batch.shape[0], 1))
                d_prediction_real = d(torch.from_numpy(batch).float())
                d_loss_real = criterion(d_prediction_real, labels_real)
                d_loss_real.backward()

                # Train D on fake data
                labels_fake = torch.zeros((batch.shape[0], 1))
                inputs_fake = torch.rand((batch.shape[0], 1))
                batch_fake = g(inputs_fake)
                d_prediction_fake = d(batch_fake)
                d_loss_fake = criterion(d_prediction_fake, labels_fake)
                d_loss_fake.backward()
                d_loss_mean += d_loss_fake.item() + d_loss_real.item()

                d_optim.step()

            # Train G
            g_optim.zero_grad()
            inputs_fake = torch.rand((batch.shape[0], 1))
            batch_fake = g(inputs_fake)
            d_prediction_fake = d(batch_fake)
            g_loss = criterion(d_prediction_fake, torch.ones((batch.shape[0], 1)))
            g_loss_mean += g_loss.item()
            g_loss.backward()
            g_optim.step()
        if epoch % 10 == 0:
            print("Epoch {}: g: {}, d: {}".format(epoch, g_loss_mean / i, d_loss_mean / (i * 3)))

        with torch.no_grad():
            inputs = torch.rand((num_samples, 1))
            outputs = g(inputs).numpy().flatten()
            dists[epoch] = outputs

    np.save('dists.npy', dists)


if __name__ == '__main__':
    main()
