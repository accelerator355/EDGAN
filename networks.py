import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, num_layers, bidirectional):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        if bidirectional:
            self.linear2 = nn.Linear(in_features = self.hidden_dim * 2,
                                     out_features = z_dim)
        else:
            self.linear2 = nn.Linear(in_features = self.hidden_dim,
                                     out_features = z_dim)

        self.rnn = nn.LSTM(input_size = hidden_dim,
                           hidden_size = hidden_dim,
                           num_layers = num_layers,
                           bidirectional = bidirectional)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x = x.permute(1, 0, 2)
        rnn_inp = self.linear1(x)
        rnn_out, _ = self.rnn(rnn_inp)
        z = self.linear2(rnn_out).permute(1, 0, 2)
        return z

class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, num_layers, bidirectional):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        if bidirectional:
            self.linear2 = nn.Linear(in_features = self.hidden_dim * 2,
                                     out_features = input_dim)
        else:
            self.linear2 = nn.Linear(in_features = self.hidden_dim,
                                     out_features = input_dim)

        self.rnn = nn.LSTM(input_size = hidden_dim,
                           hidden_size = hidden_dim,
                           num_layers = num_layers,
                           bidirectional = bidirectional)
        self.fun = nn.Tanh()

    def forward(self, z):
        self.rnn.flatten_parameters()
        z = z.permute(1, 0, 2)
        rnn_inp = self.linear1(z)
        rnn_out, _ = self.rnn(rnn_inp)
        re_x = self.linear2(rnn_out).permute(1, 0, 2)
        re_x = self.fun(re_x) * 0.5 + 0.5
        return re_x

class Generator(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, num_layers, bidirectional):
        super(Generator, self).__init__()

        self.encoder = Encoder(input_dim, z_dim, hidden_dim, num_layers, bidirectional)
        self.decoder = Decoder(input_dim, z_dim, hidden_dim, num_layers, bidirectional)

    def forward(self, x):
        z = self.encoder(x)
        re_x = self.decoder(z)
        return z, re_x

class Discriminator(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim):
        super(Discriminator, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim * window_size, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x, 1, -1)
        feature = self.feature(x)
        output = self.classifier(feature)
        output = output.view(x.shape[0])
        return feature, output