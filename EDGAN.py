import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from evaluate import metrics_calculating

class EDGAN:
    def __init__(self, generator, discriminator, data_loader, params):
        super(EDGAN, self).__init__()
        self.g = generator.to(params['device'])
        self.d = discriminator.to(params['device'])
        self.data_loader = data_loader
        self.params = params

        self.g_optimizer = optim.Adam(params = self.g.parameters(), lr = self.params['g_lr'])
        self.d_optimizer = optim.Adam(params = self.d.parameters(), lr = self.params['d_lr'])

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

        self.iter = 0
        self.time_per_epoch = None
        self.adv_loss = None
        self.rec_loss = None
        self.g_loss = None
        self.d_loss = None

        self.save_path = os.path.join(self.params['outfolder'], self.params['dataset_name'])
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train_discriminator(self, x):
        self.d.train()
        self.d_optimizer.zero_grad()

        z, re_x = self.g(x)

        real_feature, real_logits = self.d(x)
        fake_feature, fake_logits = self.d(re_x)

        real_loss = self.bce(real_logits, torch.ones(size = (x.shape[0],), dtype = torch.float, device = self.params['device']))
        fake_loss = self.bce(fake_logits, torch.zeros(size = (x.shape[0],), dtype = torch.float, device = self.params['device']))

        self.d_loss = real_loss + fake_loss
        self.d_loss.backward()
        self.d_optimizer.step()

    def train_generator(self, x):
        self.g.train()
        self.g_optimizer.zero_grad()

        z, re_x = self.g(x)
        fake_feature, fake_logits = self.d(re_x)
        real_feature, real_logits = self.d(x)

        self.adv_loss = self.mse(fake_feature, real_feature)
        self.rec_loss = self.mse(x, re_x)

        self.g_loss = self.rec_loss + self.params['alpha'] * self.adv_loss
        self.g_loss.backward()
        self.g_optimizer.step()

    def train_epoch(self):
        self.batch_count = 0
        self.adv_batch_loss = 0
        self.rec_batch_loss = 0
        self.g_batch_loss = 0
        self.d_batch_loss = 0

        for x, _ in self.data_loader['train']:
            x = x.to(self.params['device'])

            for _ in range(self.params['discriminator_num']):
                self.train_discriminator(x)
            self.train_generator(x)

            self.batch_count += 1
            self.adv_batch_loss += self.adv_loss
            self.rec_batch_loss += self.rec_loss
            self.g_batch_loss += self.g_loss
            self.d_batch_loss += self.d_loss

    def train(self):
        for _ in range(self.params['epoch']):
            self.iter += 1
            self.save_best_weights()

            print('[Epoch %d/%d] g_loss : %.4f d_loss : %.4f' % (self.iter, self.params['epoch'], self.g_batch_loss/ self.batch_count, self.d_batch_loss / self.batch_count))

    def value_reconstruction(self, values, window_size):
        piece_num = len(values) // window_size
        reconstructed_x = []
        for i in range(piece_num):
            raw_values = values[i * window_size:(i + 1) * window_size, :]
            raw_values = torch.tensor(raw_values, dtype = torch.float).unsqueeze(0)
            raw_values = raw_values.to(self.params['device'])

            z, re_x = self.g(raw_values)

            re_x = re_x.squeeze().detach().tolist()
            reconstructed_x.extend(re_x)
        return np.array(reconstructed_x)

    def test(self):
        self.load_best_weights()
        self.g.eval()

        x_test, y_test = self.data_loader['test']
        x_test_r = self.value_reconstruction(values = x_test, window_size = self.params['window_size'])

        metrics_dict = metrics_calculating(x_test[:len(x_test_r)], x_test_r, y_test[:len(x_test_r)])
        return metrics_dict

    def save_best_weights(self):
        torch.save(self.g.state_dict(), os.path.join(self.save_path, 'weights_of_g.pkl'))
        torch.save(self.d.state_dict(), os.path.join(self.save_path, 'weights_of_d.pkl'))

    def load_best_weights(self):
        self.g.load_state_dict(torch.load(os.path.join(self.save_path, 'weights_of_g.pkl')))
        self.d.load_state_dict(torch.load(os.path.join(self.save_path, 'weights_of_d.pkl')))