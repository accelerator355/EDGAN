import torch
from set_seed import set_seed
from load_data import load_data
from networks import Generator, Discriminator
from EDGAN import EDGAN

set_seed(12345)

params = {
    'dataset_name': 'real_satellite_data_1',
    'window_size': 30,  # 30 for real_satellite_data_1, 40 for real_satellite_data_2, 50 for public_satellite_data_1
    'window_stride': 1, # 1 for real_satellite_data_1, 2 for real_satellite_data_2, 5 for public_satellite_data_1
    'batch_size': 64,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    'z_dim': 5,
    'hidden_dim': 50,
    'num_layers': 1,
    'if_bidirectional': True,

    'g_lr': 0.0003,
    'd_lr': 0.0001,
    'epoch': 100,

    'discriminator_num': 1,
    'alpha': 0.1, # 0.1 for real_satellite_data_1, 100 for real_satellite_data_2, 100 for public_satellite_data_1

    'if_train': True,
    'outfolder': './result'
}

def main():
    data = load_data(dataset_name = params['dataset_name'],
                     window_size = params['window_size'],
                     window_stride = params['window_stride'],
                     batch_size = params['batch_size'])

    model = EDGAN(generator = Generator(input_dim = data['n_features'],
                                                                      z_dim = params['z_dim'],
                                                                      hidden_dim = params['hidden_dim'],
                                                                      num_layers = params['num_layers'],
                                                                      bidirectional = params['if_bidirectional']),
                                discriminator = Discriminator(input_dim = data['n_features'],
                                                                                window_size = params['window_size'],
                                                                                hidden_dim = params['hidden_dim']),
                                data_loader = data,
                                params = params)

    if params['if_train']:
        model.train()

    metrics = model.test()

if __name__ == '__main__':
    main()