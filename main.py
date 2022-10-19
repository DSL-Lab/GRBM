import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
import numpy as np
from tqdm import tqdm
from utils import setup_logging, vis_density_GMM, vis_2D_samples, visualize_sampling
from gmm import GMM, GMMDataset
from grbm import GRBM

EPS = 1e-7
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)


def save(model, results_folder, epoch):
    data = {'epoch': epoch, 'model': model.state_dict()}
    torch.save(data, f'{results_folder}/model-{epoch}.pt')


def load(model, results_folder, epoch):
    data = torch.load(f'{results_folder}/model-{epoch}.pt')
    model.load_state_dict(data['model'])


def train(model,
          train_loader,
          optimizer,
          config):
    model.train()
    for ii, (data, _) in enumerate(tqdm(train_loader)):
        if config['cuda']:
            data = data.cuda()

        optimizer.zero_grad()
        model.CD_grad(data)
        if config['clip_norm'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
        optimizer.step()

        if ii == len(train_loader) - 1:
            recon_loss = model.reconstruction(data).item()

    return recon_loss


def get_config(pid):
    config = {}
    # config['dataset'] = 'GMM_iso'
    config['dataset'] = 'GMM_aniso'
    # config['dataset'] = 'MNIST'
    # config['dataset'] = 'CelebA'
    # config['dataset'] = 'CelebA2K'
    # config['dataset'] = 'FashionMNIST'
    config['cuda'] = True
    config['model'] = 'GRBM'
    config['batch_size'] = 512
    config['epochs'] = 10000
    config['lr'] = 1.0e-2
    config['clip_norm'] = 10.0
    config['wd'] = 0.0e-4
    config['resume'] = 0
    # visualize sampling process, filters, hiddens if True
    config['is_vis_verbose'] = False
    config['init_var'] = 1e-0  # init variance of GRBM
    config['CD_step'] = 100
    config['CD_burnin'] = 0
    config['Langevin_step'] = 10
    config['Langevin_eta'] = 0.1
    config['is_anneal_Langevin'] = True
    config['Langevin_adjust_warmup_epoch'] = 0
    config['Langevin_adjust_step'] = 0
    # config['inference_method'] = 'Gibbs'
    # config['inference_method'] = 'Langevin'
    config['inference_method'] = 'Gibbs-Langevin'
    config['sampling_batch_size'] = 100
    config['sampling_steps'] = config['CD_step']
    config['sampling_gap'] = min(5, config['sampling_steps'])
    config['sampling_nrow'] = 10

    if 'GMM' in config['dataset']:
        config['batch_size'] = 100
        config['num_samples'] = 1000
        config['height'] = 1
        config['width'] = 1
        config['channel'] = 2
        config['log_interval'] = 100
        config['save_interval'] = 500
        config['epochs'] = 50000
        config['hidden_size'] = 256
    elif config['dataset'] == 'MNIST':
        config['height'] = 28
        config['width'] = 28
        config['channel'] = 1
        config['img_mean'] = torch.tensor([0.1307])
        config['img_std'] = torch.tensor([0.3081])
        config['log_interval'] = 10
        config['save_interval'] = 100
        config['epochs'] = 3000
        config['hidden_size'] = 4096
    elif config['dataset'] == 'CelebA':
        config['height'] = 32
        config['width'] = 32
        config['channel'] = 3
        config['crop_size'] = 140
        config['img_mean'] = torch.tensor([0.5240, 0.4152, 0.3590])
        config['img_std'] = torch.tensor([0.2868, 0.2530, 0.2453])
        config['log_interval'] = 1
        config['save_interval'] = 5
        config['hidden_size'] = 10000
        config['sampling_batch_size'] = 64
        config['sampling_nrow'] = 8
    elif config['dataset'] == 'CelebA2K':
        config['batch_size'] = 100
        config['height'] = 64
        config['width'] = 64
        config['channel'] = 3
        config['crop_size'] = 140
        config['img_mean'] = torch.tensor([0.5240, 0.4152, 0.3590])
        config['img_std'] = torch.tensor([0.2868, 0.2530, 0.2453])
        config['log_interval'] = 1
        config['save_interval'] = 5
        config['hidden_size'] = 10000
        config['sampling_batch_size'] = 64
        config['sampling_nrow'] = 8
    elif config['dataset'] == 'FashionMNIST':
        config['height'] = 28
        config['width'] = 28
        config['channel'] = 1
        config['img_mean'] = torch.tensor([0.2860])
        config['img_std'] = torch.tensor([0.3530])
        config['log_interval'] = 10
        config['save_interval'] = 100
        config['epochs'] = 3000
        config['hidden_size'] = 4096

    config['visible_size'] = config['height'] * \
        config['width'] * config['channel']

    if config['inference_method'] == 'Gibbs':
        config[
            'exp_folder'] = f"exp/{config['dataset']}_{config['model']}_{pid}_inference={config['inference_method']}_H={config['hidden_size']}_B={config['batch_size']}_CD={config['CD_step']}"
    elif config['inference_method'] == 'Langevin':
        config[
            'exp_folder'] = f"exp/{config['dataset']}_{config['model']}_{pid}_inference={config['inference_method']}_Langevin_adjust_warmup_epoch={config['Langevin_adjust_warmup_epoch']}_is_anneal_Langevin={config['is_anneal_Langevin']}_Langevin_adjust_step={config['Langevin_adjust_step']}_Langevin_eta={config['Langevin_eta']}_H={config['hidden_size']}_B={config['batch_size']}_CD={config['CD_step']}"
    elif config['inference_method'] == 'Gibbs-Langevin':
        config['exp_folder'] = f"exp/{config['dataset']}_{config['model']}_{pid}_inference={config['inference_method']}_Langevin_adjust_warmup_epoch={config['Langevin_adjust_warmup_epoch']}_is_anneal_Langevin={config['is_anneal_Langevin']}_Langevin_adjust_step={config['Langevin_adjust_step']}_Langevin_eta={config['Langevin_eta']}_Langevin_step={config['Langevin_step']}_H={config['hidden_size']}_B={config['batch_size']}_CD={config['CD_step']}"

    return config


def create_dataset(config):
    if 'GMM' in config['dataset']:
        if config['dataset'] == 'GMM_iso':
            # isotropic
            gmm_model = GMM(torch.tensor([0.33, 0.33, 0.34]),
                            torch.tensor([[-5, -5], [5, -5], [0, 5]]),
                            torch.tensor([[1, 1], [1, 1], [1, 1]])).cuda()
        else:
            # anisotropic
            gmm_model = GMM(torch.tensor([0.33, 0.33, 0.34]),
                            torch.tensor([[-5, -5], [5, -5], [0, 5]]),
                            torch.tensor([[1.25, 0.5], [1.25, 0.5], [0.5,
                                                                     1.25]])).cuda()

        vis_density_GMM(gmm_model, config)
        samples = gmm_model.sampling(config['num_samples'])
        vis_2D_samples(samples.cpu().numpy(), config, tags='ground_truth')
        train_set = GMMDataset(samples)
    elif config['dataset'] == 'MNIST':
        train_set = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(config['img_mean'],
                                                            config['img_std'])
                                   ]))
    elif config['dataset'] == 'CelebA':
        train_set = datasets.CelebA('./data',
                           split='train',
                           download=False,
                           transform=transforms.Compose([
                               transforms.CenterCrop(config['crop_size']),
                               transforms.Resize(config['height']),
                               transforms.ToTensor(),
                               transforms.Normalize(config['img_mean'],
                                                    config['img_std'])
                           ]))
    elif config['dataset'] == 'CelebA2K':
        train_set = datasets.CelebA('./data',
                           split='train',
                           download=False,
                           transform=transforms.Compose([
                               transforms.CenterCrop(config['crop_size']),
                               transforms.Resize(config['height']),
                               transforms.ToTensor(),
                               transforms.Normalize(config['img_mean'],
                                                    config['img_std'])
                           ]))
        train_set = torch.utils.data.Subset(train_set, range(2000))
    elif config['dataset'] == 'FashionMNIST':
        train_set = datasets.FashionMNIST('./data',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(config['img_mean'],
                                                                   config['img_std'])
                                          ]))

    if 'GMM' not in config['dataset']:
        config['img_mean'] = torch.tensor(config['img_mean'])
        config['img_std'] = torch.tensor(config['img_std'])

    return train_set


def train_model(args):
    """Let us train a GRBM and see how it performs"""
    pid = os.getpid()
    # Load config
    with open(f'config/{args.dataset}.json') as json_file:
        config = json.load(json_file)

    config['exp_folder'] = f"exp/{config['dataset']}_{config['model']}_{pid}_inference={config['inference_method']}_H={config['hidden_size']}_B={config['batch_size']}_CD={config['CD_step']}"

    if not os.path.isdir(config['exp_folder']):
        os.makedirs(config['exp_folder'])

    log_file = os.path.join(config['exp_folder'], f'log_exp_{pid}.txt')
    logger = setup_logging('INFO', log_file)
    logger.info('Writing log file to {}'.format(log_file))

    with open(os.path.join(config['exp_folder'], f'config_{pid}.json'),
              'w') as outfile:
        json.dump(config, outfile, indent=4)

    config['visible_size'] = config['height'] * \
        config['width'] * config['channel']
    train_set = create_dataset(config)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config['batch_size'],
                                               shuffle=True)

    model = GRBM(config['visible_size'],
                 config['hidden_size'],
                 CD_step=config['CD_step'],
                 CD_burnin=config['CD_burnin'],
                 init_var=config['init_var'],
                 inference_method=config['inference_method'],
                 Langevin_step=config['Langevin_step'],
                 Langevin_eta=config['Langevin_eta'],
                 is_anneal_Langevin=True,
                 Langevin_adjust_step=config['Langevin_adjust_step'])

    if config['cuda']:
        model.cuda()

    param_wd, param_no_wd = [], []
    for xx, yy in model.named_parameters():
        if 'W' in xx:
            param_wd += [yy]
        else:
            param_no_wd += [yy]

    optimizer = optim.SGD([{
        'params': param_no_wd,
        'weight_decay': 0
    }, {
        'params': param_wd
    }],
        lr=config['lr'],
        momentum=0.0,
        weight_decay=config['wd'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['epochs'])

    if config['resume'] > 0:
        load(model, config['exp_folder'], config['resume'])

    for epoch in range(config['resume']):
        scheduler.step()

    is_show_training_data = False
    for epoch in range(config['resume'] + 1, config['epochs'] + 1):
        if epoch <= config['Langevin_adjust_warmup_epoch']:
            model.set_Langevin_adjust_step(config['CD_step'])
        else:
            model.set_Langevin_adjust_step(config['Langevin_adjust_step'])

        recon_loss = train(model,
                           train_loader,
                           optimizer,
                           config)

        var = model.get_var().detach().cpu().numpy()

        # show samples periodically
        if epoch % config['log_interval'] == 0:
            if 'GMM' in config['dataset']:
                logger.info(
                    f'PID={pid} || {epoch} epoch || mean = {model.mu.detach().cpu().numpy()} || var={model.get_var().detach().cpu().numpy()} || Reconstruction Loss = {recon_loss}'
                )
            else:
                logger.info(
                    f'PID={pid} || {epoch} epoch || var={model.get_var().mean().item()} || Reconstruction Loss = {recon_loss}'
                )

            visualize_sampling(model,
                               epoch,
                               config,
                               is_show_gif=config['is_vis_verbose'])

            # visualize one mini-batch of training data
            if not is_show_training_data and 'GMM' not in config['dataset']:
                data, _ = next(iter(train_loader))
                mean = config['img_mean'].view(1, -1, 1, 1).to(data.device)
                std = config['img_std'].view(1, -1, 1, 1).to(data.device)
                vis_data = (data * std + mean).clamp(min=0, max=1)
                utils.save_image(
                    utils.make_grid(vis_data,
                                    nrow=config['sampling_nrow'],
                                    normalize=False,
                                    padding=1,
                                    pad_value=1.0).cpu(),
                    f"{config['exp_folder']}/training_imgs.png")
                is_show_training_data = True

            # visualize filters & hidden states
            if config['is_vis_verbose']:
                filters = model.W.T.view(model.W.shape[1], config['channel'],
                                         config['height'], config['width'])
                utils.save_image(
                    filters,
                    f"{config['exp_folder']}/filters_epoch_{epoch:05d}.png",
                    nrow=8,
                    normalize=True,
                    padding=1,
                    pad_value=1.0)

                # visualize hidden states
                data, _ = next(iter(train_loader))
                h_pos = model.prob_h_given_v(
                    data.view(data.shape[0], -1).cuda(), model.get_var())
                utils.save_image(h_pos.view(1, 1, -1, config['hidden_size']),
                                 f"{config['exp_folder']}/hidden_epoch_{epoch:05d}.png",
                                 normalize=True)

        # save models periodically
        if epoch % config['save_interval'] == 0:
            save(model, config['exp_folder'], epoch)

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        help='Dataset name {gmm_iso, gmm_aniso, mnist, fashionmnist, celeba, celeba2K}')
    args = parser.parse_args()
    train_model(args)
