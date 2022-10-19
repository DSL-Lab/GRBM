from torchvision import utils
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import math
import logging
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # NOQA
from PIL import Image  # NOQA

sns.set_theme(style="darkgrid")


def setup_logging(log_level, log_file, logger_name="exp_logger"):
    """ Setup logging """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(levelname)-5s | %(asctime)s | File %(filename)-20s | Line %(lineno)-5d | %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=numeric_level)

    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    formatter = logging.Formatter(
        "%(levelname)-5s | %(asctime)s | %(filename)-25s | line %(lineno)-5d: %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger(logger_name).addHandler(console)

    return get_logger(logger_name)


def get_logger(logger_name="exp_logger"):
    return logging.getLogger(logger_name)


def cosine_schedule(eta_min=0, eta_max=1, T=10):
    return [
        eta_min + (eta_max - eta_min) * (1 + math.cos(tt * math.pi / T)) / 2
        for tt in range(T)
    ]


def unnormalize_img_tuple(img_tuple, mean, std):
    if isinstance(std, torch.Tensor):
        mean = mean.view(1, -1, 1, 1).to(img_tuple[0][1].device)
        std = std.view(1, -1, 1, 1).to(img_tuple[0][1].device)

    return [(xx[0], (xx[1] * std + mean).clamp(min=0, max=1)) for xx in img_tuple]


def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


def show_img(matrix, title):
    plt.figure()
    plt.axis('off')
    plt.gray()
    img = np.array(matrix, np.float64)
    plt.imshow(img)
    plt.title(title)

    fig = plt.gcf()
    img_out = fig2img(fig)
    plt.close()

    return img_out


def save_gif_fancy(imgs, nrow, save_name):
    imgs = (show_img(utils.make_grid(xx[1],
                                     nrow=nrow,
                                     normalize=False,
                                     padding=1,
                                     pad_value=1.0).permute(1, 2, 0).cpu().numpy(), f'sample at {xx[0]:03d} step') for xx in imgs)
    img = next(imgs)
    img.save(fp=save_name,
             format='GIF',
             append_images=imgs,
             save_all=True,
             duration=400,
             loop=0)


def visualize_sampling(model, epoch, config, tag=None, is_show_gif=True):
    tag = '' if tag is None else tag
    B, C, H, W = config['sampling_batch_size'], config['channel'], config[
        'height'], config['width']
    v_init = torch.randn(B, C, H, W).cuda()
    v_list = model.sampling(v_init,
                            num_steps=config['sampling_steps'],
                            save_gap=config['sampling_gap'])

    if 'GMM' in config['dataset']:
        samples = v_list[-1][1].view(B, -1).cpu().numpy()
        vis_2D_samples(samples, config, tags=f'{epoch:05d}')
        vis_density_GRBM(model, config, epoch=epoch)
    else:
        if is_show_gif:
            v_list = unnormalize_img_tuple(v_list, config['img_mean'],
                                           config['img_std'])
            save_gif_fancy(
                v_list, config['sampling_nrow'],
                f"{config['exp_folder']}/sample_imgs_epoch_{epoch:05d}{tag}.gif")
            img_vis = v_list[-1][1]
        else:
            if isinstance(config['img_std'], torch.Tensor):
                mean = config['img_mean'].view(1, -1, 1, 1).cuda()
                std = config['img_std'].view(1, -1, 1, 1).cuda()
            else:
                mean = config['img_mean']
                std = config['img_std']

            img_vis = (v_list[-1][1] * std + mean).clamp(min=0, max=1)

        utils.save_image(
            utils.make_grid(img_vis,
                            nrow=config['sampling_nrow'],
                            normalize=False,
                            padding=1,
                            pad_value=1.0).cpu(),
            f"{config['exp_folder']}/sample_imgs_epoch_{epoch:05d}{tag}.png")


def vis_2D_samples(samples, config, tags=None):
    f, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=samples[:, 0], y=samples[:, 1], color="#4CB391")
    ax.set(xlim=(-10, 10))
    ax.set(ylim=(-10, 10))
    plt.show()
    plt.savefig(
        f"{config['exp_folder']}/samples_{tags}.png", bbox_inches='tight')
    plt.close()


def vis_density_GMM(model, config):
    fig, ax = plt.subplots()
    x_density, y_density = 500, 500
    xses = np.linspace(-10, 10, x_density)
    yses = np.linspace(-10, 10, y_density)
    xy = torch.tensor([[[x, y] for x in xses]
                      for y in yses]).view(-1, 2).cuda().float()
    log_density_values = model.log_prob(xy)
    log_density_values = log_density_values.detach().view(
        x_density, y_density).cpu().numpy()
    dx = (xses[1] - xses[0]) / 2
    dy = (yses[1] - yses[0]) / 2
    extent = [xses[0] - dx, xses[-1] + dx, yses[0] - dy, yses[-1] + dy]
    im = ax.imshow(np.exp(log_density_values),
                   extent=extent,
                   origin='lower',
                   cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('probability density')
    plt.show()
    plt.savefig(f"{config['exp_folder']}/GMM_density.png", bbox_inches='tight')
    plt.close()


def vis_density_GRBM(model, config, epoch=None):
    fig, ax = plt.subplots()
    x_density, y_density = 500, 500
    xses = np.linspace(-10, 10, x_density)
    yses = np.linspace(-10, 10, y_density)
    xy = torch.tensor([[[x, y] for x in xses]
                      for y in yses]).view(-1, 2).cuda().float()
    eng_val = -model.marginal_energy(xy)
    eng_val = eng_val.detach().view(x_density, y_density).cpu().numpy()
    dx = (xses[1] - xses[0]) / 2
    dy = (yses[1] - yses[0]) / 2
    extent = [xses[0] - dx, xses[-1] + dx, yses[0] - dy, yses[-1] + dy]
    im = ax.imshow(eng_val, extent=extent, origin='lower', cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('negative energy')
    plt.show()
    plt.savefig(f"{config['exp_folder']}/GRBM_density_{epoch:05d}.png",
                bbox_inches='tight')
    plt.close()
