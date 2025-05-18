import numpy
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics_ori as Metrics_ori
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from utils.ImageListDataset import ImageListDataset
from Uformer import model_utils as U_utils
from PIL import Image
import torch.nn.functional as F


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in sorted(os.listdir(dir)):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            fname = fname.split('.')[0]
            images.append((fname, path))
    return images


def save_image(image: Image.Image, output_folder, image_name, image_index, ext='png'):
    if ext == 'jpeg' or ext == 'jpg' or ext == 'png':
        image = image.convert('RGBA')
    folder = os.path.join(output_folder, image_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, f'{image_index}.{ext}'))


def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    x = (tensor.permute(1, 2, 0)) * 255
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


if __name__ == "__main__":
    # 设置随机数种子
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(2023)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/specular_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    m_items = None
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            m_items = F.normalize(torch.rand((512, 512), dtype=torch.float), dim=1)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt, m_items)
    if opt['distributed']:
        model_restoration_d = torch.nn.DataParallel(diffusion)
        model_restoration_d.cuda()
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for train_data in tqdm(train_loader, total=len(train_loader)):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)  # input data
                diffusion.optimize_parameters()  # diffusion
                # if current_step % (opt['train']['print_freq']*5) == 0:      # every 5 epoch     refine
                #     diffusion.optimize_optimizer(current_epoch)
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # save model
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for val_data in tqdm(val_loader, total=len(val_loader)):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        input_img = Metrics_ori.tensor2img(visuals['Input'], min_max=(0, 1))  # uint8
                        gt_img = Metrics_ori.tensor2img(visuals['GT'], min_max=(0, 1))  # uint8
                        result_img = Metrics_ori.tensor2img(visuals['Result'], min_max=(0, 1))  # uint8

                        # generation
                        Metrics_ori.save_img(
                            gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
                        Metrics_ori.save_img(
                            input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
                        Metrics_ori.save_img(
                            result_img, '{}/{}_{}_res.png'.format(result_path, current_step, idx))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (result_img, input_img, gt_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics_ori.calculate_psnr(
                            input_img, gt_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((result_img, input_img, gt_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        logger.info('End of training.')
    else:
        #################################################################
        # # load the pre_trained Uformer model
        # logger.info('Loading Uformer Model...')
        # # from Uformer.Uformer_args import U_args
        # U_opt = opt['Uformer_args']
        # U_opt.gpus = args.gpu_ids
        # model_restoration = U_utils.get_arch(U_opt)
        # U_utils.load_checkpoint(model_restoration, U_opt.get("weights"))
        # model_restoration.cuda()
        # model_restoration.eval()
        # logger.info('Loading Uformer Model Finished.')
        #################################################################

        logger.info('Begin Model Evaluation.')

        from core import metrics as Metrics
        idx = 0
        lp = Metrics.util_lpips('alex')

        avg_psnr_1 = 0.0
        avg_ssim_1 = 0.0
        avg_lpips_1 = 0.0

        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')

        for _, val_data in tqdm(enumerate(val_loader), total=len(val_loader), position=0):

            idx += 1
            diffusion.feed_data(val_data)
            # degra = model_restoration(val_data['Input'])
            # val_data['Degra'] = degra
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            eval_psnr = Metrics.calculate_psnr(visuals['Result'][-1].unsqueeze(0), visuals['GT'])
            eval_ssim = Metrics.calculate_ssim(visuals['Result'][-1].unsqueeze(0), visuals['GT'])
            eval_lpips = lp.calculate_lpips(visuals['Result'][-1].unsqueeze(0), visuals['GT'])
            avg_psnr_1 += eval_psnr
            avg_ssim_1 += eval_ssim
            avg_lpips_1 += eval_lpips.item()

            # for grid image save
            gt_img = Metrics_ori.tensor2img(visuals['GT'], min_max=(0, 1))  # uint8
            input_img = Metrics_ori.tensor2img(visuals['Input'], min_max=(0, 1))  # uint8
            input_img_mode = 'grid'
            if input_img_mode == 'single':
                result_img = visuals['Result']  # uint8
                sample_num = result_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics_ori.save_img(
                        Metrics_ori.tensor2img(result_img[iter], min_max=(0, 1)),
                        '{}/{}_{}_result_{}.png'.format(result_path, current_step, idx, iter))
            else:
                Metrics_ori.save_img(
                    Metrics_ori.tensor2img(visuals['Result'][-1], min_max=(0, 1)),
                    '{}/{}_{}_result_d2.png'.format(result_path, current_step, idx))

            Metrics_ori.save_img(
                gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
            Metrics_ori.save_img(
                input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(result_img, Metrics_ori.tensor2img(visuals['Result'][-1], min_max=(0, 1)),
                                           gt_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr_1 / len(val_loader)
        avg_ssim = avg_ssim_1 / len(val_loader)
        avg_lpips = avg_lpips_1 / len(val_loader)
        print('1:PSNR: {:.4f}; SSIM: {:.4f}; lpips: {:.4f};'.format(avg_psnr, avg_ssim, avg_lpips))

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation # LPIPS: {:.4e}'.format(avg_lpips))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
