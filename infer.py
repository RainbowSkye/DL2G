import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/specular_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='5')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = Data.create_dataset(dataset_opt, phase)
            test_loader = Data.create_dataloader(
                test_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model     load_stat_dict
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['test'], schedule_phase='test')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  test_data in enumerate(test_loader):
        idx += 1
        diffusion.feed_data(test_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        gt_img = Metrics.tensor2img(visuals['GT'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        input_img_mode = 'grid'
        if input_img_mode == 'single':
            # single img series
            input_img = visuals['Input']  # uint8
            sample_num = input_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(input_img[iter]), '{}/{}_{}_input_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            input_img = Metrics.tensor2img(visuals['Input'])  # uint8
            Metrics.save_img(
                input_img, '{}/{}_{}_input_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['Input'][-1]), '{}/{}_{}_input.png'.format(result_path, current_step, idx))

        Metrics.save_img(
            gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['Input'][-1]), gt_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
