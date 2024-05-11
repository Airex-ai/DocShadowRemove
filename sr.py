import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import os
import numpy as np
import wandb
import utils
import random
from model.sr3_modules import transformer

from tqdm import tqdm


from natsort import ns, natsorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
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
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

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
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # degradation predict model
    # model_restoration = transformer.Uformer()
    # model_restoration.cuda()
    # path_chk_rest_student = '../models/model_epoch_450.pth'
    # utils.load_checkpoint(model_restoration, path_chk_rest_student)
    # model_restoration.eval()

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
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                # if current_epoch > 5:
                #     target, input_, mask = utils.MixUp_AUG().aug(train_data['HR'].cuda(), train_data['SR'].cuda(), train_data['mask'].cuda())
                #     train_data['HR'] = target
                #     train_data['SR'] = input_
                #     train_data['mask'] = mask
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
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
                        
                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    
                    avg_psnr_ = 0.0
                    avg_ssim_ = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(tqdm(val_loader)):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        pr_img = Metrics.tensor2img(visuals['PR']) 
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'], min_max=(0, 1))  # uint8
                        
                        

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            pr_img, '{}/{}_{}_pr.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(
                        #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)
                        avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)
                        
                        avg_psnr_ += Metrics.calculate_psnr(
                            lr_img, hr_img)
                        avg_ssim_ += Metrics.calculate_ssim(lr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    
                    avg_psnr_ = avg_psnr_ / idx
                    avg_ssim_ = avg_ssim_ / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    
                    logger.info('# chushi # PSNR: {:.4e}'.format(avg_psnr_))
                    logger.info('# chushi # SSIM: {:.4e}'.format(avg_ssim_))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('chushi <epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr_, avg_ssim_))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')

    
    
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        
        avg_psnr_df = 0.0
        avg_ssim_df = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        
        # lr_dir = "/media/wit/HDD_1/xwf/Data/geo_rec/"
        # hr_dir = "/media/wit/HDD_1/xwf/Data/DocReal/test/gt/"
        
        
        lr_dir = "/media/wit/HDD_1/xwf/Data/ISTD_Dataset/test/test_A"
        hr_dir = "/media/wit/HDD_1/xwf/Data/ISTD_Dataset/test/test_C"
        from PIL import Image
        import torchvision
        import torchvision.transforms.functional as F
        import cv2
        preresize = torchvision.transforms.Resize([256,256])
        totensor = torchvision.transforms.ToTensor()
        
        
        # for lr_file,hr_file in zip(sorted(os.listdir(lr_dir)),sorted(os.listdir(hr_dir))):
        for i in range(len(natsorted(os.listdir(hr_dir)))): 
            for j in range(2):
                hr_file = natsorted(os.listdir(hr_dir))[i]
                lr_file = natsorted(os.listdir(lr_dir))[i * 2 + j]
                
                idx += 1
                imp_lr = os.path.join(lr_dir, lr_file)
                im_lr = Image.open(imp_lr).convert('RGB')
                
                imp_hr = os.path.join(hr_dir, hr_file)
                im_hr = Image.open(imp_hr).convert('RGB')
                
                h,w,c = np.array(im_lr).shape
                # h_,w_,c_ = np.array(im_hr).shape
                afresize = torchvision.transforms.Resize([h,w])
                
                io_lr = im_lr
                im_lr = preresize(im_lr)
                im_lr = totensor(im_lr)
                # im_lr = torch.stack(im_lr,0)
                im_lr = im_lr.unsqueeze(0)
                
                crop_h, crop_w = im_lr.shape[2] % 16, im_lr.shape[3] % 16
                im_lr = im_lr[:, :, :im_lr.shape[2] - crop_h, :im_lr.shape[3] - crop_w]
                im_lr = torch.unbind(im_lr, dim=0)
                
                # im_lr = im_lr * 2 - 1
                im_lr = [img * (1 - (-1)) - 1 for img in im_lr]
                # im_lr = im_lr[0]
                # im_lr = im_lr.unsqueeze(0)
                im_lr = torch.stack(im_lr, dim=0)
                
                diffusion.feed_data(im_lr)
                diffusion.test(continous=True)
                visuals = diffusion.get_current_visuals()
                
                # mp1 = visuals['MP1']
                # mp1 = afresize(mp1)
                
                # mp2 = visuals['MP2']
                # mp2 = afresize(mp2)
                
                
                mp = visuals['MP']
                mp = mp.cpu()
                mp = afresize(mp) 
                mp = F.gaussian_blur(mp, kernel_size=5, sigma=1)
                
                
                
                
                sr_img_ = visuals['SR']
                df_img_ = visuals['DF']
                score_img = visuals['S']
                mp1 = visuals['MP1']
                
                mp_ = df_img_- im_lr
                mp_ = afresize(mp_) 
                # mp_ = F.gaussian_blur(mp_, kernel_size=5, sigma=1)
                
                
                io_lr = totensor(io_lr)
                io_lr = io_lr.unsqueeze(0)
                io_lr = [img * (1 - (-1)) - 1 for img in io_lr]
                io_lr = torch.stack(io_lr, dim=0)
                
                # sr_img = mp2 + (mp1 + io_lr)
                sr_img = mp + io_lr
                sr_img[sr_img > 1] = 1
                sr_img[sr_img < -1] = -1
                
                df_img = mp_ + io_lr
                
                im_hr = preresize(im_hr)
                im_hr = totensor(im_hr)
                im_hr = im_hr.unsqueeze(0)
                # im_hr = torch.stack(im_hr,0)
                
                crop_h, crop_w = im_hr.shape[2] % 16, im_hr.shape[3] % 16
                im_hr = im_hr[:, :, :im_hr.shape[2] - crop_h, :im_hr.shape[3] - crop_w]
                im_hr = torch.unbind(im_hr, dim=0)
                
                im_hr = [img * (1 - (-1)) - 1 for img in im_hr]
                im_hr = torch.stack(im_hr, dim=0)
                
                # im_hr = im_hr * 2.0 - 1
                
                
                img_hr = Metrics.tensor2img(im_hr)
                img_lr = Metrics.tensor2img(im_lr)
                img_sr = Metrics.tensor2img(sr_img)
                img_sr_ = Metrics.tensor2img(sr_img_)
                img_df =Metrics.tensor2img(df_img)
                img_df_ = Metrics.tensor2img(df_img_)
                img_mp = Metrics.tensor2img(mp)
                img_s = Metrics.tensor2img(score_img)
                img_mp1 = Metrics.tensor2img(mp1)
                
                mp_k = cv2.applyColorMap(img_mp, cv2.COLORMAP_JET)
                mp_s = cv2.applyColorMap(img_mp1, cv2.COLORMAP_JET)
                # Metrics.save_img(
                #     img_hr, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                # Metrics.save_img(
                #     img_lr, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                # Metrics.save_img(
                #     img_sr, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                
                Metrics.save_img(
                    img_sr_, '{}/{}_sr.png'.format(result_path, lr_file[:-4]))
                Metrics.save_img(
                    img_df_, '{}/{}_df.png'.format(result_path, lr_file[:-4]))
                # Metrics.save_img(
                #     mp_k, '{}/{}_{}_mp.png'.format(result_path, current_step, idx))
                
                # Metrics.save_img(
                #     img_s, '{}/{}_{}_score.png'.format(result_path, current_step, idx))
                # Metrics.save_img(
                #     mp_s, '{}/{}_{}_mp1.png'.format(result_path, current_step, idx))
                print(idx)
                
                
                eval_psnr = Metrics.calculate_psnr(img_sr_, img_hr)
                eval_ssim = Metrics.calculate_ssim(img_sr_, img_hr)
                
                eval_psnr_df = Metrics.calculate_psnr(img_df_, img_hr)
                eval_ssim_df = Metrics.calculate_ssim(img_df_, img_hr)

                avg_psnr += eval_psnr
                avg_ssim += eval_ssim
                
                avg_psnr_df += eval_psnr_df
                avg_ssim_df += eval_ssim_df
                print(f"ID: {idx}; PSNR: {eval_psnr}; SSIM: {eval_ssim}")
                print(f"ID: {idx}; PSNR_DF: {eval_psnr_df}; SSIM_DF: {eval_ssim_df}")

                # if wandb_logger and opt['log_eval']:
                #     wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

                avg_psnr_ = avg_psnr / idx
                avg_ssim_ = avg_ssim / idx
                
                avg_psnr_df_ = avg_psnr_df / idx
                avg_ssim_df_ = avg_ssim_df / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr_))
                logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim_))
                
                logger.info('# Validation # DF_PSNR: {:.4e}'.format(avg_psnr_df_))
                logger.info('# Validation # DF_SSIM: {:.4e}'.format(avg_ssim_df_))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
                    current_epoch, current_step, avg_psnr_, avg_ssim_))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr_DF: {:.4e}, ssim_DF：{:.4e}'.format(
                    current_epoch, current_step, avg_psnr_df_, avg_ssim_df_))

                if wandb_logger:
                    if opt['log_eval']:
                        wandb_logger.log_eval_table()
                    wandb_logger.log_metrics({
                        'PSNR': float(avg_psnr_),
                        'SSIM': float(avg_ssim_)
                    })
                    
                    wandb_logger.log_metrics({
                        'PSNR': float(avg_psnr_df_),
                        'SSIM': float(avg_ssim_df_)
                    })

