
from __future__ import print_function

import argparse
import os
import time, platform

import cv2
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets_test import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from losses import *
from model import DexiNed
from utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result,count_parameters)

IS_LINUX = True if platform.system()=="Linux" else False




def test(checkpoint_path, dataloader, model, device, output_dir, args):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            # images = images[:, [2, 1, 0], :, :]

            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)

            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()

    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("FPS: %f.4" % (len(dataloader)/total_duration))

def testPich(checkpoint_path, dataloader, model, device, output_dir, args):
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            # images2 = images[:, [1, 0, 2], :, :]  #GBR
            images2 = images[:, [2, 1, 0], :, :] # RGB
            preds = model(images)
            preds2 = model(images2)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds,preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args, is_inchannel=True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DexiNed trainer.')
    parser.add_argument('--choose_test_data',
                        type=int,
                        default=1,
                        help='Already set the dataset for testing choice: 0 - 8')
    # ----------- test -------0--


    TEST_DATA = DATASET_NAMES[-1] # max 8
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    test_dir = test_inf['data_dir']
    is_testing =True  


    # Training settings
    #TRAIN_DATA = DATASET_NAMES[-1] # BIPED=0, MDBD=6
    #train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    #train_dir = train_inf['data_dir']


    # Data parameters
    #parser.add_argument('--input_dir',
    #                    type=str,
    #                    default=train_dir,
    #                    help='the path to the directory with the input data.')
    parser.add_argument('--input_val_dir',
                        type=str,
                        default= test_dir, #test_inf['data_dir'],
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='checkpoints',
                        help='the path to output the checkpoints.')
    #parser.add_argument('--train_data',
    #                    type=str,
    #                    choices=DATASET_NAMES,
    #                    default=TRAIN_DATA,
    #                    help='Name of the dataset.')
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')
    #parser.add_argument('--selected_test_dataset',
    #                    type=str,
    #                    default= 'HyperKvasir/Video Sequence 1 Resized', 
    #                    help = 'Name of the specific dataset being utilised')
    parser.add_argument('-test',
                        '--selected_test_specifications',
                        action = 'append',
                        help = 'Generates a list containing information regarding the selected dataset on: Dataset path, Real vs Synthetic, Mixed Dataset or Lone dataset (for the Synthetic Test sets) and also the checkpoint you want to test with.')
                        
    #parser.add_argument('--mixed_dataset',
    #                    type=bool,
    #                    default=False,
    #                    help = 'Boolean which controls whether the synthetic test dataset is mixed or not, used for the purposes of loading the data')
    #parser.add_argument('--fine_tuning_version',
    #                    type=str,
    #                    default='Transfer Learning Version 2', 
    #                    help = 'Instance of Fine Tuned Version')
    #parser.add_argument('--test_list',
    #                    type=str,
    #                    default=test_inf['test_list'],
    #                    help='Dataset sample indices list.')
    #parser.add_argument('--train_list',
    #                    type=str,
    #                    default=train_inf['train_list'],
    #                    help='Dataset sample indices list.')
    parser.add_argument('--is_testing',type=bool,
                        default=True,
                        help='Script in testing mode.')
    parser.add_argument('--double_img',
                        type=bool,
                        default=False,
                        help='True: use same 2 imgs changing channels')  # Just for test
    #parser.add_argument('--resume',
    #                    type=bool,
    #                    default=False,
    #                    help='use previous trained data')  # Just for test
    #parser.add_argument('--checkpoint_data',
    #                    type=str,
    #                    default='10/10_model.pth',# 4 6 7 9 14
    #                    help='Checkpoint path from which to restore model weights from.')
    parser.add_argument('--test_img_width',
                        type=int,
                        default=test_inf['img_width'],
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default=test_inf['img_height'],
                        help='Image height for testing.')
    parser.add_argument('--res_dir',
                        type=str,
                        default= 'result',#'Pretrained'),#'Fine Tuned Version 2'), 
                        help='Result directory')
    #parser.add_argument('--log_interval_vis',
    #                    type=int,
    #                    default=50,
    #                    help='The number of batches to wait before printing test predictions.')

    #parser.add_argument('--epochs',
    #                    type=int,
    #                    default=17,
    #                    metavar='N',
    #                    help='Number of training epochs (default: 25).')
    #parser.add_argument('--lr',
    #                    default=1e-4,
    #                    type=float,
    #                    help='Initial learning rate.')
    #parser.add_argument('--wd',
    #                    type=float,
    #                    default=1e-8,
    #                    metavar='WD',
    #                    help='weight decay (Good 1e-8) in TF1=0') # 1e-8 -> BIRND/MDBD, 0.0 -> BIPED
    #parser.add_argument('--adjust_lr',
    #                    default=[10,15],
    #                    type=int,
    #                    help='Learning rate step size.') #[5,in10]BIRND [10,15]BIPED/BRIND
    #parser.add_argument('--batch_size',
    #                    type=int,
    #                    default=8,
    #                    metavar='B',
    #                    help='the mini-batch size (default: 8)')
    parser.add_argument('--workers',
                        default=16,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard',type=bool,
                        default=True,
                        help='Use Tensorboard for logging.'),
    #parser.add_argument('--img_width',
    #                    type=int,
    #                    default=352,
    #                    help='Image width for training.') # BIPED 400 BSDS 352/320 MDBD 480
    #parser.add_argument('--img_height',
    #                    type=int,
    #                    default=352,
    #                    help='Image height for training.') # BIPED 480 BSDS 352/320
    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)
    parser.add_argument('--crop_img',
                        default=True,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_pixel_values',
                        default=[103.939,116.779,123.68, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    # Tensorboard summary writer

    #tb_writer = None
    #training_dir = os.path.join(args.output_dir,args.train_data)
    #os.makedirs(training_dir,exist_ok=True)
    
    
    dataset_path = args.selected_test_specifications[0] #Path within the data path. 
    
    #Converting string into bool
    
    if args.selected_test_specifications[1] == 'True':
        real_synthetic_bool = True
    elif args.selected_test_specifications[1] == 'False': 
        real_synthetic_bool = False
        
    if args.selected_test_specifications[2] == 'True':
        mixed_bool = True
    elif args.selected_test_specifications[2] == 'False':
        mixed_bool = False
    
    checkpoint_version = args.selected_test_specifications[3]
    
    print('Dataset path is: {}'.format(dataset_path))
    print('Real/Synthetic Boolean is: {}'.format(real_synthetic_bool))
    print('Mixed Dataset Boolean is : {}'.format(mixed_bool))
    
    checkpoint_path = os.path.join(args.output_dir, checkpoint_version)
    
    print('Checkpoint Path is: \n')
    print(f'{checkpoint_path}')
    print('\n')
    
    #if args.tensorboard and not args.is_testing:
    #    from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
    #    tb_writer = SummaryWriter(log_dir=training_dir)
    #    # saving Model training settings
    #    training_notes = ['DexiNed, Xavier Normal Init, LR= ' + str(args.lr) + ' WD= '
    #                      + str(args.wd) + ' image size = ' + str(args.img_width)
    #                      + ' adjust LR='+ str(args.adjust_lr) + ' Loss Function= BDCNloss2. '
    #                      +'Trained on> '+args.train_data+' Tested on> '
    #                      +args.test_data+' Batch size= '+str(args.batch_size)+' '+str(time.asctime())]
    #    info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
    #    info_txt.write(str(training_notes))
    #    info_txt.close()

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)
    # model = nn.DataParallel(model)
    #ini_epoch =0
    #if not args.is_testing:
    #    if args.resume:
    #        ini_epoch=11
    #        model.load_state_dict(torch.load(checkpoint_path,
    #                                     map_location=device))
    #        print('Training restarted from> ',checkpoint_path)
    #    dataset_train = BipedDataset(args.input_dir,
    #                                 img_width=args.img_width,
    #                                 img_height=args.img_height,
    #                                 mean_bgr=args.mean_pixel_values[0:3] if len(
    #                                     args.mean_pixel_values) == 4 else args.mean_pixel_values,
    #                                 train_mode='train',
    #                                 arg=args
    #                                 )
    #    dataloader_train = DataLoader(dataset_train,
    #                                  batch_size=args.batch_size,
    #                                  shuffle=True,
    #                                  num_workers=args.workers)

    #test_dir = os.path.join(args.input_val_dir, dataset_path)
    test_dir = args.input_val_dir
    
    dataset_test = TestDataset(test_dir,
                              dataset_path,
                              real_synthetic_bool,
                              mixed_bool,
                              test_data=args.test_data,
                              img_width=args.test_img_width,
                              img_height=args.test_img_height,
                              mean_bgr=args.mean_pixel_values[0:3] if len(
                                  args.mean_pixel_values) == 4 else args.mean_pixel_values,
                              arg = args
                              )
    dataloader_test = DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
    # Testing
    if args.is_testing:
         
        constituent_path = []
        while True:
            parts = os.path.split(checkpoint_version)
            if parts[0] == checkpoint_version:  
                constituent_path.insert(0, parts[0])
                break
            elif parts[1] == checkpoint_version: # sentinel for relative paths
                constituent_path.insert(0, parts[1])
                break
            else:
                checkpoint_version = parts[0]
                constituent_path.insert(0, parts[1])
       
        fine_tuning_version = constituent_path[0]
        print(fine_tuning_version)
        
        output_dir = os.path.join(args.res_dir, fine_tuning_version, dataset_path)
        print(f"output_dir: {output_dir}")
        if args.double_img:
            # predict twice an image changing channels, then mix those results
            testPich(checkpoint_path, dataloader_test, model, device, output_dir, args)
        else:
            test(checkpoint_path, dataloader_test, model, device, output_dir, args)

        num_param = count_parameters(model)
        print('-------------------------------------------------------')
        print('Number of parameters of current DexiNed model:')
        print(num_param)
        print('-------------------------------------------------------')
        return

    #criterion = bdcn_loss2 # hed_loss2 #bdcn_loss2

    #optimizer = optim.Adam(model.parameters(),
    #                       lr=args.lr,
    #                       weight_decay=args.wd)

    # Main training loop
    ''' seed=1021
    adjust_lr = args.adjust_lr
    lr2= args.lr
    for epoch in range(ini_epoch,args.epochs):
        if epoch%7==0:

            seed = seed+1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        # Create output directories
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = lr2*0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2

        output_dir_epoch = os.path.join(args.output_dir,args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.test_data + '_res')
        os.makedirs(output_dir_epoch,exist_ok=True)
        os.makedirs(img_test_dir,exist_ok=True)

        avg_loss =train_one_epoch(epoch,
                        dataloader_train,
                        model,
                        criterion,
                        optimizer,
                        device,
                        args.log_interval_vis,
                        tb_writer,
                        args=args)
        validate_one_epoch(epoch,
                           dataloader_val,
                           model,
                           device,
                           img_test_dir,
                           arg=args)

        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        if tb_writer is not None:
            tb_writer.add_scalar('loss',
                                 avg_loss,
                                 epoch+1)
        print('Current learning rate> ', optimizer.param_groups[0]['lr'])
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('~Number of parameters of current DexiNed model:')
    print(num_param)
    print('-------------------------------------------------------')
    '''

if __name__ == '__main__':
    args = parse_args()
    main(args)
