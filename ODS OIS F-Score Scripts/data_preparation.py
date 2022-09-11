import os
import cv2
import natsort
import numpy as np
import argparse
import scipy.io

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--result_path',
    #                    type=str,
    #                    default = 'ODS/OIS F-Scores',                         
    #                    help='Path where the results will be saved')
     
    parser.add_argument('--input_dir',
                        type = str,
                        default = 'result',
                        help = 'Subdirectory which contains the sequences which are going to be evaluated.')
    
    parser.add_argument('-specs',
                        '--test_specifications',
                        action = 'append',
                        help = 'Generates a list containing information regarding:Dataset path, Mixed Dataset or Lone dataset, and network version.')
    args = parser.parse_args()
    return args




def main(args):

    dataset_name   = args.test_specifications[0]

    if args.test_specifications[1] == 'True':
        mixed_bool = True
    elif args.test_specifications[1] == 'False':
        mixed_bool = False
    

    current_dir = os.getcwd()
    overall_dir = os.path.split(current_dir)[0]

    if len(args.test_specifications) == 3:
        network_version = args.test_specifications[2]
        edgemap_path = os.path.join(overall_dir, 'result', network_version, dataset_name, 'fused')
        output_edgemap_path = os.path.join(current_dir, 'result', network_version, dataset_name, 'fused')

        if not os.path.isdir(output_edgemap_path):
            os.makedirs(output_edgemap_path)

        for img in os.listdir(edgemap_path):
            img_name_withoutext = os.path.splitext(img)[0]
            new_img_name = img_name_withoutext + '.png'
            
            image = cv2.imread(os.path.join(edgemap_path, img), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(output_edgemap_path, new_img_name), 255 - image)
    
    

    #print(current_dir)
    #print(overall_dir)
    #prediction_path = os.path.join(current_dir, 'result', network_version, dataset_name, 'fused')

    if mixed_bool:
        gt_directory_path = os.path.join(overall_dir, 'data','Test Dataset GroundTruths', dataset_name, 'test', 'rgbr', 'real')
        subdir_list = os.listdir(gt_directory_path)
        combined_output_gt_dir_path = os.path.join(current_dir, 'data', 'Test Dataset GroundTruths Raw', dataset_name, 'test', 'rgbr','real', 'combined_real')

        if not os.path.isdir(combined_output_gt_dir_path):
            os.makedirs(combined_output_gt_dir_path)
        
        for subdir in subdir_list:
           for img in os.listdir(os.path.join(gt_directory_path, subdir)):
            img_mat_ext = os.path.splitext(img)[0]
            new_img_name  = subdir + img_mat_ext + '.mat'

            image_temp = cv2.imread(os.path.join(gt_directory_path, subdir, img), cv2.IMREAD_GRAYSCALE)
            image_output = np.where(image_temp > 0.5 * 255, 1, 0).astype(np.bool_)
 
            scipy.io.savemat(os.path.join(combined_output_gt_dir_path, new_img_name), mdict={'arr': image_output})

            #cv2.imwrite(os.path.join(combined_output_gt_dir_path, new_img_name), image_output)

        '''if len(args.test_specifications)==3:

            edgemap_path = os.path.join(overall_dir, 'result', network_version, dataset_name, 'fused')
            output_edgemap_path = os.path.join(current_dir, 'result', network_version, dataset_name, 'fused')

            for img in os.listdir(edgemap_path):
                img_name_withoutext = os.path.splitext(img)[0]
                new_img_name = img_name_withoutext + '.png'
                
                image = cv2.imread(os.path.join(edgemap_path, img))
                cv2.imwrite(os.path.join(output_edgemap_path, new_img_name), image)'''
    else:
        gt_directory_path = os.path.join(overall_dir, 'data','Test Dataset GroundTruths', dataset_name, 'test', 'rgbr', 'real')
        #subdir_list = os.listdir(gt_directory_path)
        output_gt_dir_path = os.path.join(current_dir, 'data', 'Test Dataset GroundTruths Raw', dataset_name, 'test', 'rgbr','real')

        if not os.path.isdir(output_gt_dir_path):
            os.makedirs(output_gt_dir_path)
        
    
        for img in os.listdir(gt_directory_path):
            img_mat_ext = os.path.splitext(img)[0]
            new_img_name  = img_mat_ext + '.mat'

            image_temp = cv2.imread(os.path.join(gt_directory_path, img), cv2.IMREAD_GRAYSCALE)
            image_output = np.where(image_temp > 0.5 * 255, 1, 0).astype(np.bool_)

            scipy.io.savemat(os.path.join(output_gt_dir_path, new_img_name), mdict={'arr': image_output})

        
        
        
        
        #print('Mixed Dataset')
        #gt_path = os.path.join(combined_output_dir_path)
    #else:
        #print('Unmixed dataset')
        #gt_path = os.path.join(current_dir, 'data','Test Dataset GroundTruths', dataset_name, 'test', 'rgbr', 'real')

if __name__=='__main__':
    
    args = parse_args()
    main(args)