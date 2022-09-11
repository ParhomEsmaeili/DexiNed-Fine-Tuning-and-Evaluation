import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import natsort
import pickle
from metric_utils import metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Metric Evaluation')

    parser.add_argument('--input_dir',
                        type = str,
                        default = 'result',
                        help = 'Subdirectory which contains the sequences which are going to be evaluated.')
    parser.add_argument('--output_dir',
                        type = str,
                        default = '2_class_TC_metric_output',
                        help= 'Subdirectory which contains the results, e.g. pkl files or histograms.')

    parser.add_argument('--masked_result_output_dir_edgeclass',
                        type=str,
                        default='mIOU_masked_edgeclass',
                        help = 'Directory containing the results of the Network after masking to remove artefacts.')
    parser.add_argument('--masked_result_output_dir_nonedgeclass',
                        type=str,
                        default='mIOU_masked_warped_nonedgeclass',
                        help = 'Directory containing the results of the Network after masking to remove artefacts.')

    parser.add_argument('--binarised_and_masked_output_dir',
                        type = str,
                        default = 'binarised_masked_results',
                        help = 'Directory containing the binarised results of the Network after masking to remove artefacts')
    #parser.add_argument('--network_version',
    #                    type = str,
    #                    default = 'Fine Tuning Version 3 Zeta Epoch 17',
    #                    help = 'Version of the network for which the results are being evaluated.')
    parser.add_argument('--mask_directory',
                        type=str,
                        default='Video Sequence Masks',
                        help = 'Directory containing the folders containing masks for different datasets and sequences')
    #parser.add_argument('--dataset',
    #                    type = str,
    #                    default = 'HyperKvasir',
    #                    help = 'Name of the Dataset Origin, e.g. HyperKvasir, SUN etc.')
    #parser.add_argument('--selected_sequence',
    #                    type = str,
    #                    default = 'Video Sequence 1 Resized',
    #                    help = 'Name of the sequence being evaluated')
    parser.add_argument('--masked',
                        type = bool,
                        default = True,
                        help = 'Boolean that controls whether the selected sequence being evaluated needs to be masked')

    parser.add_argument('-specs',
                        '--test_specifications',
                        action = 'append',
                        help = 'Generates a list containing information regarding: Dataset path, and the version of the network in question (e.g. UCL only, EndoMapper only, both, etc).')
    args = parser.parse_args()
    return args

def masking_artefacts(image_path_1, image_path_2, mask_path, masked_edgeclass_img_dir, masked_nonedgeclass_img_dir, binarised_masked_img_dir):
    '''Function which removes consistent artefacts across a whole sequence of fused edge maps that should not be utilised within image registration,
    i.e. date, time text or the outline of an image frame.
    
    Inputs: 
    
    image_path_1: The path of the first edge map image which is going to be masked.
    image_path_2: The path of the second edge map image which is going to be masked.
    mask_path: Mask being utilised for the selected sequence.
    masked_img_dir: The path of the directory where the masked images are going to be stored.

    Outputs:

    masked_images: Edge map images which have had their artefacts removed by applying the mask. 

    Also saves the masked image to a directory for verification purposes (or use elsewhere).
    '''
    image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

    image_1_inverted_grayscale = (255 - image_1)
    image_2_inverted_grayscale = (255 - image_2)
    
    #binarised_img_1 = np.where(image_1 < 0.2*255, 255, 0)
    #binarised_img_2 = np.where(image_2 < 0.2*255, 255, 0)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255
    

    masked_image_1_edgeclass = (image_1_inverted_grayscale * mask).astype(np.uint8)
    masked_image_2_edgeclass = (image_2_inverted_grayscale * mask).astype(np.uint8) 
    
    masked_image_1_nonedgeclass = (image_1 * mask).astype(np.uint8)
    #masked_image_2_nonedgeclass = (image_2 * mask).astype(np.uint8)


    #binarised_masked_image_1 = (binarised_img_1 * mask).astype(np.uint8)  #image_1 * mask 
    #binarised_masked_image_2 = (binarised_img_2 * mask).astype(np.uint8) #image_2 * mask 

    #cv2.imshow('masked_img1', masked_image_1_nonedgeclass)
    #cv2.imshow('masked_imgcorrected', binarised*mask/255)
    #cv2.imshow('masked_img2', masked_image_2_nonedgeclass)
    #cv2.waitKey(0)

    image_name_1 = os.path.split(image_path_1)[1]
    image_name_2 = os.path.split(image_path_2)[1]


    cv2.imwrite(os.path.join(masked_edgeclass_img_dir, image_name_1), masked_image_1_edgeclass)
    cv2.imwrite(os.path.join(masked_edgeclass_img_dir, image_name_2), masked_image_2_edgeclass)
    
    cv2.imwrite(os.path.join(masked_nonedgeclass_img_dir, image_name_1), masked_image_1_nonedgeclass)
    #cv2.imwrite(os.path.join(masked_nonedgeclass_img_dir, image_name_2), masked_image_2_nonedgeclass)
    
    #cv2.imwrite(os.path.join(binarised_masked_img_dir, image_name_1), binarised_masked_image_1)
    #cv2.imwrite(os.path.join(binarised_masked_img_dir, image_name_2), binarised_masked_image_2)
    return masked_image_1_edgeclass, masked_image_2_edgeclass, image_1, image_2, mask                          #binarised_masked_image_1, binarised_masked_image_2,  

def align_images(image_1, image_2, image_1_nonedgeclass, image_2_nonedgeclass, number_of_levels, warp_mode, warp, criteria):
    '''Function estimates the affine warp using ECC alignment by finding warps at each scales in a pyramid to use as initialisations for the next scale.''' 
    
    warp = warp * np.array([[1,1,2],[1,1,2]], dtype = np.float32)**(1-number_of_levels) #np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)**(1-number_of_levels)  
    #Factor of 2 in the warp modification matrix is to account for the fact that the translation parameter in the warp matrix would have to be be scaled down 
    #accordingly if warping between scaled down images.
    failure_flag = 0
    for level in range(number_of_levels):
        sz = image_1.shape

        scale = 1/2**(number_of_levels-1-level)

        image_1_rescaled = cv2.resize(image_1, None, fx= scale, fy = scale, interpolation=cv2.INTER_AREA)
        image_2_rescaled = cv2.resize(image_2, None, fx= scale, fy= scale, interpolation=cv2.INTER_AREA)

        image_2_nonedgeclass_rescaled = cv2.resize(image_2_nonedgeclass, None, fx= scale, fy = scale, interpolation=cv2.INTER_AREA)
        try:
            (cc,warp) = cv2.findTransformECC(image_1_rescaled, image_2_rescaled, warp, warp_mode, criteria, None)
        except:
            if level == range(number_of_levels)[-1]:
                failure_flag = 1
            else: 
                continue

        if level != number_of_levels-1:
        
        # scale up for the next pyramid level
            rescaling = np.array([[1, 1, 2], [1, 1, 2]], np.float32)
            warp = warp * rescaling #tng.astype(np.float32)) 

    #print(image_2_rescaled.shape)
    #print(masked_image_2_nonedgeclass_rescaled.shape)
    #print(image_2_rescaled.dtype)
    #print(masked_image_2_nonedgeclass_rescaled.dtype)
    #print(warp.shape)
    #print(warp.dtype)
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            image_2_edgeclass_aligned = cv2.warpPerspective (image_2_rescaled, warp, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            image_2_nonedgeclass_aligned = cv2.warpPerspective (image_2_nonedgeclass_rescaled, warp, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
            # Use warpAffine for Translation, Euclidean and Affine
            image_2_edgeclass_aligned = cv2.warpAffine(image_2_rescaled, warp, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            image_2_nonedgeclass_aligned = cv2.warpAffine(image_2_nonedgeclass_rescaled, warp, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    #cv2.imshow('image1', masked_image_1_nonedgeclass)
    #cv2.imshow('image2', masked_image_2_nonedgeclass)
    #cv2.imshow('image2_aligned', image_2_nonedgeclass_aligned)
    #cv2.waitKey(0)
    return image_2_edgeclass_aligned, image_2_nonedgeclass_aligned, failure_flag


def successive_frame_comparison(fused_edge_map_dir, frame_name_pair, image_mask_path, network_version, dataset_path, args):
    '''Function which computes the Intersection over the Union using Image registration between two successive frames
    
    Inputs:
    frame1_path: Path to the first of the two frames,
    frame2_path: Path to the second of the two frames.
    image_mask_path: Path for the Mask for the selected sequence being examined (to remove image artefacts such as frame outlines)
    dataset_path: Path for the dataset sequence which is being evaluated.
    Output:

    '''

    frame_1_path = os.path.join(fused_edge_map_dir, frame_name_pair[0])
    frame_2_path = os.path.join(fused_edge_map_dir, frame_name_pair[1])
    
    masked_edgeclass_images_output_dir = os.path.join(os.getcwd(), args.masked_result_output_dir_edgeclass, network_version, dataset_path)
    masked_nonedgeclass_images_output_dir = os.path.join(os.getcwd(), args.masked_result_output_dir_nonedgeclass, network_version, dataset_path)
    binarised_masked_images_output_dir = os.path.join(os.getcwd(), args.binarised_and_masked_output_dir, network_version, dataset_path)

    if not os.path.exists(masked_edgeclass_images_output_dir):
        os.makedirs(masked_edgeclass_images_output_dir)
    if not os.path.exists(masked_nonedgeclass_images_output_dir):
        os.makedirs(masked_nonedgeclass_images_output_dir)
    if not os.path.exists(binarised_masked_images_output_dir):
        os.makedirs(binarised_masked_images_output_dir)

    # Masking out the outline in the sequence frame so that it does not influence the result of the image alignment. 
    masked_image_1_edgeclass, masked_image_2_edgeclass, image_1, image_2, mask = masking_artefacts(frame_1_path, frame_2_path, image_mask_path, masked_edgeclass_images_output_dir, masked_nonedgeclass_images_output_dir, binarised_masked_images_output_dir)
    
    '''An image warp will be applied to the first image in an image pair to warp the image onto the latter, in order to compute the intersection over the union. The warp is
    determined via direct image alignment by using the ECC algorithm. In order to initialise the algorithm closer to the optimum, warps are computed at multiple scales 
    in a pyramid of scales, such that for each successive scale in a pyramid the initialisation is given by the result at the prior scale.'''

    #Number of scale levels being utilised in pyramid
    number_of_levels = 4
    warp_mode = cv2.MOTION_EUCLIDEAN 
    warp = np.array([[1,0,0],[0,1,0]], dtype = np.float32)  
    
    #Criteria for the ECC algorithm to terminate:

    #Maximum number of iterations allowed before forced termination of the optimisation algorithm.
    iteration_limit = 5000
    #Threshold for the change in the cross correlation between successive iterations in the optimisation algorithm.
    convergence_threshold = 1e-6
    termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iteration_limit,  convergence_threshold)

    warped_img_edgeclass, warped_img_nonedgeclass, ecc_failure_flag = align_images(masked_image_1_edgeclass, masked_image_2_edgeclass, image_1, image_2, number_of_levels, warp_mode, warp, termination_criteria)
    
    #Reapplying the mask to the warped edgeclass image to remove/blackout the pixels which would be "outside of the image frame".
    masked_warped_img_edgeclass = (warped_img_edgeclass * mask).astype(np.uint8)

    #First application of the mask to the non edgeclass images to remove the pixels outside of the image frame.
    masked_image_1_nonedgeclass = (image_1 * mask).astype(np.uint8)
    masked_warped_img_nonedgeclass = (warped_img_nonedgeclass * mask).astype(np.uint8)

    #We will also save the warped images for a sanity check.
    if not ecc_failure_flag:
        image_name_2 = os.path.split(frame_2_path)[1] 
        cv2.imwrite(os.path.join(masked_nonedgeclass_images_output_dir, image_name_2), masked_warped_img_nonedgeclass)

    #If warping procedure was not a failure then compute the Intersection over the Union metric. Initialise placeholder values for the metrics and flags.
    mean_iou_metric = None
    iou_edge_failure_flag = 0
    iou_nonedge_failure_flag = 0
    if not ecc_failure_flag:
        threshold = 0.8
        #Instantiating the IOU Class.
        iou_metric_class_edge = metrics.IOU(masked_image_1_edgeclass, masked_warped_img_edgeclass, np.round(threshold * 255), True)
        iou_edge_metric, iou_edge_failure_flag = iou_metric_class_edge.main()

        iou_metric_class_nonedge = metrics.IOU(masked_image_1_nonedgeclass, masked_warped_img_nonedgeclass, np.round((1-threshold) * 255), True)
        iou_nonedge_metric, iou_nonedge_failure_flag = iou_metric_class_nonedge.main()
        #print(iou_metric)
        #try: 
        #    if iou_metric > optimal_iou_metric:
        #        optimal_iou_metric = iou_metric 
        #        optimal_threshold = threshold 
        #        optimal_iou_failure_flag = iou_failure_flag
        #except TypeError:
        #    pass #In the instance where the iou overlap is insufficient, in which case the failure flag picks it up.

        #print('Image registration successful!')


        #ECC alignment has a tendency to produce very poor affine warps which are effectively a failure case.
        if not iou_edge_failure_flag and not iou_nonedge_failure_flag:
            #print('Image registration successful!')
            mean_iou_metric = (iou_edge_metric + iou_nonedge_metric)/2
            print('Mean Intersection over the union is {}'.format(mean_iou_metric))
        else:
            print('Insufficient or no overlap.')
    else:
        print('ECC Image registration totally failed!')

    return mean_iou_metric, ecc_failure_flag, iou_edge_failure_flag, iou_nonedge_failure_flag


def main(args):
    
    dataset_path = args.test_specifications[0]
    
    #if args.test_specifications[1] == 'True':
    #    real_synthetic_bool = True
    #elif args.test_specifications[1] == 'False':
    #    real_synthetic_bool = False

    #if args.test_specifications[2] == 'True':
    #    mixed_bool = True
    #elif args.test_specifications[2] == 'False':
    #    mixed_bool = False 
        
    network_version = args.test_specifications[1]




    current_dir = os.getcwd()
    overall_dir = os.path.split(current_dir)[0]

    #Directory containing the results of the DexiNed network.
    fused_edge_map_dir = os.path.join(overall_dir, args.input_dir, network_version, dataset_path, 'fused')
    #Directory to print the results of the script, i.e. the metric values. 
    output_dir = os.path.join(current_dir, args.output_dir, network_version, dataset_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if args.masked:
        mask_path = os.path.join(current_dir, args.mask_directory, dataset_path, 'mask.jpg')
    
    #Extract the list of image names which will be evaluated; first sorts the images in the list such that successive frames are ordered correctly.
    image_name_list = natsort.natsorted(os.listdir(fused_edge_map_dir))
    image_pair_list = image_name_list
    #Form a set of image pairs by pairing each frame with the frame two frames after it? (THIS ONLY APPLIES TO THE ORIGINAL DATA FPS SETTINGS).
    image_pairs =  list(zip(image_pair_list, image_pair_list[1:]))

    #Loop through the list of image pairs to warp the image pairs onto one another and compute the IoU values. 
    metric_vals = []
    ecc_failures = 0
    iou_failures = 0
    ecc_failure_pairs = []
    iou_failure_pairs = []
    registration_success_pairs = []

    for pair_num, image_pair in enumerate(image_pairs):
        print('Pair number is {}'.format(pair_num))
        mean_iou, ecc_failure_flag, iou_edge_failure_flag, iou_nonedge_failure_flag = successive_frame_comparison(fused_edge_map_dir, image_pair, mask_path, network_version, dataset_path, args)
        
        if not ecc_failure_flag and not iou_edge_failure_flag and not iou_nonedge_failure_flag:
            metric_vals.append(mean_iou)
            registration_success_pairs.append(pair_num)
        elif ecc_failure_flag:
            ecc_failures += ecc_failure_flag
            ecc_failure_pairs.append(pair_num)
        elif iou_edge_failure_flag or iou_nonedge_failure_flag:
            iou_failures += 1 #iou_failure_flag
            iou_failure_pairs.append(pair_num)

    print('ECC Convergence Failures {}'.format(ecc_failures))
    print('Mean IOU Failures {}'.format(iou_failures))

    ecc_timeseries_failure_flag = []
    iou_timeseries_failure_flag = []
    iou_timeseries_values = []   #Note that this does not include the frames in which it totally fails, i.e. no plot points for those points.

    for image_pair_index in range(len(image_pairs)):
        if image_pair_index in ecc_failure_pairs:
            ecc_timeseries_failure_flag.append(True)
        else:
            ecc_timeseries_failure_flag.append(False)
    
    for image_pair_index in range(len(image_pairs)):
        if image_pair_index in iou_failure_pairs:
            iou_timeseries_failure_flag.append(True)
        else:
            iou_timeseries_failure_flag.append(False)
    
    plt.figure(figsize=(12,6))
    plt.scatter(registration_success_pairs, metric_vals,)
    plt.style.use("seaborn")
    plt.savefig(os.path.join(output_dir, 'iou_scatter_plot.png'), dpi = 200)
    
    plt.figure(figsize=(12,12))
    hist = np.array(metric_vals)
    plt.hist(hist, bins=20)
    plt.style.use("seaborn")
    plt.savefig(os.path.join(output_dir, 'iou_histogram_plot.png'), dpi = 200)
    #align_images(args)


    ecc_timeseries_array = np.array(ecc_timeseries_failure_flag).reshape((1, len(image_pairs)))
    iou_timeseries_array = np.array(iou_timeseries_failure_flag).reshape((1, len(image_pairs)))
    stacked_logical_arrays = np.concatenate((ecc_timeseries_array, iou_timeseries_array), axis=0)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(ecc_timeseries_array, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    #labels = ['ECC Convergence Bool']
    #ax.set_yticks(np.arange(len(labels)))
    #ax.set_yticklabels(labels)
    ax.set_title('ECC Convergence Bool')
    plt.savefig(os.path.join(output_dir, 'ecc_bool_plot.png'), dpi = 200)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(iou_timeseries_array, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    #labels = ['IOU Failure Bool']
    #ax.set_yticks(np.arange(len(labels)))
    #ax.set_yticklabels(labels)
    ax.set_title('IOU Failure Bool')

    plt.savefig(os.path.join(output_dir, 'iou_bool_plot.png'), dpi = 200)

    print('The average mIOU is {}'.format(np.mean(metric_vals)))
    print('The median mIOU is {}'.format(np.median(metric_vals)))
    
    with open(os.path.join(output_dir, 'miou_value_array'), 'wb') as fp:
        pickle.dump(metric_vals, fp)
    with open(os.path.join(output_dir, 'average_miou'), 'wb') as fp:
        pickle.dump(np.mean(metric_vals), fp)
    with open(os.path.join(output_dir, 'median_miou'), 'wb') as fp:
        pickle.dump(np.median(metric_vals), fp)

    with open(os.path.join(output_dir, 'ECC_failure_count'),'wb') as fp:
        pickle.dump(ecc_failures, fp)

    with open(os.path.join(output_dir, 'IOU_failure_count'),'wb') as fp:
        pickle.dump(iou_failures, fp)
    
if __name__=='__main__':
    args = parse_args()
    main(args)
    print('Finished alignment of sequence of images')
