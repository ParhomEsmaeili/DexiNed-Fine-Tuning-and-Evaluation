import numpy as np
import matplotlib.pyplot as plt 
import sklearn.metrics as skl
import cv2

class IOU:
    '''
    Class for computing the IoU metric on an input image with a ground truth, or for comparing two images.
    '''
    def __init__(
                self,
                image_1: np.ndarray,
                image_2: np.ndarray,
                threshold: float,
                image_to_image: bool
            ):
        '''
        Inputs:
        image_1: Image which is being compared to the ground truth (or a successive frame) 
        image_2: Ground truth mask (or successive image frame)
        
        -- Both images have pixels in the range [0,1]

        threshold: Threshold for binarising the input image(s) for comparison with a ground truth.
        image_to_image: Boolean which controls whether a binarising threshold is applied to both images, depending on
        whether image_2 is a ground truth mask or a comparison image (i.e. comparison of successive image frames).
        '''
        #
        self.image_1 = image_1
        self.image_2 = image_2
        self.threshold = threshold
        self.image_to_image = image_to_image
        
         
    def main(self):
        failure_flag = 0
        iou = None

        if self.image_to_image:
            image_1_binarised, image_2_binarised = self.preprocessing()
            try:
                iou = self.iou(image_1_binarised, image_2_binarised)
                #print('Intersection over the union is {}'.format(iou))
                #Additionally, other than zero division errors, a miniuscule IoU value is treated as a failure case due to the minimal/unmeaningful overlap.
                    #print('Insufficient overlap for Intersection over Union')
            except ZeroDivisionError:
                #print('Insufficient overlap for Intersection over Union')
                failure_flag = 1
        else:
            image_1_binarised = self.preprocessing()
            try:
                iou = self.iou(image_1_binarised, self.image_2)
                #print('Intersection over the union is {}'.format(iou))
            except ZeroDivisionError:
                #print('Insufficient overlap for Intersection over Union')
                failure_flag = 1
            #Additionally, other than zero division errors, a miniuscule IoU value is treated as a failure case due to the minimal/unmeaningful overlap.
            
        
        return iou, failure_flag

    def preprocessing(self):
        '''
        Inputs: 
        Self: image_1, image_2 and image_to_image boolean. 

        Outputs: 
        Depending on the image_to_image boolean outputs either both of the images binarised by the threshold, or just image_1.
        '''
        if self.image_to_image:
            image_1_binarised = np.where(self.image_1 > self.threshold, 1, 0)
            image_2_binarised = np.where(self.image_2 > self.threshold, 1, 0)
            #cv2.imshow('im1_binarise', (image_1_binarised*255).astype(np.uint8))
            #cv2.imshow('im2 binarised', (image_2_binarised*255).astype(np.uint8))
            #cv2.waitKey(0)
            return image_1_binarised, image_2_binarised
        else:
            image_1_binarised = np.where(self.image_1 > self.threshold, 1, 0)
            return image_1_binarised
    
    def iou(self, image_1, image_2):
        '''
        Inputs:

        image_1: Binarised numpy array where edges have pixel values of 1 and non edges have pixel values of 0.
        image_2: Ground truth mask (or binarised numpy array) where edges have pixel values of 1 and non edges have pixel
        values of 0.

        Outputs:
        Intersection over the union metric using the two input images.
        '''

        #Converting the binarised image arrays to boolean datatype.
        image_1_boolean_array = np.array(image_1, dtype = bool)
        image_2_boolean_array = np.array(image_2, dtype = bool)

        #Computing the area (number of pixels) which are edges for both images.  
        image_1_area = np.count_nonzero(image_1_boolean_array)
        image_2_area = np.count_nonzero(image_2_boolean_array)

        #Computing the area (number of pixels) where image 1 and image 2 share edges in common. 
        intersection = np.count_nonzero(np.logical_and(image_1_boolean_array, image_2_boolean_array))
        
        #Union = Area of Edges Image 1 + Area of Edges Image 2 - Intersection 
        union = image_1_area + image_2_area - intersection
        intersection_over_union = intersection/union

        #cv2.imshow('im1', np.array(255 * image_1, dtype = np.uint8))
        #cv2.imshow('im2', np.array(255 * image_2, dtype = np.uint8))
        #cv2.imshow('overlap', np.array(255 * np.logical_and(image_1_boolean_array, image_2_boolean_array), dtype = np.uint8))
        #cv2.waitKey(0)
        return intersection_over_union
