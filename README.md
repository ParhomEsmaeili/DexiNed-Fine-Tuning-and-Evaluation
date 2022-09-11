# DexiNed Fine Tuning and Evaluation Repository

This is the repository which contains the adapted DexiNed code from Soria et al. https://github.com/xavysp/DexiNed. It adapts some of the base scripts from DexiNed for the purposes of fine tuning and testing. The repository also contains the scripts utilised for evaluation (i.e. the ODS/OIS F-score and AP Scores and the Temporal Consistency).

# Prerequisites:

For the purposes of testing, you will need to fill the checkpoints folder with some of the checkpoints of the fine tuning versions implemented in the dissertation. These can be downloaded from [here](https://drive.google.com/drive/u/2/folders/11jE3KV-cE1BbMNU8t5QQr-rEMr-L9dBt). To test on something, we have also provided the HyperKvasir sequences which were tested on for the Dissertation, in addition to one of the combined test datasets and one of the isolated test datasets (Datasets 1 and 4 . These are all located [here](https://drive.google.com/drive/u/2/folders/1bxryu9bDaOi53Lbky4gLtXXyzJ-iY9HY).

Required Packages for the Python Scripts include:

* [Python 3.7](https://www.python.org/downloads/release/python-370/g)
* [Pytorch >=1.4](https://pytorch.org/) (Last test 1.9)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Matplotlib](https://matplotlib.org/3.1.1/users/installing.html)
* [Kornia](https://kornia.github.io/)
* [Natsort](https://pypi.org/project/natsort/)
* Other package like NumPy, h5py, PIL, json. 

For the purposes of computing ODS/OIS F-scores and AP scores you will require a couple of MATLAB toolboxes:




## Project Architecture

```
├── data                        # Sample images for testing (paste your image here)
|   ├── lena_std.tif            # Sample 1
├── DexiNed-TF2                 # DexiNed in TensorFlow2 (in construction)   
├── figs                        # Images used in README.md
|   └── DexiNed_banner.png      # DexiNed banner
├── legacy                      # DexiNed in TensorFlow1 (presented in WACV2020)
├── utls                        # A series of tools used in this repo
|   └── image.py                # Miscellaneous tool functions
├── datasets.py                 # Tools for dataset managing 
├── dexi_utils.py               # New functions still not used in the currecnt version
├── losses.py                   # Loss function used to train DexiNed (BDCNloss2)
├── main.py                     # The main python file with main functions and parameter settings
                                # here you can test and train
├── model.py                    # DexiNed class in pythorch
