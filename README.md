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


