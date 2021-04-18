# Dependencies
- will require to install Opencv library

Run main.py from terminal and give command line input as path to query image, the code will display query image first, click enter and it will display matched reference image.
train_model is used for training, that is extracting features from reference images. The obtained descriptors along with names(index) of reference images will be stored as pickle files as pickle_descriptors_SIFT.pck (if algorithm used for feature extraction is SIFT)
The repository already contains trained pickle files on both SIFT as well as KAZE.This training is done from referenceImages folder.
