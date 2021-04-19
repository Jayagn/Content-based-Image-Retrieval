# Dependencies
- will require to install Opencv library
# Run
- Run main.py from terminal and give command line input as path to query image, the code will display query image first, click enter and it will display matched reference image.
- For example: python main.py path/to/queryimage (Make sure there are no spaces in image file name)
- train_model is used for training, that is extracting features from reference images. 
- The obtained descriptors along with names(index) of reference images will be stored as pickle files as pickle_descriptors_SIFT.pck (if algorithm used for feature extraction is SIFT)
- The repository already contains trained pickle files on both SIFT as well as KAZE.This training is done from referenceImages folder.

![caption](https://github.com/Jayagn/jayagn13/blob/main/bandicam%202021-04-18%2019-38-35-971.gif)
