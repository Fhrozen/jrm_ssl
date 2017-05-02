# jrm_ssl

Files for the paper: "Sound Source Localization using Deep Residual Learning"

the programs run most on Python (Windows - Linux)

## Requirements

[Chainer](https://github.com/pfnet/chainer) - Install from pip

[Hark](http://www.hark.jp/) - to obtain Audio Features

## Training

Run ./chainer_train.py t -C $(config_file) from training folder to train a model

## Evaluation

The forwarding file is located in the microcone folder 

- Run ./ssl_test.py $(DATE_OF_TRAINED_MODEL) to forward the audio files (Any corpus, any language is fine)
- Run ./compile_results.py to obtain the block accuracy (median angle) - change the exp variable inside the
file according to the folder you want to test
- Run ./eval_correc_acc.py to obtain the point-to-point accuracy  - change the exp variable inside the
file according to the folder you want to test

## Folder Structure

- dataset_preparation : Two examples of the dataset prepared for the training
- microcone : Files to evaluate any model and a network example to be trained
- python_utils : extra files for training, preparing data, etc.
- training : files for training a network
- training_files : an example of a generated network and the files to test

## Impulses Response

To generate the impulse use [ISM](http://www.eric-lehmann.com/) of Eric A. Lehmann

Information of Microcone position microphones at [HARK](http://www.hark.jp/) Supported Hardwares

## Publication

[JRM Vol.29 No.1 (Feb. 20, 2017)](https://www.fujipress.jp/jrm/rb/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
