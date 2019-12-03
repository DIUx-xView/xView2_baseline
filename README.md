# xView 2 Challenge 

xview2-baseline

Copyright 2019 Carnegie Mellon University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Released under a BSD-3-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of the following Third-Party Software subject to its own license:
1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.

DM19-0988

## xView 

xView2 is a challenge set forth by [DIU](https://diu.mil) for assessing damaged buildings after a natural disaster. The challenge is open to all to participate and more information can be found at the [xview2](https://xview2.org) challenge site.

## xBD

xBD is the name of the dataset that was generated for this challenge. The dataset contains over 45,000KM<sup>2</sup> of polygon labeled pre and post disaster imagery. The dataset provides the post-disaster imagery with transposed polygons from pre over the buildings, with damage classification labels. See the [xBD paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf) for more information.  

# CMU SEI Baseline Submission 

## About 

Our submission uses a combination of a localization polygon detection model and a classification model. 

## Environment Setup

During development we used `Python`'s `miniconda` This is recommended as it can help control dependencies and the version of those dependencies are imported easily without breaking your existing repositories. You can also use whichever other virtual environment you want. You can get this through the `Python` package manager [pip](https://pip.pypa.io/en/stable/installing/). 

Our **minimum Python version is 3.6+**, you can get it from [here](https://www.python.org/downloads/release/python-369/). 

 Before you get started be sure to have `Python`, `pip` (through conda), `conda` installed and running on Python 3.6+.
   * We highly recommend using `conda` to manage your packages and versions. Initialize this by running `conda create -n $ENV_NAME` within your xView2 root directory. 
   * This will create an anaconda virtual environment ready to install dependencies. 
   * Finally install pip like `conda activate $ENV_NAME && conda install pip`

Read more about `miniconda` [here](https://docs.conda.io/en/latest/miniconda.html).

Once in your own virtual environment you can install the packages required to train and run the baseline model. 

Check you are using `conda`'s `pip` and `python` by doing `$ which pip` it should report something like: `/path/to/miniconda/envs/$ENV_NAME/bin/pip`, if not you may need to run `conda activate` or check your `.bashrc`. 

Before installing all dependencies run `$ pip install numpy tensorflow` for CPU-based machines or `$ pip install numpy tensorflow-gpu && conda install cupy` for GPU-based (CUDA) machines, as they are install-time dependencies for some other packages.

Finally, use the provided [requirements.txt](./requirements.txt) file for the remainder of the Python dependencies like so, `$ pip install -r requirements.txt` (make sure you are in the same environment as before (`conda activate $ENV_NAME`), or else the packages will be scattered and not easily found).

## Data Downloads

The data during the challenge is available from [xView2](https://xview2.org) challenge website, please register or login to download the data! 

The current total space needed is: about **10GB** compressed and about **11GB** uncompressed. 

### Other Methods

If you are using our pipeline the best way to format data is like so (the pipeline scripts `mask_polygons.py`, and `data_finalize.sh` provided in the repository expect this structure): 

```
xBD 
 ├── disaster_name_1
 │      ├── images 
 │      │      └── <image_id>.png
 │      │      └── ...
 │      ├── labels
 │      │      └── <image_id>.json
 │      │      └── ...
 ├── disaster_name_2
 │      ├── images 
 │      │      └── <image_id>.png
 │      │      └── ...
 │      ├── labels
 │      │      └── <image_id>.json
 │      │      └── ...
 └── disaster_name_n
```

You can do this with [`split_into_disasters.py`](./utils/split_into_disasters.py). This will take the location of your `train` directory and ask for an output directory to place each disaster with the subfolders `image/` and `label/`. 

Example call: 

`$ python split_into_disasters.py --input ~/Downloads/train/ --output ~/Downloads/xBD/`


## Baseline 
 
We are using a fork of motokimura's [SpaceNet Building Detection](https://github.com/motokimura/spacenet_building_detection) for our localization in order to automatically pull our polygons to feed into our classifier. 

We have provided several resources for formatting the dataset for this submission in the [utils](./utils) folder. Below are our pipeline steps. 

### Installing Packages and Dependencies 

Before you get started be sure to have followed the steps mentioned in [Environment Setup](https://github.com/DIUx-xView/xView2#Environment-Setup) and have `conda activated $ENV_NAME` to have your dependencies loaded into your environment. 

###  Training

#### Localization Training 

Below we will walk through the steps we have used for the localization training. 

#### Localization Training Pipeline

These are the pipeline steps are below for the instance segmentation training (these programs have been written and tested on Unix systems (Darwin, Fedora, Debian) only).

First, we must create masks for the localization, and have the data in specific folders for the model to find and train itself. The steps we have built are described below:

1. Run `mask_polygons.py` to generate a mask file for the chipped images.
   * Sample call: `python mask_polygons.py --input /path/to/xBD --single-file --border 2`
   * Here border refers to shrinking polygons by X number of pixels. This is to help the model separate buildings when there are a lot of "overlapping" or closely placed polygons
   * Run `python mask_polygons.py --help` for the full description of the options. 
2. Run `data_finalize.sh` to setup the image and labels directory hierarchy that the `spacenet` model expects (will also run a python `compute_mean.py` to create a mean image our model uses during training. 
   * sample call: `data_finalize.sh -i /path/to/xBD/ -x /path/to/xView2/repo/root/dir/ -s .75`
   * -s is a crude train/val split, the decimal you give will be the amount of the total data to assign to training, the rest to validation
     * You can find this later in /path/to/xBD/spacenet_gt/dataSplit in text files, and easily change them after the script has been run. 
   * Run `data_finalize.sh` for the full description of the options. 
 
After these steps have been ran you will be ready for the instance segmentation training. 

The directory structure will look like: 

```
/path/to/xBD/
├── guatemala-volcano
│   ├── images
│   ├── labels
│   └── masks
├── hurricane-florence
│   ├── images
│   ├── labels
│   └── masks
├── hurricane-harvey
│   ├── images
│   ├── labels
│   └── masks
├── hurricane-matthew
│   ├── images
│   ├── labels
│   └── masks
├── hurricane-michael
│   ├── images
│   ├── labels
│   └── masks
├── mexico-earthquake
│   ├── images
│   ├── labels
│   └── masks
├── midwest-flooding
│   ├── images
│   ├── labels
│   └── masks
├── nepal-flooding
│   ├── images
│   ├── labels
│   └── masks
├── palu-tsunami
│   ├── images
│   ├── labels
│   └── masks
├── santa-rosa-wildfire
│   ├── images
│   ├── labels
│   └── masks
├── socal-fire
│   ├── images
│   ├── labels
│   └── masks
└── spacenet_gt
    ├── dataSet
    ├── images
    └── labels
```

The original images and labels are preserved in the `./xBD/org/$DISASTER/` directories, and just copies the images to the `spacenet_gt` directory.  

#### Training the SpaceNet Model

Now you will be ready to start training a model (based off our provided [weights](https://github.com/DIUx-xView/xview2-baseline/releases/tag/v1.0), or from a baseline).

Using the `spacenet` model we forked, you can control all of the options via command line calls.

In order for the model to find all of its required files, you will need to `$ cd /path/to/xView2/spacenet/src/models/` before running the training module. 

The main file is [`train_model.py`](./spacenet/src/models/train_model.py) and the options are below: 

```
usage: train_model.py [-h] [--batchsize BATCHSIZE]
                      [--test-batchsize TEST_BATCHSIZE] [--epoch EPOCH]
                      [--frequency FREQUENCY] [--gpu GPU] [--out OUT]
                      [--resume RESUME] [--noplot] [--tcrop TCROP]
                      [--vcrop VCROP]
                      dataset images labels

positional arguments:
  dataset               Path to directory containing train.txt, val.txt, and
                        mean.npy
  images                Root directory of input images
  labels                Root directory of label images

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE, -b BATCHSIZE
                        Number of images in each mini-batch
  --test-batchsize TEST_BATCHSIZE, -B TEST_BATCHSIZE
                        Number of images in each test mini-batch
  --epoch EPOCH, -e EPOCH
                        Number of sweeps over the dataset to train
  --frequency FREQUENCY, -f FREQUENCY
                        Frequency of taking a snapshot
  --gpu GPU, -g GPU     GPU ID (negative value indicates CPU)
  --out OUT, -o OUT     Directory to output the result under "models"
                        directory
  --resume RESUME, -r RESUME
                        Resume the training from snapshot
  --noplot              Disable PlotReport extension
  --tcrop TCROP         Crop size for train-set images
  --vcrop VCROP         Crop size for validation-set images
```

A sample call we used is below: 

`$ python train_model.py /path/to/xBD/spacenet_gt/dataSet/ /path/to/xBD/spacenet_gt/images/ /path/to/xBD/spacenet_gt/labels/ -e 100`

**WARNING**: You must be in the `./spacenet/src/models/` directory to run the model due to relative paths for `spacenet` packages, as they have been edited for this instance and would be difficult to package. 

#### Damage Classification Training

**WARNING** If you have just ran the (or your own) localization model, be sure to clean up any localization specific directories (e.g. `./spacenet`) before running the classification pipeline. This will interfere with the damage classification training calls as they only expect the original data to exist in directories separated by disaster name. You can use the [`split_into_disasters.py`](./utils/split_into_disasters.py) program if you have a directory of ./images and ./labels that need to be separated into disasters.  

The damage classification training processing and training code can be found under `/path/to/xView2/model/` 

You will need to run the `process_data.py` python script to extract the polygon images used for training, testing, and holdout from the original satellite images and the polygon labels produced by SpaceNet. This will generate a csv file with polygon UUID and damage type
as well as extracting the actual polygons from the original satellite images. If the `val_split_pct` is defined, then you will get two csv files, one for test and one for train. 

**Note** The process_data script only extracts polygons from post disaster images

```
usage: python process_data [-h] --input_dir --output_dir --output_dir_csv [--val_split_pct]

arguments:

-h                                     show help message and exit
--input_dir                            path to xBD data
--output_dir                           existing path to an output directory
--output_dir_csv                       path to where you want train.csv and test.csv to be saved
--val_split_pct                        a decimal number between 0.0 and 1.0 for the split percentage in validation set

```

**Note** Please update the LOG_DIR to where you want your tensorboard logs to go before running damage_classification.py
If you want to update details like the BATCHSIZE, NUM_WORKERS, EPOCHS, or LEARNING_RATE you will need to open damage_classification.py and edit the configuration variables at the top of the script. 

```
usage: python damage_classification.py [-h] --train_data TRAIN_DATA_PATH
                                       --train_csv TRAIN_CSV
                                       --test_data TEST_DATA_PATH
                                       --test_csv TEST_CSV
                                       [--model_in MODEL_WEIGHTS_PATH]
                                       --model_out MODEL_WEIGHTS_OUT

arguments:

 -h                                    show help message and exit
 --train_data TRAIN_DATA_PATH          Path to training polygons
 --train_csv TRAIN_CSV                 Path to train csv 
 --test_data TEST_DATA_PATH            Path to test polygons
 --test_csv TEST_CSV                   Path to test csv
 --model_in MODEL_WEIGHTS_PATH         Path to any input weights (optional)
 --model_out MODEL_WEIGHTS_OUT         Path to output weights (do not add file extention)
```
 
Sample command: `$ python damage_classification.py --train_data /path/to/XBD/$process_data_output_dir/train 
--train_csv train.csv --test_data /path/to/XBD/$process_data_output_dir/test --test_csv test.csv --model_out path/to/xBD/baseline_trial --model_in /path/to/saved-model-01.hdf5`

### Inference 

Since the inference is done over two models we have created a script [inference.sh](./utils/inference.sh) to script together the inference code. 

If you would like to run the inference steps individually the shell script will provide you with those steps. 

To run the inference code you will need: 

1. The `xView2` repository (this one, where you are reading this README) cloned
2. An input pre image
3. An input post image that matches the pre in the same directory
4. Weights for the localization model
5. Weights for the classification model 

You can find the weights we have trained in the [releases section](https://github.com/DIUx-xView/xview2-baseline/releases/tag/v1.0) of this Github repository.

As long as we can find the post image by replacing pre with post (`s/pre/post/g`) everything else should be run, this is used to dockerize the inference and run in parallel for each image individually based off the submission requirements of the challenge. 

Sample Call: `./utils/inference.sh -x /path/to/xView2/ -i /path/to/$DISASTER_$IMAGEID_pre_disaster.png -p  /path/to/$DISASTER_$IMAGEID_post_disaster.png -l /path/to/localization_weights.h5 -c /path/to/classification_weights.hdf5 -o /path/to/output/image.png -y`

The inference script writes log files to `/tmp/inference_log` and the intermediately steps to `/tmp/inference`, normally `/tmp/inference` is removed after each run, but can be kept (and you can see the final json for visual checking in `/tmp/inferences/inference.json`) by commenting out [this line](./utils/inference.sh#L172) in `./utils/inference.sh`.

### Docker container 

The submission [Dockerfile](./submission/Dockerfile) wraps the above inference script and requires the user to mount a folder with the pre/post image pair, and a separate folder for the inference output image. 

To simplify the creation of the docker follow these steps: 
1. Ensure Docker is installed and running (with the ability to have ~8GB of RAM)
2. move into the `xview2-baseline/submission/` folder
3. Build by calling `$ docker build -t cmu-xview2-baseline .` 

This will build the image (based off Ubuntu Docker baseline) and add in the repository and published weight files from the public git. 

**WARNING**: The docker image downloads some files (CMU xView2 baseline code, weights, ResNet etc.) be sure to be off a VPN or give the Docker daemon access to the local network environment for network requests. 

Running the image will need a directory for the files, and an output directory to write to. Below is a sample call to run the inference docker image for submission (this will output two of the same output files for scoring (scoring is done at the localization and classification stages, but we output the file at the end and would use the same polygons found in the localization stage anyways). 


`$ docker run -v /local/path/to/xBD/folder/with/images/:/submission/ -v /local/path/to/output/scoring/files/:/output/ cmu-xview2-baseline /submission/path/to/pre_image.png /submission/path/to/post_image.png /output/localization_output_name.png /output/classification_output_name.png`

A more specific example would be: 

`$ docker run -v ~/Downloads/xBD/:/submission/ -v ~/Downloads/xBD/:/output/ cmu-xview2-baseline /submission/guatemala-volcano/images/guatemala-volcano_00000023_pre_disaster.png /submission/guatemala-volcano/images/guatemala-volcano_00000023_post_disaster.png /output/test/loc.png /output/test/dmg.png`

### Output 

The output will be two identical grey scale PNG with the following pixel values: 

```
0 for no building 
1 for building found and classified no-damaged 
2 for building found and classified minor-damage 
3 for building found and classified major-damage
4 for building found and classified destroyed
```

See the code we use to translate the json outputs to this submission image [inference_image_output.py](./utils/inference_image_output.py), also see the submission rules on the [xview2 website](https://xview2.org/challenge).

## Contact 

See the [xView2 FAQ](https://xview2.org/challenge) page first, if you would like to talk to us further reach out to the xView2 chat on [discord](https://xview2.org/chat).

