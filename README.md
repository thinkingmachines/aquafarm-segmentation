<div align="center">

# 🦐🐟 Aquafarm Mapping Segmentation

</div>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

<br/>
<br/>


# 📜 Description

This repo contains the neural network model for classifying pond aquaculture into 3 types: intensive, extensive, and abandoned.

This project is developed under a [CCAI 2022 grant for scaling out Climate Smart Shrimp in Indonesia and the Philippines](https://www.climatechange.ai/blog/2022-06-16-grants-mangrove).
<br/>
<br/>

# 💻 Usage
## Data Access and Download
The datasets and model are available [here](https://drive.google.com/drive/folders/1VRnK7vPCWMTSRQOwuDie18zES3EqA27S?usp=share_link)

We've uploaded the model, the training data, and rollout data.

- `model.zip` contains the Pytorch Lightning model checkpoint of the trained model
- `training_data.zip` contains TIFF files of the training images and raster masks.
    - It also contains a geopackage (`training_polygons.gpkg`) of the training polygons that were rasterized into masks
    - A lookup table `cloud_cover.csv` is provided to filter out annotated images in the training set that are covered in clouds based on NICFI satellite imagery.
- `rollout_data.zip` contains TIFF files of images across Indonesia and the Philippines for aquaculture segmentation.
    - It also contains a geopackage (`pred_polygons.gpkg`) of the predicted aquaculture polygons.
<br/>
<br/>

## Training the model
1. Download and unzip the model and training data
    - Place in the appropriate directories. For contents of `model.zip` in `model`, and the contents of `training_data` in `data`.
2. Set up the config file `config/pond_config.yaml`
3. Run the notebooks in the `notebooks` directory in order. These notebooks train a Pytorch Lightning semantic segmentation model based on the data and config.
    - Inspect the data using the `01` notebooks.
    - The `02` notebooks are more optional and are meant to assist in determining the learning rate and batch size. There's also a overfitting a batch notebook for model troubleshooting.
    - Train the model itself using the `03` notebook.
    - Test model inference on the training data with the `04` notebook.
    - Visualize the model predictions with the `05` notebook.
        - Note: If you'll use the NICFI Basemap to visualize, you'll need to sign up for a [NICFI account and get an API key](https://www.planet.com/nicfi/)

<br/>
<br/>

## Rolling out the model
1. Download and unzip the model and training data
    - Place in the appropriate directories. For contents of `model.zip` in `model`, and the contents of `rollout_data` in `data`.
2. Set up the config file `config/rollout_pond_config.yaml`
3. Run the notebooks in the `notebooks_rollout` directory in order.
    - Inspect the data using the `01` notebook.
    - Perform model inference using the `02` notebook.
    - Visualize the model predictions with the `03` notebook.
        - Note: If you'll use the NICFI Basemap to visualize, you'll need to sign up for a [NICFI account and get an API key](https://www.planet.com/nicfi/)

<br/>
<br/>

# ⚙️ Local Setup for Development

This repo assumes the use of [conda](https://docs.conda.io/en/latest/miniconda.html) for simplicity in installing GDAL.


## Requirements

1. Python 3.9
2. make
3. conda


## 🐍 One-time Set-up
Run this the very first time you are setting-up the project on a machine to set-up a local Python environment for this project.

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your environment if you don't have it yet.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

2. Create a local conda env and activate it. This will create a conda env folder in your project directory.
```
make conda-env
conda activate aquafarm-segmentation
```

3. Run the one-time set-up make command.
```
make setup
```

## 📦 Dependencies

Over the course of development, you will likely introduce new library dependencies. This repo uses [pip-tools](https://github.com/jazzband/pip-tools) to manage the python dependencies.

There are two main files involved:
* `requirements.in` - contains high level requirements; this is what we should edit when adding/removing libraries
* `requirements.txt` - contains exact list of python libraries (including depdenencies of the main libraries) your environment needs to follow to run the repo code; compiled from `requirements.in`


When you add new python libs, please do the ff:

1. Add the library to the `requirements.in` file. You may optionally pin the version if you need a particular version of the library.

2. Run `make requirements` to compile a new version of the `requirements.txt` file and update your python env.

3. Commit both the `requirements.in` and `requirements.txt` files.

Note: When you are the one updating your python env to follow library changes from other devs (reflected through an updated `requirements.txt` file), simply run `pip-sync requirements.txt`
