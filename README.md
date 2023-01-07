# datalib
> datalib is a Python library for data analysis and preprocessing. It provides various functions for calculating correlation and mutual information of datasets, discretizing data, performing feature scaling, filtering data, calculating metrics, and plotting various graphs.

## Table of Contents
* [Features](#features)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Usage](#usage)
<!-- * [License](#license) -->

## Features

- Discretize data using the equal width and equal frequency algorithms.
- Calculate correlation and mutual information of datasets.
- Perform feature scaling (both normalization and standardization).
- Filter data based on certain conditions.
- Calculate metrics such as entropy, variance, and AUC.
- Plot ROC curves (and the AUC) , correlation matrices, and entropy.


## Dependencies

- pandas >= 0.25.1
- matplotlib >= 3.1.1
- numpy >=1.17.2

## Installation
If you are working with anaconda you can use the following comands to install datalib:

`conda install git\\
pip install git+https://github.com/Iruzu/datalib.git`

## Usage
To use datalib, run the following command:

`import datalib`

running

`python setup.py pytest`

will execute all tests stored in the ‘tests’ folder.
