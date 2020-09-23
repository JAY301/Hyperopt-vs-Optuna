# A Study to compare two HPO libraries : Optuna and Hyperopt using two well-known datasets 1. CFIAR10 and 2. MNIST

## Methodology followed during this project

After shortlisting two widespread HPO libraries, Optuna and Hyperopt, the performance of these two libraries was validated. For the validation of these libraries, two small tasks were created: 1. Digit recognition task and 2. Image classification task. The motive behind performing these two tasks was to compare the performance of these two HPO libraries and to select the best performing HPO library.

Two well-known open-source datasets were piloted: 1. MNIST Dataset and 2. CIFAR10 Dataset.

These two datasets contain images of very small size (28x28x1 pixels in MNIST and 32x32x3 pixels in CIFAR10).

The purpose of these pilot studies was not limited to validate the performance of HPO libraries, but also to compare important hyperparameters.

To perform this study, Anaconda was used. Anaconda is a free and open-source distribution of the Python programming language for scientific computing (data science, machine learning applications, deep learning applications etc.) that aims to simplify package management and deployment. The distribution includes deep learning packages suitable for Windows. 

There were a total of four scripts developed for each task (Digit recognition task and Image classification task). Out of these four scripts, three scripts were for the HPO using Optuna and one script was for the HPO using Hyperopt. 

1. MNIST Optuna CMAES MultiHP.ipynb
2. MNIST Optuna TPE MultiHP.ipynb
3. MNIST Optuna Mix MultiHP.ipynb
4. MNIST Hyperopt MultiHP.py

1. CIFAR10 Optuna CMAES MultiHP.ipynb
2. CIFAR10 Optuna TPE MultiHP.ipynb
3. CIFAR10 Optuna Mix MultiHP.ipynb
4. CIFAR10 Hyperopt MultiHP.py

Optuna has 3 different hyperparameter optimization algorithms integrated into it: 1. TPE 2. CMA-ES and 3. A mixture of TPE and CMA-ES. Hyperopt has only one hyperparameter optimization algorithm integrated into it which is TPE. 

For this study, one development environment was created in Anaconda, where the most important ML library Tensorflow-Keras was installed. Along with Tensorflow-Keras, some other useful python libraries were installed for data pre-processing and visualization 

The study was carried out in a remote PC equipped with NVIDIA GPU GeForce RTX 1080Ti.

Citations:

1. Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2623–2631, 2019.

2. James Bergstra, Brent Komer, Chris Eliasmith, Dan Yamins, and David D Cox. Hyperopt: a python library for model selection and hyperparameter optimization. Computational Science & Discovery, 8(1):014008, 2015.
