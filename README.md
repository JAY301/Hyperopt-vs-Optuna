# A Study to compare two HPO libraries : Optuna and Hyperopt using two well-known datasets 1. CFIAR10 and 2. MNIST

## Methodology followed during this project

After shortlisting two widespread HPO libraries, Optuna and Hyperopt, the performance of these two libraries was validated. For the validation of these libraries, two small tasks were created: 1. Digit recognition task and 2. Image classification task. The motive behind performing these two tasks was to compare the performance of these two HPO libraries and to select the best performing HPO library.

Two well-known open-source datasets were piloted: 1. MNIST Dataset and 2. CIFAR10 Dataset.

These two datasets contain images of very small size (28x28x1 pixels in MNIST and 32x32x3 pixels in CIFAR10). Because of their small size, training of ML models became faster and optimization processes could be completed in less time. This way, this study could be done in a shorter time using these two small tasks and results could be achieved early.

The purpose of these pilot studies was not limited to validate the performance of HPO libraries, but also to compare important hyperparameters.

Citations:

1. Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2623–2631, 2019.

2. James Bergstra, Brent Komer, Chris Eliasmith, Dan Yamins, and David D Cox. Hyperopt: a python library for model selection and hyperparameter optimization. Computational Science & Discovery, 8(1):014008, 2015.
