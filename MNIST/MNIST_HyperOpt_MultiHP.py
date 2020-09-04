import os
os.environ['PYTHONHASHSEED']=str(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(0)
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.utils import to_categorical
import csv
MNIST_image_path = 'MNIST_data'
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform
import sys

tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_logical_devices('GPU')))



def load_MNIST_data(MNIST_image_path, filename):
    
	features = []
	labels = []
    
	with open(MNIST_image_path + '/' + filename + '.csv') as file:
		file_csv = csv.reader(file)
        
		next(file_csv, None)
		for row in file_csv:
        
			labels.append(int(row[0]))
			features.append(row[1:785])
            
	features = np.array(features)
	labels = np.array(labels)
	features = features.astype(np.float)
	print('Data size is {}'.format(features.shape))
	features = features.reshape(len(features), 28, 28)
    
	return features, labels

def load_MNIST_test_data(MNIST_image_path, filename):
    
	features = []
	labels = []
    
	with open(MNIST_image_path + '/' + filename + '.csv') as file:
		file_csv = csv.reader(file)
        
		next(file_csv, None)
		for row in file_csv:
        
			labels.append(int(row[0]))
			features.append(row[1:785])
            
	features = np.array(features)
	labels = np.array(labels)
	features = features.astype(np.float)
	print('Data size is {}'.format(features.shape))
	features = features.reshape(len(features), 28, 28)
    
	return features, labels



def model(x_train, y_train, x_test, y_test):
	
	from hyperopt import Trials, STATUS_OK, tpe
	from hyperas import optim

	from hyperas.distributions import choice, uniform, loguniform
	import tensorflow as tf
	from keras.optimizers import Adam, SGD
	from keras.preprocessing.image import ImageDataGenerator
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
	from keras.applications import ResNet50
	import sys
	from keras import backend as K 
		
	
	model = Sequential()
	
	model.add(
		Conv2D(
			filters= {{choice([32, 64])}},
			kernel_size = (3,3),
			padding = 'same',
			activation= 'relu',
			input_shape=(28,28,1),
			)
		)
	

	if {{choice(['two', 'three', 'four'])}} == 'two':
		model.add(
		Conv2D(
			filters= {{choice([32, 64])}},
			kernel_size= (3,3),
			padding = 'same',
			activation= 'relu',
			input_shape=(28,28,1),
			)
		)
		if {{choice(['two', 'three', 'four'])}} == 'three':
			model.add(
			Conv2D(
				filters= {{choice([64, 128])}},
				kernel_size= (3,3),
				padding = 'same',
				activation= 'relu',
				input_shape=(28,28,1),
				)
			)
			if {{choice(['two', 'three', 'four'])}} == 'four':
				model.add(
				Conv2D(
					filters= {{choice([64, 128])}},
					kernel_size= (3,3),
					padding = 'same',
					activation= 'relu',
					input_shape=(28,28,1),
					)
				)
		
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	
	if {{choice(['two', 'three', 'four'])}} == 'two':
		num_hidden2 = {{choice([128, 256, 512, 1024, 2048, 4096])}}
		model.add(Dense(num_hidden2, activation="relu"))
		dropout2 = {{uniform(0.1, 0.3)}}
		model.add(Dropout(rate=dropout2))
		if {{choice(['two', 'three', 'four'])}} == 'three':
			num_hidden3 = {{choice([128, 256, 512, 1024, 2048, 4096])}}
			model.add(Dense(num_hidden3, activation="relu"))
			dropout3 = {{uniform(0.1, 0.3)}}
			model.add(Dropout(rate=dropout3))
			if {{choice(['two', 'three', 'four'])}} == 'four':
				num_hidden4 = {{choice([128, 256, 512, 1024, 2048, 4096])}}
				model.add(Dense(num_hidden4, activation="relu"))
				dropout4 = {{uniform(0.1, 0.3)}}
				model.add(Dropout(rate=dropout4))

	model.add(Dense(10, activation="softmax"))

	LR = {{uniform(0.00001, 0.01)}}

	opt = SGD(lr = LR)
	
	model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	
	model.fit(x_train, y_train, epochs=10, verbose=2, validation_split=0.1, shuffle = False)
	
	loss, acc = model.evaluate(x_test, y_test, verbose=2)
	
	print ('Test loss: {}, Test accuracy: {}'.format(loss, acc))
	sys.stdout.flush()
	del model
	K.clear_session()
	return {'loss': -acc, 'status': STATUS_OK}



def data():
	from hyperopt import Trials, STATUS_OK, tpe
	from hyperas import optim
	from hyperas.distributions import choice, uniform, loguniform
	from keras.utils import np_utils
	from MNIST_HyperOpt import load_MNIST_data
	from MNIST_HyperOpt import load_MNIST_test_data
	MNIST_image_path = 'MNIST_data'
	nb_classes = 10
	x_train, y_train = load_MNIST_data(MNIST_image_path, 'train')
	x_test, y_test = load_MNIST_test_data(MNIST_image_path, 'test')
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_train = x_train/255.0
    
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	x_test = x_test/255.0
	#X_train = X_train.reshape(60000, 784)
	#X_test = X_test.reshape(10000, 784)
	#x_train = x_train.astype('float32')
	#x_test = x_test.astype('float32')
	#x_train /= 255.0
	#x_test /= 255.0
	#y_train = np_utils.to_categorical(y_train, nb_classes)
	#y_test = np_utils.to_categorical(y_test, nb_classes)
	return x_train, y_train, x_test, y_test




if __name__ == '__main__':
	

	#X_train, Y_train = load_cifar10_data(cifar_image_path, 5)
	#X_test, Y_test = load_cifar10_test_data(cifar_image_path)
	from hyperopt import Trials
	import csv
	
	trials = Trials()
	best_run, best_model = optim.minimize(model=model, data = data, algo=tpe.suggest, max_evals=50, trials=trials)
	X_train, Y_train, X_test, Y_test = data()
	
	
	Acc = []
	for item in trials.trials:
		Acc.append(-1*item['result']['loss'])
	print(Acc)
	
	with open("HyperOpt_result_dict.csv", "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["Results"])
		for i in item['misc']['vals']:
			writer.writerow([i])
		for item in trials.trials:
			for i in item['misc']['vals']:
				writer.writerow([item['misc']['vals'][i][0]])	
		writer.writerow(["Acc"])
		for i in Acc:
			writer.writerow([i])
	
		
	
	#print("Evalutation of best performing model:")
	#print(best_model.evaluate(X_test, Y_test))
	#print("Best performing model chosen hyper-parameters:")
	#print(best_run)
	
