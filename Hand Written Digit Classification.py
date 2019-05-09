import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Activation,Flatten,Dense,Conv2D,MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import time
import os


def predict_images(network,model):
	img_arr = cv2.imread("test.png",cv2.IMREAD_GRAYSCALE)
	img_arr = img_arr/255.0
	if(network == 1):
		reshaped_img = img_arr.reshape(-1,28,28)
	elif(network == 2):
		reshaped_img = img_arr.reshape(-1,28,28,1)
	predictions = model.predict([reshaped_img])
	draw(img_arr,predictions)

def build_model(hidden_layers,Neurons):
	m = Sequential()
	m.add(Flatten())
	for i in range(hidden_layers):
		m.add(Dense(Neurons[i],activation = 'relu'))
	m.add(Dense(10, activation = 'softmax'))
	m.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
	return m

def choose_arch():
	os.system('cls')
	print("Choose your Network Architecture :\n\n\tConv -- Convolutional layer\tpool -- MaxPool layer\tFC -- FullyConnected layer")
	print("\n1) Conv(128,3x3) --> pool(2,2) --> Flatten --> FC(128) --> Output(10)")
	print("\n2) Conv(128,3x3) --> pool(2,2) --> Conv(64,3x3) --> pool(2,2) --> Flatten --> FC(128) --> Output(10)")
	print("\n3) Conv(128,5x5) --> pool(2,2) --> Conv(256,5x5) --> Conv(512,5x5) --> pool(2,2) --> Flatten --> FC(256) --> Output(10)")
	return int(input("\n\nEnter your choice : "))

def build_cnn_model():
	c = choose_arch()
	if(c==1):
		m = Sequential()
		m.add(Conv2D(128,(3,3),activation='relu'))
		m.add(MaxPooling2D(2,2))
		m.add(Flatten())
		m.add(Activation('relu'))
		m.add(Dense(128,activation='relu'))
		m.add(Dense(10,activation = 'softmax'))
	elif(c==2):
		m = Sequential()
		m.add(Conv2D(128,(3,3),activation='relu'))
		m.add(MaxPooling2D(2,2))
		m.add(Conv2D(64,(3,3),activation='relu'))
		m.add(MaxPooling2D(2,2))
		m.add(Flatten())
		m.add(Activation('relu'))
		m.add(Dense(128,activation='relu'))
		m.add(Dense(10,activation = 'softmax'))
	elif(c==3):
		m = Sequential()
		m.add(Conv2D(128,(5,5),activation='relu'))
		m.add(MaxPooling2D(2,2))
		m.add(Conv2D(256,(5,5),activation='relu'))
		m.add(Conv2D(512,(5,5),activation='relu'))
		m.add(MaxPooling2D(2,2))
		m.add(Flatten())
		m.add(Activation('relu'))
		m.add(Dense(128,activation='relu'))
		m.add(Dense(10,activation = 'softmax'))
	else:
		print("Invalid Choice, Rolling back in a moment")
		build_cnn_model()

	m.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
	return m


def train(choice,h_layers=None,neurons=None,b_size=None,eps=None):
	if(choice == 1):
		model = build_model(h_layers,neurons)
		print("\nLoading dataset...")
		(X_train,y_train),(X_test,y_test) = mnist.load_data()
		print("\nLoading dataset Successful...")
		print("\nNormalizing data...")
		X_train = X_train/255.0
		X_test = X_test/255.0
		print("\nNormalizing data Successful...")
		input("\nPress any key to start training your model...")
		os.system('cls')
		print("Training your model with : \nHidden Layers : ",h_layers,"\nNeurons : ,",neurons,"\nBatch Size : ",b_size,"\nEpochs : ",eps,"\n")
		model.fit(X_train,y_train,batch_size = b_size, epochs = eps, validation_data = (X_test,y_test))
		time.sleep(3)
		os.system('cls')
		print("Model training Successful,Your Model Details")
		print("\nModel trained on ",len(X_train)," Samples and tested on ",len(X_test)," Samples");
		print("\nModel Testing Accuracy : ", model.evaluate(X_test,y_test)[1])
		print("\nModel Summary :")
		print(model.summary())
		input("\nPress any key to continue...(Your trained model will be saved automatically...)")
		print("\nSaving your model...")
		model.save('DIGIT.model')
	else:
		model = build_cnn_model()
		print("\nLoading dataset...")
		(X_train,y_train),(X_test,y_test) = mnist.load_data()
		print("\nLoading dataset Successful...")
		print("\nNormalizing data...")
		X_train = X_train/255.0
		X_test = X_test/255.0
		print("\nNormalizing data Successful...")
		print("\nConverting the input Image into Volume...")
		X_train = X_train.reshape(-1,28,28,1)
		X_test = X_test.reshape(-1,28,28,1)
		print("\nConvertion Successful...")
		input("\nPress any key to start training your model...")
		os.system('cls')
		model.fit(X_train,y_train,batch_size = b_size, epochs = eps, validation_data = (X_test,y_test))
		time.sleep(3)
		os.system('cls')
		print("Model training Successful,Your Model Details")
		print("\nModel trained on ",len(X_train)," Samples and tested on ",len(X_test)," Samples");
		print("\nModel Testing Accuracy : ", model.evaluate(X_test,y_test)[1])
		print("\nModel Summary :")
		print(model.summary())
		input("\nPress any key to continue...(Your trained model will be saved automatically...)")
		print("\nSaving your model...")
		model.save('DIGIT_CNN.model')
		
def main():
	os.system('cls')
	use_network = int(input("Select your model :\n\t1.Deep Neural Network\n\t2.Convolutional Neural Network\n\nplease choose your desired model : "))
	if(use_network == 1):
		if(os.path.isfile('DIGIT.model')):
			model = keras.models.load_model('DIGIT.model')
			predict_images(use_network,model)
		else:
			try :
				print("\nIt seems you didn't trained your model yet...\nEntering model training phase...")
				hidden_layers = int(input("\nEnter No. of hidden_layers in your Network : "))
				Neurons = []
				for i in range(hidden_layers):
					print("\nEnter No of Neurons in Hidden Layer ",(i+1)," : ",end="")
					Neurons.append(int(input()))
				e = int(input("\nEnter No. of epochs (Best 100 Unless you have a large dataset) : "))
				b = int(input("\nEnter the batch_size (Best 32 Unless you have a large dataset) : "))
				train(1,h_layers = hidden_layers, neurons = Neurons,b_size = b, eps = e)
			except :
				print("Something went worng. Enter data correctly...\nExiting...")
				exit()
	elif(use_network == 2):
		if(os.path.isfile('DIGIT_CNN.model')):
			model = keras.models.load_model('DIGIT_CNN.model')
			predict_images(use_network,model)
		else:
			try :
				print("\nIt seems you didn't trained your model yet...\nEntering model training phase...")
				e = int(input("\nEnter No. of epochs (Best 100 Unless you have a large dataset) : "))
				b = int(input("\nEnter the batch_size (Best 32 Unless you have a large dataset) : "))
				train(2,b_size = b, eps = e)
			except :
				print("Something went worng. Enter data correctly...\nExiting...")
				exit()
	else:
		print("Invalid Input, Rolling back in a moment")
		time.sleep(2)
		main()


def draw(img,prediction):
	ax1 = plt.subplot2grid((4,4),(0,0),rowspan = 1, colspan = 4)
	ax2 = plt.subplot2grid((4,4),(1,0),rowspan = 3, colspan = 4)
	ax1.imshow(img,cmap = plt.cm.binary)
	ax2.bar(range(10),prediction[0])
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax2.set_xticks(range(10))
	ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
	plt.show()

if __name__ == "__main__":
	main()