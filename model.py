import csv
from sklearn.model_selection import train_test_split
import sklearn
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read data from driving_log.csv
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split the data into train and validation sets
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)


def generator_data(samples, batch_size):
	# Generator function to preprocess data and feed to the model in batches.
    num_samples = len(samples)
    while True:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            corr = 0.2
            for line in batch_samples:
                for i in range(3):
                    current_path = './data/' + line[i].lstrip()
                    image = cv2.imread(current_path)
                    images.append(image)
                    if i==1:
                        measurement = float(line[3]) + corr
                    elif i==2:
                        measurement = float(line[3]) - corr
                    else:
                        measurement = float(line[3])
                    measurements.append(measurement)
                    images.append(cv2.flip(image,1))
                    measurements.append(measurement*-1.0)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Nvidia model
model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

train_generator = generator_data(train_samples, batch_size=32)
valid_generator = generator_data(validation_samples, batch_size=32)

# Compile and run the model for 3 epochs
model.compile(loss='mse', optimizer='adam')
history_object=model.fit_generator(generator=train_generator,samples_per_epoch=len(train_samples), validation_data=valid_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')
