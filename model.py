#######################################################
# Udacity Self Driving Car  Nanodegree
#
# Project #3 Behavioral-Cloning
# by Ghicheon Lee
#######################################################

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.callbacks  import EarlyStopping,ModelCheckpoint
from sklearn.utils    import shuffle

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import csv
import sys

import pickle

list=[]    # it has all entries of cvs files.

#number of epochs
N_EPOCHS = 20

# the number of entries of cvs files.
REAL_ENTRY_SIZE = 16

# the number of data. 1 cvs entry makes 6 data.
BATCH_SIZE = REAL_ENTRY_SIZE*6

#loading cvs files: 
for i in [1]:
	fname = './data_track' + str(i) + '/driving_log.csv'
	with open( fname ) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
                     list.append(line)
for i in [1,2,3]:
	fname = './my_data' + str(i) + '/driving_log.csv'
	with open( fname ) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
                     list.append(line)


#mix entries
shuffle(list)

#the number of entries
LEN = len(list)

# train set : validation set : test set   = 7:2:1
#round up to REAL_ENTRY_SIZE
slice1 = (int(LEN*0.7)   + (REAL_ENTRY_SIZE -1 ) )  &  ~(REAL_ENTRY_SIZE -1 )
slice2 = (int(LEN*0.9)   + (REAL_ENTRY_SIZE -1 ) )  &  ~(REAL_ENTRY_SIZE -1 )
data4train = list[0     : slice1]
data4valid = list[slice1: slice2]
data4test =  list[slice2: LEN   ] 

print("----------------------------")
print("data4train size(cvs entries):", len(data4train))
print("data4valid size(cvs entries):", len(data4valid))
print("data4test size(cvs entries):", len(data4test))
print("----------------------------")

#management value needs to be added a little bit for left camera image.
#management value needs to be reduced a little for right camera image.
correction_factor=[0 ,0.20 ,-0.20 ]

# batch_size means the number of data for training per batch.
def generator_core(sample, batch_size=96):

    #real batch_size for cvs entries is batch_size times 3*2.
    #3 for center/left/right images
    #2 for  a flip
    batch_size //= (3*2)

    while 1:
            #shuffle for each epoch 
            shuffle(sample)

            #print("")
            #print("--------- shuffled sample(", len(sample),") -----------")
            #print("")
            for start in range(0, len(sample), batch_size):
                    images=[]
                    measurements=[]
                    for i in range(start, start + batch_size):
                            if i >= len(sample): # out of range
                                    break;

                            line = sample[i]

                            #1 entry produces 6 data.
                            for camera in range(3):
                                image = plt.imread(line[camera])
                                steer = float(line[3]) + correction_factor[camera]

                                #append it to result sets
                                images.append(image)
                                measurements.append(steer)

                                #append flipped image to result sets
                                images.append(np.fliplr(image))  # == np.flip(image,1)
                                measurements.append( -steer)

                    images = np.array(images)
                    measurements = np.array(measurements)

                    
                    #print("[DEBUG] images size:",len(images))
                    yield sklearn.utils.shuffle(images,measurements)


# creating a model
model=Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(12,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(24,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(48,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(96,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Dropout(0.5))
model.add(Dense(1))
model.summary()

model.compile(loss='mse',optimizer='adam')

#early stopping for validation loss. I set patience to 5 because sometimes, the loss decreses a little bit later...
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

#save only best model so far.
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')



#the REAL number of training data
n_total_train = len(data4train)*6

#the REAL number of validation data
n_total_valid = len(data4valid)*6


#print("training start........")
history_object = model.fit_generator(
                    generator = generator_core(data4train,BATCH_SIZE), 
                    steps_per_epoch = n_total_train // BATCH_SIZE,
                    validation_data = generator_core(data4valid,BATCH_SIZE), 
                    validation_steps = n_total_valid // BATCH_SIZE,
                    epochs=N_EPOCHS,callbacks=[early, checkpoint] )


# Test loss & accuracy
images=[]
measurements=[]
for line in data4test:
    #1 entry produces 6 data.
    for camera in range(3):
        image = plt.imread(line[camera])
        steer = float(line[3]) + correction_factor[camera]

        #append it to result sets
        images.append(image)
        measurements.append(steer)

        #append flipped image to result sets
        images.append(np.fliplr(image))  # == np.flip(image,1)
        measurements.append( -steer)

images = np.array(images)
measurements = np.array(measurements)

print("-------------- Test loss ----------------")
score = model.evaluate(images , measurements, verbose=0)
print("Test loss:", score)
print("-----------------------------------------")



## plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


#surpress some error messages in the end.
import gc
gc.collect()


sys.exit(1)
