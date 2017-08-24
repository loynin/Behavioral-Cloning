
#Import necessary modules for application

import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import color, io
import sklearn
from random import shuffle
import time, datetime
from keras.layers import Dropout
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
#from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D


print("Done importing modules...")

#----Set training hyperameter---------------------------------------------

epoch = 3
batch_size = 100

#-----Load data from recorded folder into reader -------------------------

print("loading data from file into reader")

lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#shuffle data and create 20% of validation data from training data
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#Select one image for sample and display in program-----------------------
row = train_samples[0]
selected_image = cv2.imread(row[0])
print ("Number of Sample data: " , len(lines))
print ("Number of Validation dataset:", len(validation_samples))
print ("Selected Image Shape", selected_image.shape)


#Function use to augment data before feed into model......................

img_size=selected_image.shape

def showimg(img, img1, img2):
    plt.figure(figsize=(6, 2))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Augmented image")
    plt.subplot(1, 3, 2)
    plt.imshow(img1)
    plt.title("Augmented image")
    plt.subplot(1, 3, 3)
    plt.imshow(img2)
    plt.title("Augmented image")
    plt.show()
#Histogram Equalization   
def eq_Hist(img):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img

def scale_img(img):
    img2=img.copy()
    sc_y=0.4*np.random.rand()+1.0
    img2=cv2.resize(img, (img2.shape[0],img2.shape[1]), interpolation = cv2.INTER_CUBIC)
    #c_x,c_y, sh = int(img2.shape[0]), int(img2.shape[1]), int(img_size)
    return img2

def crop(img, mar=0):
    c_x,c_y, sh = int(img.shape[0]/2), int(img.shape[1]/2), int(img_size/2-mar)
    return img[(c_x-sh):(c_x+sh),(c_y-sh):(c_y+sh)]

#random subsample image
def crop_rand(img, mar=0):
    shift = (1-1)*2+1
    a = int(np.random.randint(shift)-shift/2)
    b = int(np.random.randint(shift)-shift/2)
    c_x,c_y, sh = int(img.shape[0]/2), int(img.shape[1]/2), int(img_size/2-mar)
    return img[(c_x-sh+a):(c_x+sh+a),(c_y-sh+b):(c_y+sh+b)]

def rotate_img(img):
    c_x,c_y = int(img.shape[0]/2), int(img.shape[1]/2)
    ang = 30.0*np.random.rand()-15
    Mat = cv2.getRotationMatrix2D((c_x, c_y), ang, 1.0)
    return cv2.warpAffine(img, Mat, img.shape[:2])

def sharpen_img(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

#Compute linear image transformation ing*s+m
def lin_img(img, s=1.0, m=0.0):
    img2=cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))

#Apply random brightness on an image
def bright_img(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    br = 0.3+np.random.uniform()
    if br > 1.0:
        br = 1.0
    img[:,:,2] = img[:,:,2]*br
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img
#Change image contrast; s>1 - increase
def contr_img(img, s=1.0):
    m=127.0*(1.0-s)
    return lin_img(img, s, m)

def transform_img(img, rand=False):
    img2 = eq_Hist(img)
    img2=sharpen_img(img2)
    if rand:
        #img2 = crop_rand(img2,4)
        img2 = bright_img(img2)
        img2 = scale_img(img2)
    else:
        #img2 = crop(img2,4)
        img2 = scale_img(img2)
        #img2 = eq_Hist(img2)
    img2=contr_img(img2, 1.5)
    return img2

def augment_img(img, rand=True):
    img=contr_img(img, 0.9*np.random.rand()+0.1)
    #img=rotate_img(img)
    img=scale_img(img)
    return transform_img(img, rand)


# function use to display an image........................................

def showimage(img, caption = "Image"):
    plt.figure(figsize=(10, 3))
    plt.subplot(1,1,1)
    plt.imshow(img)
    plt.title(caption)
    plt.show()
    
    
# Display sample flip image..............................................
img1 = cv2.flip(selected_image,1)
#img2 = img1[:,:,:]
img2 = augment_img(img1)
showimage(selected_image,"Center Image")
showimage(img1,"Flip Image")
showimage(img2,"Augmented Image")




# generator function is used to batch the dataset so that computer
# could run the model without memory error

def generator(X_data, batch_size = 32):
    num_samples = len(X_data)
    while 1:
        shuffle(X_data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = X_data[offset:offset + batch_size]
            
            images = []
            measurements = []
            for row in batch_samples:
                steering_center = float(row[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                #directory = "..." # fill in the path to your training IMG directory
                img_center = cv2.imread(row[0])
                img_left = cv2.imread(row[1])
                img_right = cv2.imread(row[2])

                # add images and angles to data set
                images.append(img_center)
                measurements.append(steering_center)
                images.append(img_left)
                measurements.append(steering_left)
                images.append(img_right)
                measurements.append(steering_right)
                
                #Augment the data and measurement                
                augmented_images, augmented_measurements = [],[]
                for image,measurement in zip(images,measurements):                    
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)                   
                    augmented_images.append(cv2.flip(image,1))
                    augmented_measurements.append(measurement* -1.0)    
    
            X_sub_train = np.array(augmented_images)
            y_sub_train = np.array(augmented_measurements)
            #print ("X_sub_train shape", X_sub_train.shape)
            yield sklearn.utils.shuffle(X_sub_train, y_sub_train)
            
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

print ("done loading data")


#Model architecture.......................................................

print ("Training...")
start_time= time.time()
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
#--------- Nvidia Model -----------------------------
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#---------LeNet Model --------------------------------
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))
#-----------------------------------------------------

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train,y_train, validation_split =0.2, shuffle=True, epochs=3)
model.fit_generator(train_generator, samples_per_epoch = len(train_samples) / batch_size ,validation_data = validation_generator, nb_val_samples = len(validation_samples)/batch_size, nb_epoch = epoch)
                   
model.save('model.h5')
print("Finish Training...")
print("Total process time: " , time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

