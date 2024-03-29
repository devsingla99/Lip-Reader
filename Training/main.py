import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
#from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
import pylab as plt

# path to the model weights files.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 90, 90 

train_data_dir = 'dataset-small/train'
validation_data_dir = 'dataset-small/val'
#
#nb_train_samples = 23400
nb_train_samples = 2600
nb_validation_samples = 400
nb_epoch = 20

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height,3)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1',trainable=False))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2',trainable=False))
model.add(MaxPooling2D((2, 2), padding='same',strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1',trainable=False))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2',trainable=False))
model.add(MaxPooling2D((2, 2), padding='same',strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1',trainable=False))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2',trainable=False))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3',trainable=False))
model.add(MaxPooling2D((2, 2), padding='same',strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1',trainable=False))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2',trainable=False))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3',trainable=False))
model.add(MaxPooling2D((2, 2), padding='same',strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1',trainable=False))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2',trainable=False))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3',trainable=False))
model.add(MaxPooling2D((2, 2),padding='same', strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
'''
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
'''
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
# with l2 regularizer
#, W_regularizer=l2(0.1))
top_model.add(Dense(4096, activation='relu'))
# drop out layer
top_model.add(Dropout(0.5))
# with l2 regularizer
top_model.add(Dense(4096, activation='relu'))
# drop out layer
top_model.add(Dropout(0.5))
top_model.add(Dense(20, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

# print out a look at the model
#model.summary()
#top_model.summary()

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
#for layer in model.layers[:25]:
#    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])


#set to heavy augmented mode
heavy_augmentation = True
if heavy_augmentation:
    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.5,
        rescale=1./255,
        shear_range=0.2,
        channel_shift_range=0.5,
        fill_mode='nearest')
else:
    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

#    train_datagen = ImageDataGenerator(
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=10,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=10,
        class_mode='categorical')

# checkpoint
filepath="model/weights-VggFinetune-{epoch:02d}-{val_accuracy:.2f}.f5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# fine-tune the model
history = model.fit_generator(
        train_generator,
        steps_per_epoch=3064,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=765, callbacks=callbacks_list)
#model.save_weights("vgg-finetune-model-lr=5e-3.h5")
# steps_per_epoch=nb_train_samples,
#validation_steps=nb_validation_samples
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy-graph-for-vgg-finetune-lr=1e-3&nestero")
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss-graph-for-vgg-finetune-lr=1e-3$nestero')
