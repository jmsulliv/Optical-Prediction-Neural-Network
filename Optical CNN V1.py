import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
from matplotlib.gridspec import GridSpec
import kerastuner as kt
from contextlib import redirect_stdout
from tensorflow.keras import layers
import pickle
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sys
import itertools
from tensorflow.python.keras import backend as K

"""from keras.utils.generic_uti
.1ls import CustomObjectScope
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable Tensorflow Notifcations

data = loadmat('Data/CNN Test Data (3962 Points, 100 Output)- 20210416.mat',
               squeeze_me=True,
               struct_as_record=False)

# Models - CNNTest_50 is a MAE = 0.012 model

ModelRun = 'Yes'
Model_Iter = 4
HyperP = 'Yes'

# Iter 3 + Iter 3 CNN HyperP
# Iter Cyl 1 + Model Iter 4 :  Cyl + Tri (MAE =
'''no_run_name = "CNN Output\CNNTest_50.hdf5"'''

no_run_name = "CNN Output\CNN_V1_Seg50_Iter3.hdf5"

# Import the image map -- the images of the domain to be tested/trained on
# The "Segement" refers to the interval average values -- normally 100 emissivity outputs, but segmented into 10-20-25-40-50-100 segments
# change the number after _P_ to change the training data
Images = data['ImageMap']
Output = data['Output']

# Transpose MATLAB matrix input to be compatible with Python
Output = Output.T

# Shuffle Images
Images, Output = shuffle(Images, Output)

# Load images for predictions
Pred = loadmat('Data/CNN Input for Prediction- 20210304.mat')
Prediction = Pred['ImagePredMap']

# Assign Filename based on segmentation and assignment

if ModelRun == 'No':
    filename = no_run_name
elif ModelRun != 'No':
    filename = "CNN Output\CNN_V1_Seg{0}_Iter{1}.hdf5".format(Output.shape[1], Model_Iter)

# Determine the training/testing/validation Images
Spacing = round(len(Output) * 0.8)
Val_Spacing = round(len(Output) * 0.9)
TotLen = round(len(Output))

train_Images = Images[0:Spacing, ...]
y_train = Output[0:Spacing]

val_Images = Images[Spacing:Val_Spacing, ...]
y_val = Output[Spacing:Val_Spacing]

test_Images = Images[Val_Spacing:TotLen]
y_test = Output[Val_Spacing:TotLen]

# Check the Size of the Image inputs
print(train_Images.shape)
print(val_Images.shape)
print(test_Images.shape)

train_Images = train_Images.reshape(Spacing, Images.shape[1], Images.shape[2], 1)
val_Images = val_Images.reshape((Val_Spacing - Spacing), Images.shape[1], Images.shape[2], 1)
test_Images = test_Images.reshape((Val_Spacing - Spacing), Images.shape[1], Images.shape[2], 1)

train_Images = train_Images.astype('float32')
val_Images = val_Images.astype('float32')
test_Images = test_Images.astype('float32')

print('Combined Size of Output Datasets: ', (len(y_test) + len(y_val) + len(y_train)))  # Shape of the input vector


# Function to create architecture  of the CNN --- both the model as well as the hyperparameter model
def CNN_model(Input, OutputV, Fils=[64, 128, 256], KernNum=3, Pooling=2, DropNum=0.6, l_rate=1e-3, metrics=['mae']):
    # Create Model
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(Fils[0], (KernNum, KernNum), activation=tf.nn.relu6,
                                  input_shape=(Input, Input, 1)))
    model.add(keras.layers.MaxPooling2D((Pooling, Pooling)))
    model.add(keras.layers.Conv2D(Fils[1], (KernNum, KernNum), activation=tf.nn.relu6))
    model.add(keras.layers.MaxPooling2D((Pooling, Pooling)))
    model.add(keras.layers.Conv2D(Fils[2], (KernNum, KernNum), activation=tf.nn.relu6))
    model.add(keras.layers.MaxPooling2D((Pooling, Pooling)))
    model.add(keras.layers.Conv2D(Fils[2], (KernNum, KernNum), activation=tf.nn.relu6))
    model.add(keras.layers.MaxPooling2D((Pooling, Pooling)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(DropNum))
    model.add(keras.layers.Dense(256, activation=tf.nn.relu6))
    model.add(keras.layers.Dense(OutputV, activation=tf.nn.relu6))

    opt = keras.optimizers.Adam(lr=l_rate, decay=1e-3 / 200)

    model.compile(loss="mean_absolute_error", optimizer=opt, metrics=metrics)
    return model


# Function to create architecture  of the CNN

def build_model(hp):
    inputs = tf.keras.Input(shape=(256, 256, 1))
    x = inputs
    for i in range(hp.Int('conv_blocks', 3, 5, default=4)):
        filters = hp.Int('filters_' + str(i), 32, 512, step=32)
        for _ in range(2):
            x = tf.keras.layers.Convolution2D(
                filters, kernel_size=(3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(
        hp.Int('hidden_size', 100, 500, step=25, default=300),
        activation=tf.nn.relu6)(x)
    x = tf.keras.layers.Dropout(
        hp.Float('dropout', 0, 0.5, step=0.1, default=0.4))(x)
    outputs = tf.keras.layers.Dense(100, activation=tf.nn.relu6)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='mean_absolute_error',
        metrics=['mae'])
    return model


def convert_dataset(item):
    image = item['ImageMap']
    label = item['Segment_P_50']

    label = label.T

    image = tf.dtypes.cast(image, 'float32')
    return image, label


# Model Run -- hyper parameter tuning included in model construction
if ModelRun != 'No' and HyperP != 'Yes':
    model = CNN_model(Images.shape[1], Output.shape[1])

    # Save Locations
    save_dir = 'Saved_Models_NoFold/'
    model_save = "CNN_{0}".format(Output.shape[1])

    # Establish callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + model_save + '.hdf5',
                                                    montior='mae', verbose=1,
                                                    save_best_only=True, mode='min')
    '''earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=18, verbose=0, mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.75, patience=8, verbose=1,
                                                      min_delta=0.5e-4,
                                                      mode='min')'''

    '''callbacks_list = [checkpoint, earlyStopping, reduce_lr_loss]'''
    callbacks_list = [checkpoint]

    # Fit and Run Results
    results = model.fit(x=train_Images, y=y_train, epochs=200, batch_size=32,
                        validation_data=(val_Images, y_val), callbacks=callbacks_list,
                        verbose=2)

    # Save File
    model.save(filename)

elif ModelRun != "No" and HyperP == "Yes":
    tuner = kt.Hyperband(build_model, objective='val_mae', max_epochs=10, hyperband_iterations=3,
                         directory='CNN_HP_Tuning', project_name='Iter1_wCyl')

    '''train_set = data.map(convert_dataset).shuffle(1000).batch(100).repeat()'''
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=2, verbose=0, mode='min')
    callbacks_list = [earlyStopping]

    tuner.search_space_summary()

    tuner.search(x=train_Images, y=y_train, batch_size=20, epochs=30, verbose=2, callbacks=callbacks_list,
                 validation_data=(val_Images, y_val))

    model = tuner.get_best_models(num_models=1)
    tuner.results_summary(num_trials=1)

    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hp)

    print(best_hp)

    # Save Locations for Model Checkpoints
    save_dir = 'Saved_Models_hp_Output/'
    model_save = "CNN_{0}_Iter{1}".format(Output.shape[1], Model_Iter)

    # Establish callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + model_save + '.hdf5',
                                                    montior='mae', verbose=1,
                                                    save_best_only=True, mode='min')
    fitcalls = [checkpoint, earlyStopping]

    model.fit(train_Images, y_train, epochs=100, validation_data=(val_Images, y_val), callbacks=fitcalls,
              batch_size=20, verbose=2)

    # Save File
    model.save(filename)

# Special Load Function to account for the Relu layer
with tf.keras.utils.CustomObjectScope(
        {'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = keras.models.load_model(filename)

'''model = keras.models.load_model(pretrained_model_name,
                       custom_objects={'relu6': keras_applications.mobilenet.relu6,
                                       'DepthWiseConv2D': keras.applications.mobilenet.DepthwiseConv2D})'''
# -------------  Evaluating the Model from Test Data ------------
# Evaluate and Predict off of the Test Data (Unseen in training)
Test_Image_Pred = model.predict(test_Images)
test_loss, test_acc = model.evaluate(test_Images, y_test)

# Test Accuracy of Prediction Test Dataset compared to true results
test_num = 32

# This chooses a particular point to show. Since the splitting process shown earlier is random, this number is essentially meaningless
# as the test prediction dataset will change every time this code is run. This is the benefit of the static prediction dataset, the position and number
# datapoints does not change from iteration to iteration.

test_pred_plot = Test_Image_Pred[test_num]
test_true = y_test[test_num]

# Simplify matrix to one column for plotting
plt_img = test_Images[test_num]
print(plt_img.shape)
plt_img = np.squeeze(plt_img, axis=(2,))
print(plt_img.shape)

# Plotting of the outcome -- Model Prediction vs the Simulation Results (Ground Truth)
hfont = {'fontname': 'Times New Roman'}  # Font Specification

font = {'family': 'normal',
        'weight': 'bold',
        'size': 26}
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

(width, height) = (16, 8)
fig = plt.figure(figsize=(width, height))

(n_row, n_col) = (1, 2)
gs = GridSpec(n_row, n_col)

wl = np.linspace(0.3, 16, Output.shape[1])

ax = fig.add_subplot(gs[0])
plt.plot(wl, test_true, linestyle='--', marker='o', color='b', label="FDTD Data")
plt.plot(wl, test_pred_plot, linestyle='--', marker='o', color='r', label="Predicted Data")
plt.title('Simulation vs. CNN Output', fontsize=26)
plt.legend(loc='lower left')
plt.ylabel('Emissivity', fontsize=24)
plt.xlabel('Wavelength (\u03BCm)', fontsize=24)
plt.ylim([0, 1])
plt.xlim([0.3, 16])
plt.legend()

ax = fig.add_subplot(gs[1])
plt.imshow(plt_img)
plt.title('Simulation Image Input')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')

# -------- PREDICTION From Gridpoint Dataset ---------
# Prediction off of the grid point data - geometric gridpoints are used to generate images and then predict the emissivity
# of the input image geometry
Pred_Len = 2500
Prediction_Images = Prediction.reshape(Pred_Len, 256, 256, 1)
Output_Pred = model.predict(Prediction_Images)

# Assign Outputs for Post-Processing
mat_ids_1 = dict(
    CNN_pred=Output_Pred,
)

filename_mat = 'Prediction Data\TestCNNOutput_{0}.mat'.format(Output.shape[1])
savemat(filename_mat, mat_ids_1)

print('---- CNN - Prediction for Grid Geometry ----')
print(Output_Pred[1])

plt.show()
