import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable Tensorflow Notifcations

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
from contextlib import redirect_stdout
from tensorflow.keras import layers
import pickle
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
import sys
import itertools

# Title
#  <--------------- Jonathan Sullivan (JS) Optical Prediction Code V3  ------------->
#   <--------------- Last Updated: 4/8/2021 by JS                    ------------->

ModelRun = 'No'  # Yes or No

# Model Number for Saving/Loading Data - this number determines what model number you are loading and/or saving
Iter_number = 15

# Data Type (FDTD or RCWA, FDTD_New)
Input_Type = 'FDTD_Multi'

# Model Calibration -- Hyperparameter tuning
Model_Calib = 'No'

# K-fold Cross-Validation - manual hyperparameter tuning
K_Fold = 'No'

# Note: FDTD_New is the preferred type, all other models are outdated
# Good/Excellent MODELS
# --- for 27775 Datapoint set-- Model V3 #12 --- MAE loss < 5.4e-4, VAL MAE < 0.0121
# --- for 27775 Datapoint set-- Model V3 #14 --- MAE loss < 4.5e-4, VAL MAE < 0.0082
# --- for 31775 Datapoint set-- Model V3 #15 --- MSE loss < 3.8e-4, VAL MAE < 0.0093

# Specification of File Name based on Input Type
if Input_Type == 'FDTD':
    model_filename = "Models\model_Ni%dFD" % Iter_number
    model_save = 'model_Ni%dFD' % Iter_number
elif Input_Type == 'FDTD_New':
    model_filename = 'Models\model_V2_Ni%dFD' % Iter_number
    model_save = 'model_V2_Ni%dFD' % Iter_number
elif Input_Type == 'FDTD_Multi':
    model_filename = 'Models\model_V3_Ni%dFD' % Iter_number
    model_save = 'model_V3_Ni%dFD' % Iter_number

# Print the name to confirm
print('Filename Desired for Simulation:', model_filename)

# Breakdown of Models -
# FDTD Models 3.0 (new code structure), Model_V2_Ni#FD
#

# Title - Data: Loading and Assignment
# Note: AR, Segmented and Filtered Data Selection - Anything other than "Yes" will result in loading the default FDTD Dataset
AR_Filter = 'Yes'

if AR_Filter == 'Yes':
    # Load data to preserve the structure
    # data = loadmat('Normalized AR Data for FDTD Simulations (1627 Starting Points)- 20201015.mat', squeeze_me=True,
    #               struct_as_record=False)
    data = loadmat('Data/Normalized AR Data for FDTD Simulations (31775 Starting Points)- 20210409.mat',
                   squeeze_me=True,
                   struct_as_record=False)

    # To change the constraining Aspect Ratio, change the middle number in (AR_#_LinN) to whatever the desired Aspect Ratio is (assuming it is contained in the input structure)
    # X data is used for the Input neurons and Y data is the predicted/simulated emissivity data
    # TODO: Attempted to pass the AR section as a variable that could be changed based on an "AR" variable but this did not
    #   work. Find a way to simplify this to one number, if possible

    X = data['ARInput'].AR_100_LinN
    Y = data['AROutput'].AR_100_LinN

    ARCond = data['Cond'].AR_100_LinN

    # Load Predicted/Static Dataset that is Normalized for each AR condition - starts at 256 but has datapoints removed
    # as the aspect ratio decreases
    XP = data['PredictInput'].AR_100_LinN
    YP = data['PredictOutput'].AR_100_LinN

    # Data for plotting and checking the normalization factors used
    PCon = data['PredictCon'].AR_100_LinN
    XPlot = data['ActInp'].AR_100_LinN

    # Grid Data Load
    GDX = data['GridData'].AR_100_LinN
    Grid = data['GridCoords'].AR_100_LinN

else:
    data = loadmat('Nickel Restructured FDTD Data (100 Points, 1627 Output)- 20201011.mat')

    # X data is used for the Input neurons and Y data is the predicted/simulated emissivity data
    X = data['Input']
    Y = data['Output']
    # Y = data['T']

    # Load Predicted/Static Dataset that is not normalized -- full 256 datapoints
    dataP = loadmat('Prediction Dataset Normalized for (1627 Points) Dataset- 20201013.mat')

    XP = dataP['PredictIn']
    YP = dataP['PredictOut']

# End of Data Conditions

# Transpose Data from Matlab data to be compatible with python
X = X.T
Y = Y.T

# Shuffle Data X, Y from Dataset
X, Y = shuffle(X, Y)

# Setup Prediction Dataset
XP = XP.T
YP = YP.T
XPlot = XPlot.T

# Seed Information
seed = 7
np.random.seed(seed)

# Hyperparameter Tuning
epochnum = 200
results = 0

# Print Shapes

# Title ------ Section conditions based on Calib/Test Size Specification --------
#  ----------------------- CONDITIONS FOR TESTING -------------------------------
# Testing the Best Parameters for the Neural Network
if Model_Calib == 'Yes':
    # Note: Specification for the number of layers and size of layers
    # grid specification of points/parameters to be examined
    Results = 0
    Test_Ratio = 0.8
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - Test_Ratio), random_state=seed)

    '''print(x_train.shape[1])
    print(y_train.shape[1])'''

elif Model_Calib == 'No':

    # Note: Default Number of Layers/Neurons if Model Calibration is not being performed
    # Note: Previous optimum 1600/9
    N_Neuron = 1050
    N_Layer = 14

    # Note: Default Test Ratio Value if Test Size isn't being tested
    Test_Ratio = 0.9
    Val_Ratio = 0.2 / Test_Ratio

    # New Method - new dataset each time
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - Test_Ratio), random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=Val_Ratio, random_state=seed)

    # Print Output/Input vector size to confirm
    print('-------------------\n')  # Shape of the input vector
    print('Size of Model Input         :', len(X))  # Shape of the input vector
    print('Combined Size of Datasets   :', (len(x_train) + len(x_val) + len(x_test)))  # Shape of the input vector
    print('\n-------------------')
    print('Shape of Input Vector       :', x_train.shape)  # Shape of the input vector
    print('Shape of Output Vector      :', y_train.shape)  # Shape of the output vector
    print('Shape of Input Val Vector   :', x_val.shape)
    print('Shape of Output Val Vector  :', y_val.shape)
    print('Shape of Input Test Vector  :', x_test.shape)
    print('Shape of Output Test Vector :', y_test.shape)
    print('\n-------------------\n')
    print('Input and Model Input Diff  : ',
          (len(X) - (len(x_train) + len(x_val) + len(x_test))))  # Shape of the input vector
    print('\n-------------------\n')


# Title: ----------------- CLASS/FUNCTION DEFINITIONS ---------------------
# Functions:
# Category 1: MultilayerSequential defines the model based on the inputs, MSE is the default regression loss function
# Category 2: Neuron Layer returns a variable that specifies the number of layers/number of neurons per layer
# Category 3: Functions for plotting/running the simulation evaluation tools

def MultiLayerSequential(input_dim, neurons, outputs, metrics=['mae']):  # Previously using MSE
    model = keras.Sequential()
    model.add(layers.Dense(neurons[0], input_dim=input_dim, kernel_initializer='normal', activation=tf.nn.relu6,
                           kernel_regularizer=keras.regularizers.l2(1e-8),
                           name="Input_layer"))  # Previously used 'relu', 'tf.nn.relu6'
    for i in range(1, len(neurons)):
        model.add(layers.Dense(neurons[i], kernel_initializer='normal', activation=tf.nn.relu6,
                               kernel_regularizer=keras.regularizers.l2(1e-8)))  # Previously used 'relu, no regularization function', , kernel_regularizer='l1'
    model.add(layers.Dense(outputs, kernel_initializer='normal', name="Output_layer"))  # , activation='sigmoid')
    opt = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=metrics)
    return model


def Neuron_Layer(N_Number, Num_Layers):
    global Neuron_Vec
    Neuron_Vec = []
    for j in range(Num_Layers):
        Neuron_Vec.append(N_Number)

    print(Neuron_Vec)
    return Neuron_Vec


def get_model_name(k):
    return 'model_' + str(k) + '.h5'

def create_model(neurons=1050, layer_num=14, metrics=['mae'], l_rate=1e-3,
                     init='normal', beta_1=0.9, beta_2=0.999):

        # Parameters
        outputs = 200
        Inputs = 304

        # Create Model
        model = keras.Sequential()
        model.add(layers.Dense(neurons, input_dim=Inputs, kernel_initializer=init, activation=tf.nn.relu6,
                               kernel_regularizer=keras.regularizers.l2(1e-8),
                               name="Input_layer"))  # Previously used 'relu', 'tf.nn.relu6'
        for i in range(1, layer_num):
            model.add(layers.Dense(neurons, kernel_initializer=init, activation=tf.nn.relu6,
                                   kernel_regularizer=keras.regularizers.l2(1e-8)))
        model.add(layers.Dense(outputs, kernel_initializer=init, name="Output_layer",
                               activation='sigmoid'))  # , activation='sigmoid')
        opt = keras.optimizers.Adam(learning_rate=l_rate, beta_1=beta_1, beta_2=beta_2)
        # opt = keras.optimizers.Adam(learning_rate=l_rate)

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=metrics)
        return model


# Title <------------------ MODEL CREATION, OPTIMIZATION, AND RUNNING --------------------------------->
# --------- The section that runs is based on the Model Run Answer (Yes/No) -------------
# and the Model Calib/Test Size answers (Yes/No) Sections Below will run the models according to the conditions
# specified previously


# Note: For if the model should be run and the test size/ratio is being tested

if ModelRun == 'Yes' and Model_Calib == 'Yes':

    # Epochs = 200, established as optimal value
    model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, epochs = 200, verbose=0)

    batch_size = [80]
    # epochs = [50, 100, 150, 200, 250]
    # l_rate = [1e-5, 2e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2.5e-3, 0.01]
    # init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    init = ['glorot_normal']

    # param_grid = dict(batch_size=batch_size, epochs=epochs, learn_rate=learn_rate, init_mode=init_mode, beta_1=beta_1,
    #                   beta_2=beta_2,neurons=neurons)
    # Used a parameter to specify the optimizer

    # param_grid = dict(init=init, l_rate=l_rate, batch_size=batch_size, epochs=epochs)
    param_grid = dict(init=init, batch_size=batch_size)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', refit=True, cv=5)

    grid_result = grid.fit(x_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    std = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, params in zip(means, std, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    keras.backend.clear_session()

# Note: Section for if the model is being run/constructed by neither optimization is used
# JS future: less hard coded values here, goal to be more streamlined for rapid model construction

elif ModelRun == 'Yes' and Model_Calib == 'No' and K_Fold == 'Yes':
    # Manual K-fold cross-validation for single value/target estimation
    # Create Input/Target Arrays
    inputs = np.concatenate((x_train, x_val), axis=0)
    targets = np.concatenate((y_train, y_val), axis=0)
    num_folds = 5

    # Define Fold Function
    k_fold = KFold(n_splits=num_folds, shuffle=True)

    fold_no = 1

    # Define per-fold score containers
    TEST_ACCURACY = []
    TEST_LOSS = []

    save_dir = 'Saved_Models/'

    for train, test in k_fold.split(inputs, targets):
        Neuron_Vec = Neuron_Layer(N_Neuron, N_Layer)

        # Redefine the k-fold output to a more friendly version
        x_train = inputs[train]
        y_train = targets[train]
        y_val = targets[test]
        x_val = inputs[test]

        # Create new model
        model = MultiLayerSequential(x_train.shape[1], Neuron_Vec, y_train.shape[1])

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + get_model_name(fold_no),
                                                        montior='val_mae', verbose=1,
                                                        save_best_only=True, mode='min')
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=20, verbose=0, mode='min')
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=10, verbose=1,
                                                              min_delta=0.5e-4,
                                                              mode='min')

        callbacks_list = [checkpoint, earlyStopping, reduce_lr_loss]

        results = model.fit(x_train, y_train, epochs=epochnum, verbose=0, callbacks=callbacks_list,
                            validation_data=(x_val, y_val))

        # Load Best Model to evaluate the performance of the model
        model.load_weights("Saved_Models/" + model_filename + "_" + str(fold_no) + ".h5")
        results = dict(zip(model.metrics_names, results))

        scores = model.evaluate(x_test, y_test, verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
        TEST_ACCURACY.append(scores[1])
        TEST_LOSS.append(scores[0])

        tf.keras.backend.clear_session()

        # Increase fold number
        fold_no = fold_no + 1

    # Print Averaged Scores and Metrics for k-fold validation method
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(TEST_ACCURACY)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {TEST_LOSS[i]} - MAE: {TEST_ACCURACY[i]}')
        print('------------------------------------------------------------------------')

    # Save History of File
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(TEST_ACCURACY)} (+- {np.std(TEST_ACCURACY)})')
    print(f'> Loss: {np.mean(TEST_LOSS)}')
    print('------------------------------------------------------------------------')

    H_fname = 'History/results.pkl'
    with open(H_fname, 'wb') as f:
        pickle.dump(results.history, f)

    History = results.history

    model.save(model_filename)

elif ModelRun == 'Yes' and Model_Calib == 'No' and K_Fold == 'No':
    save_dir = 'Saved_Models_NoFold/'
    Neuron_Vec = Neuron_Layer(N_Neuron, N_Layer)

    # Create new model
    model = MultiLayerSequential(x_train.shape[1], Neuron_Vec, y_train.shape[1])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + model_save + '.h5',
                                                    montior='mae', verbose=1,
                                                    save_best_only=True, mode='min')
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=20, verbose=0, mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, verbose=1,
                                                          min_delta=0.5e-4,
                                                          mode='min')

    callbacks_list = [checkpoint, earlyStopping, reduce_lr_loss]

    results = model.fit(x_train, y_train, epochs=epochnum, verbose=2, callbacks=callbacks_list,
                        validation_data=(x_val, y_val))

    model.load_weights(save_dir + model_save + '.h5')

    # Save History of File
    H_fname = 'History/results.pkl'
    with open(H_fname, 'wb') as f:
        pickle.dump(results.history, f)

    History = results.history

    model.save(model_filename)

# Note: Section for if the model is not intended to be run but only to be LOADED and PREDICTED

elif ModelRun == 'No':
    # Print Size of the X_train Dataset
    print('====== MODEL WAS NOT RUN: MODEL LOADED ======')
    print('Simulation Filename Loaded:', model_filename)
    print('Size of the Training Dataset used in Model Run: ', len(x_train))
    print('\n-------------------\n')

    # Print and Load File Information
    model = keras.models.load_model(model_filename)

    # Load History of File
    H_fname = 'History/results.pkl'
    with open(H_fname, 'rb') as f:
        History = pickle.load(f)

# Title ---------- Print Outcomes from Optimizations ----------------------------
# Print Outcomes from Optimizations/Sweeps -- this section needs to be reworked after model calib was changed
if Model_Calib == 'Yes' and ModelRun == 'Yes':
    with open('Model_Calib.pkl', 'wb') as f:
        pickle.dump([grid_result.best_score_, grid_result.best_params_], f)

    means = grid_result.cv_results_['mean_test_score']
    std = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    f = open('ModelCalib.txt', 'w')

    for mean, std, params in zip(means, std, grid.cv_results_['params']):
        f.write('{1} (+/-%{2}f) for {3}\n'.format(*mean, *std * 2, *params))

    f.close()

    print("------ Detailed classification report: -------\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set. \n")
    y_true, y_pred = y_test, grid.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()

    # Terminate the code here
    exit()


# Title: ----------------- CLASS/FUNCTION DEFINITIONS FOR POST-PROCESSING---------------------
# Functions: Plot history: plot the epoch/results relationship

def plot_hist(History, x_start=0, x_end=len(History['val_loss']), logscale=1):
    metric = History['mae']
    val_metric = History['val_mae']
    loss = History['loss']
    val_loss = History['val_loss']
    if logscale == 1:
        metric = np.log10(metric)
        val_metric = np.log10(val_metric)
        loss = np.log10(loss)
        val_loss = np.log10(val_loss)

    epochs = range(len(metric))

    (width, height) = (16, 4)
    fig = plt.figure(figsize=(width, height))

    hfont = {'fontname'  'Helvetica'}  # Font Specification

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'serif'})

    plt.rc('font', size=15)  # controls default text sizes
    plt.rc('axes', titlesize=15)  # fontsize of the axes title
    plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
    plt.rc('legend', fontsize=15)  # legend fontsize
    plt.rc('figure', titlesize=15)  # fontsize of the figure title

    (n_row, n_col) = (1, 2)
    gs = GridSpec(n_row, n_col)

    ax = fig.add_subplot(gs[0])
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=3)
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
    if logscale == 1:
        plt.title('Log$_{10}$(Loss) vs. Epochs', fontweight='bold')
    else:
        plt.title('Loss vs. Epochs', fontweight='bold')
    ax.set_xlim(x_start, x_end)
    plt.legend()

    ax = fig.add_subplot(gs[1])
    plt.plot(epochs, metric, 'b-', label='Training MAE', linewidth=3)
    plt.plot(epochs, val_metric, 'r--', label='Validation MAE', linewidth=2)
    if logscale == 1:
        plt.title('Log$_{10}$(MAE) vs. Epochs', fontweight='bold')
    else:
        plt.title('MAE vs. Epochs', fontweight='bold')
    ax.set_xlim(x_start, x_end)
    plt.legend()


#  Title ---------- Evaluate Accuracy from Optimizations / Sweeps / Model(s) ----------------------------
# Plot History of Model Fitting/Training
plot_hist(History, logscale=0)
plot_hist(History, logscale=1)

print('\n====== MODEL SUMMARY ======')

# Save Model Summary to a file with trainable parameters
with open('Model Summary\model_V2_Ni%dFD_summary.txt' % Iter_number, 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Print/show model config that was saved to file previously
print(model.summary())
model.get_config()

print('\n====== PREDICTION and EVALUATION METRICS: ====== \n')

# TODO: The section with the prefab test-set needs more work and is not complete yet

# Accuracy of Test Dataset (from the split earlier)
mae_loss, mae_acc = model.evaluate(x_test, y_test, verbose=0)
print('MAE of the Model (Test Dataset)       :', mae_acc)

# Test Accuracy of Prediction Test Dataset compared to true results
test_num = 3

'''print(XP.shape)
print(x_test.shape)
print(x_test[test_num].shape)
print(XP[test_num].shape)'''

test_pred = model.predict(x_test, verbose=0)
test_pred_plot = test_pred[test_num]
test_true = y_test[test_num]

# Simplify matrix to one column for plotting
x_test_1 = x_test[test_num]
x_pyr_test = round(x_test_1[0]*10, 2)
z_pyr_test = round(x_test_1[1]*10, 2)
x_test_plot = (x_test_1[4:104]) * (16 - 0.3) + 0.3

# This section saves to a mat file the actual test data (both x and y) and then the y predicted data with the accuracy
# value

mat_ids_1 = dict(
    test_pred=test_pred,
    test_pred_plot = test_pred_plot,
    XTest=x_test,
    YTest=y_test,
    test_true=test_true,
    mae_acc=mae_acc
)

filename_mat3 = 'Prediction Data\model_V2_Ni%dFD_TestPrediction.mat' % Iter_number
savemat(filename_mat3, mat_ids_1)

hfont = {'fontname'  'Helvetica'}     # Font Specification

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': 'serif'})

plt.figure()
plt.plot(x_test_plot, test_pred_plot[:100], linestyle='--', marker='o', color='b', label="FDTD Data")
plt.plot(x_test_plot, test_true[:100], linestyle='--', marker='o', color='r', label="Predicted Data")
plt.title('Test Dataset: \n X span = {0} \u03BCm, Z span = {1} \u03BCm'.format(x_pyr_test, z_pyr_test), fontsize=16)
plt.legend(loc='lower left')
plt.ylabel('Emissivity', fontsize=14)
plt.xlabel('Wavelength (\u03BCm)', fontsize=14)
plt.ylim([0, 1])
plt.xlim([min(x_test_plot), max(x_test_plot)])
plt.savefig('Prediction vs Test.png')


# Load Standardized Prediction Dataset and use to predict/plot - new x-test and y-test based on static prediction values
y_test = YP
x_test = XP

# Test Dataset to Plot and Compare

# Number in pre-fab dataset to test. Max of 256 Points, TODO: Exact specifications on the test number and geometry/aspect ratio being used for each of those points
test_number = 192
# Test Numbers: 168 and 54, 188

# Output/Input are fixed sweeps of geometric parameters, with 256 data points for testing/predicting. Currently this dataset
# has been normalized Linearly, 'PLin_Input' and has the standard output of 100 emissivity points 'PredictOut'

# ----- Standardized Prediction Dataset ------
# New Predictions based on standardized prediction dataset
prediction = model.predict(x_test, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('MAE of the Static Prediction Dataset  :', test_acc)

# ----- GRID Prediction Dataset -----
# Grid Data Prediction with Timing Function
tic = time.perf_counter()

# Grid Data Predict
GDX = GDX.T
Grid_Predict = model.predict(GDX, verbose=0)

toc = time.perf_counter()

# ----- Unseen Prediction Dataset -----
# Predictions from Unseen Data - Ti
# Load Unseen (US) Data
USData = loadmat('Data/Normalized Unseen Data - 20210409.mat',
                   squeeze_me=True,
                   struct_as_record=False)

Unseen_Input = USData['Unseen_Input'].T
Unseen_Output = USData['Unseen_Output'].T
Unseen_Input2 = USData['Unseen_Input2'].T
Unseen_Output2 = USData['Unseen_Output2'].T

USPred = model.predict(Unseen_Input, verbose=0)
US_loss, US_acc = model.evaluate(Unseen_Input, Unseen_Output, verbose=0)

USPred2 = model.predict(Unseen_Input2, verbose=0)
US_loss2, US_acc2 = model.evaluate(Unseen_Input2, Unseen_Output2, verbose=0)

print('MAE of the Ti Unseen Prediction Dataset:', US_acc)
print('MAE of the Al2O3 Prediction Dataset    :', US_acc2)

# Print timing from Grid Predict
print(f"Gridpoint Predictions occured for {len(GDX):4.0f} grid points in {toc - tic:0.4f} seconds")


# Export Data to Matlab File for Post-Processing -- Included in this: Standardized, Grid, and Unseen Predictions
mat_ids = dict(
    Prediction=prediction,
    XPlot=XPlot,
    XP=XP,
    YP=YP,
    Grid_Predict=Grid_Predict,
    GridCoords=Grid,
    USPred=USPred,
    US_loss=US_loss,
    US_acc=US_acc,
    US_out=Unseen_Output,
    USPred2= USPred2,
    US_loss2=US_loss2,
    US_acc2=US_acc2,
    US_out2=Unseen_Output2)

filename_mat2 = 'Prediction Data\model_V2_Ni%dFD_Prediction.mat' % Iter_number
savemat(filename_mat2, mat_ids)

# Print the dataset being used as well as the prediction for it
'''print(x_test[test_number])
print(prediction[test_number])
print(y_test[test_number])'''
x_test = XP
y_test = YP

y_test = y_test[test_number]
prediction_e = prediction[test_number]

# De-normalize the geometric datapoint for the plottitle
x_test_p = XPlot[test_number]
x_pyr = round(x_test_p[0]*1e6, 2)
z_pyr = round(x_test_p[1]*1e6, 2)

# de-normalize the x plotting for 0.3 and 10 um system
# future: this will be revamped significantly in the future to be more consistent and easy to use
x_plot_1 = (x_test_p[4:104]) * (10 - 0.3) + 0.3
y_test = y_test[:100]
prediction_e = prediction_e[:100]

# Plotting and Specification of Plot Properties
hfont = {'fontname'  'Helvetica'}     # Font Specification

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': 'serif'})

# future: plotting section will be reworked to be more streamlined with user selection of previous data

# Plotting for Wavelength Data
plt.figure()
plt.plot(x_plot_1, y_test, linestyle='--', marker='o', color='b', label="FDTD Data")
plt.plot(x_plot_1, prediction_e, linestyle='--', marker='o', color='r', label="Predicted Data")
plt.title('X span = {0} \u03BCm, Z span = {1} \u03BCm'.format(x_pyr, z_pyr), fontsize=16)
plt.legend(loc='lower left')
plt.ylabel('Emissivity', fontsize=14)
plt.xlabel('Wavelength (\u03BCm)', fontsize=14)
plt.ylim([0, 1])
plt.xlim([min(x_plot_1), max(x_plot_1)])
plt.savefig('Prediction.png')
plt.show()

# Non-functional output of image of network
'''tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
)'''
'''tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)'''
