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
from sklearn.preprocessing import PowerTransformer
from tensorflow.keras.utils import plot_model
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
import sys
import itertools

# Title
#  <--------------- Jonathan Sullivan (JS) Optical Prediction Code V6  ------------->
#   <--------------- Last Updated: 12/02/2021 by JS                    ------------->


# =======================================================================================================
# =============================== SECTION: FILE AND MODE SELECTION  =====================================
# =======================================================================================================

# This will determine if the model is generated or if it is loaded from previous generations
ModelRun = 'Yes'  # Yes or No

# Model Number for Saving/Loading Data - this number determines what model number you are loading and/or saving
Iter_number = 1

# Data Type (FDTD or RCWA, FDTD_New)
Input_Type = 'FDTD_Multi'

# Model Calibration -- Hyperparameter tuning
Model_Calib = 'No'

# K-fold Cross-Validation - manual hyperparameter tuning
K_Fold = 'No'

# Incorporation of Unseen Data into Datasets - Yes for testing, no for not
Unseen_Test = 'No'

# ------------------- PREFERRED/SAVED MODELS ------------------------------
# Note: FDTD_New is the preferred type, all other models are outdated
# Good/Excellent MODELS- -- Models for V6

# ---- Model 22: 8 Input Neurons, 2 output (Lin: C, G) - (Log: n) - (Quantile Uniform -- k, ereal, TS, eim)
#              5 simulations included in unseen data, 10 Ti, 10 Al2O3 -- 200/1 layer + 400 neurons/7 layers, with new
#               MLP structure (4/100 and 4/100)
# ---- Model 23: 8 Input Neurons, 2 output (Lin: C, G) - (Log: n) - (Quantile Uniform -- k, ereal, TS, eim)
#              0 Unseen Data Included -- 200/1 layer + 400 neurons/7 layers, with new
#               MLP structure (4/100 and 4/100)

# Specification of File Name based on Input Type
if Input_Type == 'FDTD':
    model_filename = "Models\model_Ni%dFD" % Iter_number
    model_save = 'model_Ni%dFD' % Iter_number
elif Input_Type == 'FDTD_New':
    model_filename = 'Models\model_V7_Ni%dFD' % Iter_number
    model_save = 'model_V7_Ni%dFD' % Iter_number
elif Input_Type == 'FDTD_Multi':
    model_filename = 'Models\model_V7_Ni%dFD' % Iter_number
    model_save = 'model_V7_Ni%dFD' % Iter_number

# Print the name to confirm
print('Filename Desired for Simulation:', model_filename)


# =================================================================
# ---------------------- LOADING THE DATA -------------------------
# =================================================================
# Load data to preserve the structure
data = loadmat('Data/35500 Simulation Dataset - 8 Input 2 Output - 20220208.mat',
               squeeze_me=True,
               struct_as_record=False)

# Load Training Input/Output
# Data is already normalized and transposed

X = data['Main_Input']
Y = data['Main_Output']

# Seed Information
seed = 7
np.random.seed(seed)

# Hyperparameter Tuning
epochnum = 200
results = 0

# =======================================================================================================
# ================================ SECTION: SPLIT AND PREPARE DATA  =====================================
# =======================================================================================================


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
    # Note: Previous optimums: 1600/9, 1050/14
    N_Neuron = 400
    N_Layer = 11
    MLP_Size = 100
    MLP_layers = 4
    MLP_Input = (4, 1)

    # Note: Default Test Ratio Value if Test Size isn't being tested
    Test_Ratio = 0.9
    Val_Ratio = 0.2 / Test_Ratio

    # New Method - new dataset each time
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - Test_Ratio), random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=Val_Ratio, random_state=seed)

    # Divide into new Datasets for Multi-MLP scheme
    x_train1 = x_train[:, 0:4]
    x_train2 = x_train[:, 4:8]

    x_val1 = x_val[:, 0:4]
    x_val2 = x_val[:, 4:8]

    x_test1 = x_test[:, 0:4]
    x_test2 = x_test[:, 4:8]

    # Print Output/Input vector size to confirm
    print('-------------------\n')  # Shape of the input vector
    print('Size of Model Input         :', len(X))  # Shape of the input vector'''
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

# =======================================================================================================
# ================================== SECTION: FUNCTION DEFINITIONS  =====================================
# =======================================================================================================


# Title: ----------------- CLASS/FUNCTION DEFINITIONS ---------------------
# Functions: These functions are used to generate the models, for optimization, and for the neuron construction

def MultiLayerSequential(input_dim, neurons, outputs, metrics=['mae']):  # Previously using MSE
    model = keras.Sequential()
    model.add(layers.Dense(neurons[0], input_dim=input_dim, kernel_initializer='normal', activation=tf.nn.relu6,
                           kernel_regularizer=keras.regularizers.l2(1e-8),
                           name="Input_layer"))  # Previously used 'relu', 'tf.nn.relu6'
    for i in range(1, len(neurons)):
        model.add(layers.Dense(neurons[i], kernel_initializer='normal', activation=tf.nn.relu6,
                               kernel_regularizer=keras.regularizers.l2(
                                   1e-8)))  # Previously used 'relu, no regularization function', , kernel_regularizer='l1'
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
                           activation=tf.nn.LeakyReLU(alpha=0.05)))  # , activation='sigmoid')
    opt = keras.optimizers.Adam(learning_rate=l_rate, beta_1=beta_1, beta_2=beta_2)
    # opt = keras.optimizers.Adam(learning_rate=l_rate)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=metrics)
    return model


def DNNModel(input_dim, neurons, outputs):
    # Define Input
    Input = keras.layers.Input(shape=(input_dim,))
    # Loop Over the Filter Input
    for jj in range(1, len(neurons)):
        if jj == 1:
            x2 = Input
        x2 = keras.layers.Dense(neurons[jj], kernel_initializer='normal', activation=tf.nn.relu6,
                               kernel_regularizer=keras.regularizers.l2(1e-8))(x2)  # Previously used 'relu, no
        # regularization function', , kernel_regularizer='l1'

    # Pass the model along
    Output = keras.layers.Dense(outputs, activation=tf.nn.relu6)(x2)
    model = keras.models.Model(Input, Output)
    return model


# =======================================================================================================
# ======================== SECTION: MODEL TRAINING AND OPTIMIZATION  ====================================
# =======================================================================================================

# Depending on what selections are made for "model run","model calib", and "K-fold", the appropriate sections
# will run. Generally, we only run

# =================================================================
# ---------------- MODEL OPTIMIZATION USING GRID-SEARCH -----------
# =================================================================

# Note: This is an outdated module for a previous version of the data without the 2 MLP setup

if ModelRun == 'Yes' and Model_Calib == 'Yes':

    # Epochs = 200, established as optimal value
    model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, epochs=200, verbose=0)

    batch_size = [50, 80, 100]
    l_rate = [1e-5, 2e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2.5e-3, 0.01]
    init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

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


# =================================================================
# ----------------------- MANUAL K-FOLD ---------------------------
# =================================================================

# NOTE: This section is seldom used and is not updated for the model construction with multiple DNN input models
# Note: Section for if the model is being run/constructed by neither optimization is used

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

# =================================================================
# ------------------- GENERAL RUN CONDITION -----------------------
# =================================================================

# This is the general run section that is most commonly utilized

elif ModelRun == 'Yes' and Model_Calib == 'No' and K_Fold == 'No':
    save_dir = 'Saved_Models_NoFold/'
    # New Section for Dividing the Simulation Dataset into two MLPs that concatenate into a DNN

    # Evaluation Metrics
    '''print('New Training 1 Set Size Cols:', x_train1.shape[1])
    print('New Training 1 Set Size Rows:', x_train1.shape[0])
    print('New Training 2 Set Size Cols:', x_train2.shape[1])
    print('New Training 2 Set Size Rows:', x_train2.shape[0])

    print('X Input for Old Training Data:', x_train[1,])
    print('X Input for New Training 1 Data:', x_train1[1, ])
    print('X Input for New Training 2 Data:', x_train2[1, ])'''

    # Vector for First Model
    MLP_vector1 = Neuron_Layer(MLP_Size, MLP_layers)

    # Vector for Second Model
    MLP_vector2 = Neuron_Layer(MLP_Size, MLP_layers)

    # Input model for the Geometry
    model1 = DNNModel(x_train1.shape[1], MLP_vector1, MLP_Size)
    print(model1.summary())

    # Input model for the material properties
    model2 = DNNModel(x_train2.shape[1], MLP_vector2, MLP_Size)
    print(model2.summary())

    combinedMLP = keras.layers.concatenate([model1.output, model2.output])

    # Build new vector for the DNN
    Neuron_Vec = Neuron_Layer(N_Neuron, N_Layer)
    print('Length of Neuron Vector:', len(Neuron_Vec))
    print('Neuron Vector', Neuron_Vec)

    # DNN Generation for Analysis
    for kk in range(1, len(Neuron_Vec)):
        if kk == 1:
            x = combinedMLP
        if kk == 2:
            x = keras.layers.Dense(MLP_Size*2, kernel_initializer='normal', activation=tf.nn.relu6,
                                   kernel_regularizer=keras.regularizers.l2(1e-8))(x)
        elif kk > 2:
            x = keras.layers.Dense(Neuron_Vec[kk], kernel_initializer='normal', activation=tf.nn.relu6,
                               kernel_regularizer=keras.regularizers.l2(1e-8))(x)  # Previously used 'relu, no
    DNN_Output = keras.layers.Dense(y_train.shape[1], kernel_initializer='normal', name="Output_layer")(x)

    model = tf.keras.Model(inputs=[model1.input, model2.input], outputs=DNN_Output)

    # Establish Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + model_save + '.h5',
                                                    montior='mae', verbose=1,
                                                    save_best_only=True, mode='min')
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, verbose=0, mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, verbose=1,
                                                          min_delta=0.5e-4,
                                                          mode='min')

    callbacks_list = [checkpoint, earlyStopping, reduce_lr_loss]

    # Compile Model
    opt = keras.optimizers.Adam(learning_rate=1e-3)
    metrics = ['mae']

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=metrics)


    # Fitting Results
    results = model.fit([x_train1, x_train2], y_train, epochs=epochnum, verbose=2, callbacks=callbacks_list,
                        validation_data=([x_val1, x_val2], y_val))

    # Load Best Weights
    model.load_weights(save_dir + model_save + '.h5')

    # Save History of File
    H_fname = 'History/results.pkl'
    with open(H_fname, 'wb') as f:
        pickle.dump(results.history, f)

    History = results.history

    # Save Final Model
    model.save(model_filename)

# =================================================================
# --------------------- "NO RUN" CONDITION ------------------------
# =================================================================

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


# =======================================================================================================
# ============================= SECTION: SHOW MODEL RESULTS AND  ========================================
# =======================================================================================================

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

# =======================================================================================================
# ====================== SECTION: EVALUATE AND PREDICT USING UNSEEN DATA ================================
# =======================================================================================================

# In this section we predict/run for the unseen datasets: we evaluate/predict on the test data, griddata,
# and library data

# =================================================================
# ----------------------- PLOT HISTORY ----------------------------
# =================================================================

#  Title ---------- Evaluate Accuracy from Optimizations / Sweeps / Model(s) ----------------------------
# Plot History of Model Fitting/Training
plot_hist(History, logscale=0)
plot_hist(History, logscale=1)

# Save the Accuracy of the Train/Validation Accuracy
train_loss = History['loss']
val_loss = History['val_loss']

# Show the Model Architecture
print('\n====== MODEL SUMMARY ======')

# Save Model Summary to a file with trainable parameters
with open('Model Summary\model_V7_Ni%dFD_summary.txt' % Iter_number, 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Print/show model config that was saved to file previously
print(model.summary())
model.get_config()


# =================================================================
# ----------------------- TEST DATASET ----------------------------
# =================================================================

print('\n====== PREDICTION and EVALUATION METRICS: ====== \n')

# Accuracy of Test Dataset (from the split earlier)
mae_loss, mae_acc = model.evaluate([x_test1, x_test2], y_test, verbose=0)
print('MAE of the Model (Test Dataset)       :', mae_acc)

# Test Accuracy of Prediction Test Dataset compared to true results
test_pred = model.predict([x_test1, x_test2], verbose=0)

# This section saves to a mat file the actual test data (both x and y) and then the y predicted data with the accuracy
# value

mat_ids_1 = dict(
    test_pred=test_pred,
    XTest=x_test,
    YTest=y_test,
    mae_acc=mae_acc,
    mae_loss=mae_loss,
    History=History,
    train_loss = train_loss,
    val_loss = val_loss
)

filename_mat3 = 'Prediction Data\model_V7_Ni%dFD_TestDataset_Prediction.mat' % Iter_number
savemat(filename_mat3, mat_ids_1)


# =================================================================
# ------------------- LOAD UNSEEN DATASETS ------------------------
# =================================================================

# ------- Load Unseen Datasets -------
unseenmats = loadmat('Data/Unseen Datasets - 8 Input 2 Output - 20220208.mat',
                     squeeze_me=True,
                     struct_as_record=False)

# =================================================================
# ------- GRID DATA --- LARGE PREDICTION SECTION ------------------
# =================================================================


# ------- Predicting Large Amounts of GridData -------
Unseen_MatGrid = unseenmats['Grid_Input']
Unseen_Coords = unseenmats['GridCoords_Expanded']

# Initialize Timing Function
tic2 = time.perf_counter()

# Make Prediction for GridData
Unseen_GridPredict = model.predict(Unseen_MatGrid, verbose=0)

toc2 = time.perf_counter()

# Save Grid Data predictions to a MAT file
mat_ids = dict(
    Unseen_GridPredict=Unseen_GridPredict,
    Unseen_GridCoords=Unseen_Coords,
    Unseen_MatGrid=Unseen_MatGrid)

# Save File
filename_unseen = 'Prediction Data\model_V7_Ni%dFD_UnseenGrid_Prediction.mat' % Iter_number
savemat(filename_unseen, mat_ids)

# Print timing from the Grid Predict
print(f"Unseen Gridpoint Predictions occured for {len(Unseen_MatGrid):4.0f} grid points in {toc2 - tic2:0.4f} seconds")


# =================================================================
# --------------- MATERIAL LIBRARY PREDICTIONS --------------------
# =================================================================

# ------- Predicting for Included FDTD Simulations -------
# Unseen Predictions for Pre-existing simulation (FDTD) data of materials unused in training - 100+ sims per material
Multi_Input = unseenmats['Multi_Input']
Multi_Output = unseenmats['Multi_Output']

# Segment out inputs for multi DNN input
Multi_InputA = Multi_Input[:, 0:4]
Multi_InputB = Multi_Input[:, 4:8]

# Make Model Predictions
Unseen_Sim_Pred = model.predict([Multi_InputA, Multi_InputB], verbose=0)
unseen_loss, unseen_acc = model.evaluate([Multi_InputA, Multi_InputB], Multi_Output, verbose=0)

print('MAE of the Unseen Mat FDTD Dataset  :', unseen_acc)


# Wrap Files
mat_ids = dict(
    Unseen_Input=Multi_Input,
    Unseen_Output=Multi_Output,
    Unseen_Sim_Pred=Unseen_Sim_Pred,
    Unseen_loss=unseen_loss,
    unseen_acc=unseen_acc)

filename_unseen = 'Prediction Data\model_V7_Ni%dFD_UnseenFDTD_Prediction.mat' % Iter_number
savemat(filename_unseen, mat_ids)

# =================================================================
# ------------ UNSEEN (Ti/Al2O3) LIBRARY PREDICTIONS --------------
# =================================================================

# ----- Unseen Prediction Dataset -----
# Predictions from Unseen Data - Ti, Al2O3

Unseen_Input = unseenmats['Unseen_Input']
Unseen_Output = unseenmats['Unseen_Output']
Unseen_Input2 = unseenmats['Unseen_Input2']
Unseen_Output2 = unseenmats['Unseen_Output2']

Unseen_InputA = Unseen_Input[:, 0:4]
Unseen_InputB = Unseen_Input[:, 4:8]

Unseen_Input2A = Unseen_Input2[:, 0:4]
Unseen_Input2B = Unseen_Input2[:, 4:8]

USPred = model.predict([Unseen_InputA, Unseen_InputB], verbose=0)
US_loss, US_acc = model.evaluate([Unseen_InputA, Unseen_InputB], Unseen_Output, verbose=0)

USPred2 = model.predict([Unseen_Input2A, Unseen_Input2B], verbose=0)
US_loss2, US_acc2 = model.evaluate([Unseen_Input2A, Unseen_Input2B], Unseen_Output2, verbose=0)

print('MAE of the Ti Unseen Prediction Dataset:', US_acc)
print('MAE of the Al2O3 Prediction Dataset    :', US_acc2)

# Export Data to Matlab File for Post-Processing -- Included in this: Standardized, Grid, and Unseen Predictions
# Data export that includes coordinates/grid

mat_ids = dict(USPred=USPred,
               US_loss=US_loss,
               US_acc=US_acc,
               US_out=Unseen_Output,
               USPred2=USPred2,
               US_loss2=US_loss2,
               US_acc2=US_acc2,
               US_out2=Unseen_Output2)

# Save the File
filename_mat2 = 'Prediction Data\model_V7_Ni%dFD_Prediction.mat' % Iter_number
savemat(filename_mat2, mat_ids)

# Finally, show the plots generated
plt.show()


