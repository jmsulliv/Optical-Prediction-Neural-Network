import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from contextlib import redirect_stdout
from tensorflow.keras import layers

# data = loadmat('Nickel Data Restructured w AR (600 Points)- 20200713.mat')
# data = loadmat('Nickel Data Restructured w AR (600 Points) - 5 Increment - 20200722.mat')
# data = loadmat('Nickel Data Restructured w AR FDTD Data (457 Points)- 20200803.mat')
data = loadmat('Nickel Data Restructured w AR FDTD Data (600 Points, 40 Output)- 20200806.mat')

'''print(type(data))
print(data.keys())'''

X = data['Input']
Y = data['Output']
# Y = data['T']

X = X.T
Y = Y.T

size = 510

x_train, x_test = X[:size], X[size:]
y_train, y_test = Y[:size], Y[size:]

'''print(x_test[1])
print(x_train[1])'''  # Test the import of the data
print(y_train[1])

print(x_train.shape)  # Shape of the input vector
print(y_train.shape)  # Shape of the output vector

"""model = keras.Sequential([
    keras.Input(shape=(103,)),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(8, activation="softmax"),
    keras.layers.Dense(100, activation="sigmoid")
])"""

"""input_data = keras.layers.Input(shape=(73,))
hidden1 = keras.layers.Dense(30, activation="softmax")(input_data)
hidden2 = keras.layers.Dense(15, activation="softmax")(hidden1)
hidden3 = keras.layers.Dense(15, activation="softmax")(hidden2)
hidden4 = keras.layers.Dense(15, activation="softmax")(hidden3)
hidden5 = keras.layers.Dense(15, activation="softmax")(hidden4)
hidden6 = keras.layers.Dense(15, activation="softmax")(hidden5)
hidden7 = keras.layers.Dense(15, activation="softmax")(hidden6)
hidden8 = keras.layers.Dense(30, activation="softmax")(hidden7)
output = keras.layers.Dense(70, activation='sigmoid')(hidden8)

model = keras.models.Model(inputs=[input_data], outputs=[output])
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])"""

'''input_data = keras.layers.Input(shape=(102,))
hidden1 = keras.layers.Dense(34, activation="softmax")(input_data)
output = keras.layers.Dense(100, activation='sigmoid')(hidden1)

model = keras.models.Model(inputs=[input_data], outputs=[output])

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])'''

def MultiLayerSequential(input_dim,neurons,outputs,metrics='mse'):
    '''
        Define a model 
    '''
    model = keras.Sequential()    
    model.add(layers.Dense(neurons[0], input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    for i in range(1,len(neurons)):
        model.add(layers.Dense(neurons[i], kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(outputs, kernel_initializer='normal')) 
    opt = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='mean_squared_error', optimizer = opt, metrics=metrics)
    return model

# input_data = keras.layers.Input(shape=[43])
# hidden1 = keras.layers.Dense(50, activation="softmax")(input_data)
# hidden2 = keras.layers.Dense(30, activation="softmax")(hidden1)
# hidden3 = keras.layers.Dense(10, activation="softmax")(hidden2)
# hidden4 = keras.layers.Dense(10, activation="softmax")(hidden3)
# hidden5 = keras.layers.Dense(10, activation="softmax")(hidden4)
# hidden6 = keras.layers.Dense(10, activation="softmax")(hidden5)
# hidden7 = keras.layers.Dense(30, activation="softmax")(hidden6)
# hidden8 = keras.layers.Dense(50, activation="softmax")(hidden7)
# concat = keras.layers.concatenate([input_data, hidden8])
# output = keras.layers.Dense(40, activation="sigmoid")(concat)
# model = keras.models.Model(inputs=[input_data], outputs=[output])
# model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
# model.summary()
model = MultiLayerSequential(43,[128,128,128,128],40)

model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))
model.save("model-paht.Ni7FD")

# model = keras.models.load_model("model.Ni7FD")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("Tested Acc", test_acc)

'''print([x_train[7]])'''
prediction = model.predict([x_train])

test_number = 8

# Breakdown of Models - Model 15, 4 Increment Data - 200 data points
#                       Model 16, 4 increment data - 600 data points
#                       Model 14, 600 datapoints -  Full model
#                       Model 17, 5 increment data - 600 data points
#                       Model 18, 10 increment data - 600 data points
#                       Model 19, 25 increment data - 600 data points
#                       Model 20, 50 increment data- 600 data points
# Test #'s 23 and 12 are useful for the 600 point data set

# Break down of FDTD Models (model.Ni#FD)
#                       Model 1 - 300 datapoints,
#                       Model 2 - 457 Datapoints,
#                       Model 3 - 457 Datapoints, (7 layers, 100 input points) - con. model
#                       Model 4 - 600 Datapoints, (7 layers, 70 input points) - con. model
#                       Model 5 - 600 Datapoints, (7 layers, 70 input points) - sequential model
#                       Model 6 - 600 Datapoints, (7 layers, 60 input points) - con. model
#                       Model 7 - 600 Datapoints, (7 layers, 40 input points) - con. model

"""tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
)"""

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

print(x_test[test_number])
print(prediction[test_number])
print(y_test[test_number])

print(model.summary())
model.get_config()

x_test_p = x_test[test_number]
x_pyr = round(x_test_p[0] * 9.9, 3)
z_pyr = round(x_test_p[1] * 9.9, 3)

x_plot = x_test[test_number]
x_plot = (x_plot[3:])*(10 - 0.3) + 0.3


print(x_pyr)
print(z_pyr)

# Plotting and Specification of Plot Properties
'''hfont = {'fontname'  'Helvetica'}     # Font Specification'''

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': 'serif'})

'''tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)'''

# Plotting for Segmented Data
"""wl = np.linspace(0.3, 10, 80)
plt.plot(wl, y_test[test_number], linestyle='--', marker='o', color='b', label="FDTD Data")
plt.plot(wl, prediction[test_number], linestyle='--', marker='o', color='r', label="Predicted Data")
plt.title('X span = {0} \u03BCm, Z span = {1} \u03BCm' .format(x_pyr, z_pyr),fontsize=16)
plt.legend(loc='lower left')
plt.ylabel('Emissivity', fontsize=14)
plt.xlabel('Wavelength (\u03BCm)', fontsize=14)
plt.ylim([0, 1])
plt.xlim([0.3, 10])
plt.savefig('Prediction.png')
plt.show()"""

# Plotting for Weighted Wavelength Data
plt.plot(x_plot, y_test[test_number], linestyle='--', marker='o', color='b', label="FDTD Data")
plt.plot(x_plot, prediction[test_number], linestyle='--', marker='o', color='r', label="Predicted Data")
plt.title('X span = {0} \u03BCm, Z span = {1} \u03BCm' .format(x_pyr, z_pyr), fontsize=16)
plt.legend(loc='lower left')
plt.ylabel('Emissivity', fontsize=14)
plt.xlabel('Wavelength (\u03BCm)', fontsize=14)
plt.ylim([0, 1])
plt.xlim([0.3, 10])
plt.savefig('Prediction.png')
plt.show()

# Plotting for 100 Data point output
'''wl = np.linspace(0.3, 10, 100)
plt.plot(wl, y_test[test_number], color='b', label="FDTD Data")
plt.plot(wl, prediction[test_number], color='r', label="Predicted Data")
plt.title('X span = {0} \u03BCm, Z span = {1} \u03BCm' .format(x_pyr, z_pyr),fontsize=16)
plt.ylabel('Emissivity', fontsize=14)
plt.xlabel('Wavelength (\u03BCm)', fontsize=14)
plt.legend(loc='lower left')
plt.ylim([0, 1])
plt.xlim([0.3, 10])
plt.savefig('Prediction.png')
plt.show()'''

