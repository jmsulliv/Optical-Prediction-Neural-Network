# Optical-Prediction-Neural-Network
Optical Data Prediction Tool
# You need to add a readme to describe your project 

Optical predictions (Emissivity and Reflectivity) based on inputs from finite difference time domain (FDTD) simulations. The inputs of the neural network have a geometric component (X, Z, substrate thickness, and Aspect Ratio (AR)), a component that depends on the plane wave source used in the simulation (linearly spaced wavelength points from a min and max simulation injection wavelength), and material properties defined by the complex refractive index (n and k). 

Excel Datafiles show an annotated version of the data, with X, Z, AR, t_sub, the linearly spaced wavelength points, and n/k values that are linearly spaced corresponding to the wavelength point (i.e, if wavelength point #2 is at 1 um, then n/k #2 will be material data at 1 um as well). All data shown for the inputs is already normalized between 0 and 1. 

The .mat files included also have the normalized data but without the annotations. Calls to this data are done in the .py files already, so minimial work is required if you want to build and explore a new model. 

The Neural Network .py file is comprehensive code that includes optimizaiton methods for the neural network, ways to run and build new neural networks for the dataset. The preincluded model (Iter 15, or Model_V3_Ni15FD) is an already optimized and generated model that can be used to process new data or to reprocess old data. To load and use this dataset, simply make sure that the "Model Run" option is set to "No" and that the "Iter Number" is set to 15. If you do want to run new models, simply change the Iter Number and set "Model Run" to "Yes". Hyperparameter optimizaiton is also included and can be turned on, but this is not recommended due to the time involved. 

The CNN File takes in a 256 x 256 image of the 10 um x 10 um simulation domain and makes predictions based on the binary image input for what the simulation output should be. The included .mat file has a 256 x 256 image for each geometry simulated as well as the emissivity/reflectivity outputs. This file has a similar execution style to the Neural Network .py file, simply change the "Model Run", "HyperP" and other parameters according to what you want to do. 
