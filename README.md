# Optical-Prediction-Neural-Network
Optical Data Prediction Tool for micropyramid style surface structures, with a neural network predicting the emissivity and reflectivity of the domain. 
# General Description

Optical predictions (Emissivity and Reflectivity) based on inputs from finite difference time domain (FDTD) simulations. The inputs of the neural network have a geometric component (X, Z, substrate thickness), a component that depends on the plane wave source used in the simulation (linearly spaced wavelength points from a min and max simulation injection wavelength), and material properties defined by the complex refractive index (n and k). 

# Background -- Simulations
FDTD Simulations are Finite Element Model (FEM) style simulations, where a mesh is applied over an area and equations are solved over the domain. In this case, the solutions are of Maxwell's equations for electromagnetic radiation, and the solutions to these equations over the mesh allow us to compute optical properties for the simulated structure/geometry such as the reflectivity, transmissivity, and the emissivity. For our simulations, we simulate pyramids on a substrate in 2-D (i.e, triangles on a rectangular substrate) with periodic boundary conditions. This simulation setup predicts the optical properties for a triangular unit cell that infinitely extends along the x-axis in both directions (the y-axis, or 3rd dimension, is not considered for these simulations). The inputs we use are as follows: X span of the pyramid, Z span of the pyramid, thickness of the underlying substrate, and the wavelength (min/max) of the injected wave source into the simulation. 

For all material simulations, X/Z spans are selected from a randomly generated matrix of predefined values that range from a minimum value of ~0 to maximum of ~10 microns. 

The substrate thickness and wavelength properties depend on the selected material. For metals, the max/min values for the substrate thickness are 5/1 microns respectively. For non-metals, it is 100/5. Some select materials we use 40/1. These are somewhat arbitrarily defined, but generally rely on the material data as defined by the complex refractive index for the material and a knowledge of what materials are transmissive. Metals have minimal transmission due to a high extinction coefficient (k) value that limits tranmission thorugh the medium. Thus, simuating a thick substrate (> 5 um) would be a waste of simulation time as most metals need much less than 1 um to have a tranmission that is ~ 0 throughout the electromagnetic spectrum. Some metals/non-metals are more transmissive, so we define the max/min to capture the full range of tranmissivity behavior for the given material. Some materials do not require a large range substrate thickness range to capture the full range of behavior. The precise value for the substrate thickness is a singular value for a given simulation, and these values are selected from a randomly generated matrix of values constrained by the min/max for a given material. 

This logic is also true for the wavelength range. We chose minimum and maximum values based on the expected behavior of the material. For transmissive materials such as PDMS (a polymer) simulating in the visible light spectrum (~300 - 800 nm) is useless as the material is transmissive and the added range only serves to increase simulation time. For each material, we have a predefined min/max wavelength starting/ending point. From there, the simulation creates a linearly spaced vector of wavelengths to be simulated over. 

The simulation output (emissivity, reflectivity) match the wavelength points simulated one-for-one. From a mathmatical prospective, we are asking the simulation to simulate the equations across the mesh at particular frequencies, and each frequency point has a solution for the power transmitted through the domain and reflected from the domain (tranmission/reflection respectively). According to Kirchhoff's Law, The emission is the difference between the reflectivity and transmission. That is, the power that is not tranmitted or reflected is the emitted power. These values occur from 0 - 1 (ratios of input to output), and are defined as the emissivity, reflectivity, and transmissivity of the domain. More details on the process and simulation structure can be found here: 

https://support.lumerical.com/hc/en-us/articles/360042089573-Reflection-and-transmission-calculations-using-a-planewave
https://support.lumerical.com/hc/en-us/articles/360042706493-Thermal-emission-from-a-micro-hole-array
https://doi.org/10.1364/OE.14.008785

At each frequency point of the solution, the material properties have to be considered for the simulation to work. For optical simulations, the permittivity/refractive index are how materials are differentiated from one another and what the simulation runs on. The simulation draws from a library of material data; we add to the material library for materials that do not exist or require refinement for our purposes (such as the wavelengths we want to simulate are not included in the library data). The data we use and import is based on literature values of n,k (refractive index) values from experimental measurements. For each frequency point of solution there is a matching n and k value used in the simulation. It should be noted that a material model is constructed (i.e, a curve-fit) based on existing literature values for frequency/wavelength points that may not match the literature values precisely. 

Examples of n,k complex refractive index data that we draw upon can be found here for several example materials:  <br />
Silver:  https://refractiveindex.info/?shelf=main&book=Ag&page=Yang <br />
Aluminum: https://refractiveindex.info/?shelf=main&book=Al&page=Ordal <br />

# Understanding the Inputs in the Datafiles -- Neural Network Input
## Summary of Inputs/Outputs and Datafile setup
In total, there are 8 inputs put into the neural network. Three of those are the independent geometric variables: X, Z, and t_sub. These are the randomly generated values discussed previously that were used in the simulation. Each simulation has a single X, Z, and t_sub. Each simulation has an output of emissivity/reflectivity based on these input parameters. 

The fourth neural input is the wavelength. For our simulations, the wavelength points are generated via a linspace (linearly spaced) vector as defined by the minimum and maximum wavelength. The importance of this vector is shown in the simulation, but as the outputs match one to one, we include the wavelength vector in the neural network inputs. The simulations use 100 datapoints linearly spaced from the min/max wavelength point. We divide the larger simulation into 100 smaller network inputs to the neural network, such that each set of inputs is for a single frequency point (wavelength) solution. Thus, for each of our FDTD simulations, we have 100 neural network input sets. 

The fifth and six input parameters are n and k values of the complex refractive index. We match the material properties to the wavelength vector. This means that as our wavelength vector is 100 points linearly spaced, we find the material n/k (complex refractive index) values that match the given wavelength points. So, if lambda #1 (wavelength) = 0.3 um, then n/k #1 are the n/k values for the material at 0.3 um, if lambda #2 = 0.5 um, then n/k values for the material at 0.5 um, and so on until the maximum (lambda #100) is reached. For our n,k values we use material data from literature and  build a curve-fit model for the data as we seldom match the exact wavelength point that the n,k values were measured at. 

The seventh and eigth parameters are derived from the complex refractive index: that is, the real and imaginary relative permettivity values. We compute these values as E_real = n^2 - k^2 and E_im = 2*n*k. We have found that including these additional correlated parameters significantly strengthens the relationship between the material input and the output, allowing for our model to much more accurately predict libraries of materials that are very dissimilar from those used in the training of the network. 

## Input/Output Breakdown
![DNN Architecture](https://imgur.com/2TFVQDh)

The ordering is as follows for the input: <br />
**Neuron 1 (X) <br />
Neuron 2 (Z) <br />
Neuron 3 (t_sub) <br />
Neuron 4 (wavelength of the light) <br />
Neuron 5 (n value of complex refractive index matching the wavelength for the given material) <br />
Neuron 6 (k value of complex refractive index matching the wavelength for the given material) <br />
Neuron 7 (ereal (n^2 - k^2) value of complex refractive index matching the wavelength for the given material) <br />
Neuron 8 (eim (2*n*k) value of complex refractive index matching the wavelength for the given material) <br />**

The outputs of these simualations are transmissivity and reflectivity. These points -- like the material properties -- are specific to the wavelength. We compute the emissivity later using the reflectivity and transmissivity, in the same way we would use the power monitors to compute emissivity in the simulation. 
Output Neurons: <br />
**Neuron 1 (reflectviity) <br />
Neuron 2 (transmissivity) <br />**

We we compute emissivity as E = 1 - R - T, assuming Kirchhoff's law holds. 

# Included Datafiles
## .csv Datafiles
The included Excel .csv Datafiles show an annotated version of the translated FDTD data, with X, Z, t_sub, the wavelength, and the material properties aligned for input into the network. All data shown for the inputs is already normalized between 0 and 1 in the "Input" and "Output" .csv files, and unnormalized or "raw" data is included in the "Raw_Input" and "Raw_Output" excel file. Additionally, we "tag" each set with the material, specified in an additional column such that it is evident which material is which. 

These datasets are transposed from the original matlab matricies for more transparent viewing. Thus, all of the neurons are categorized in the .csv column headers whereas the neurons are the categorized in the rows in the .mat files. The .mat files are transposed in python using the .T function. 

## Normalization Notes
Normalization of each variable (or neuron row) depends on our previous experimentation and what we have determined to be the most effective based on the composition and distribution of our datasets. 

Normalization is broken down as follows: <br />
(X,Z)                  - Normalized together using a linear normalization (Znorm = (Z - Zmin)/(Z_max - Z_min)) <br />
(lambda - wavelength)  - Normalized  using a linear normalization <br />
(T_sub)                - Normalized using quantile regression <br />
(n - refractive index) - Normalized using log/linear normalization (z_norm = log10(z) as the first step)  <br />
(k - extinction coef.) - Normalized using quantile regression <br />
(ereal - real perm)    - Normalized using quantile regression <br />
(eim - im perm)        - Normalized using quantile regression <br />

It should be noted that all of these properties are normalized by "group", and the matricies are reconstructed after each column is normalized separately. 

## .mat Datafiles
### .mat File used for Model Evaluation, Training, Testing, and Predicting for Materials used in Model Training
The .mat files included also have the normalized data but without the annotations. These files are currently used as the inputs for our network, either for predictions or for training the model. 

Each .mat file has the following variables:  <br />

***Simulation Dataset Input/Ouptut*** <br />-- *"35500 Simulation Dataset - 8 Input 2 Output"* <br />
**Main_Input**  (Combination of all normalized datasets, 304 Inputs for each simulation) <br />
**Main_Output**  (Output of each simulation corresponding to the input of geometry, material, wavelength) <br />

***Simulation Input Normalization Conditions*** <br /> *Conditions - Not AR Normalized* <br />
**Cond** (Normalization Conditions, i.e, min/max values, as limited by the maximum aspect ratio)   <br /> 

***Grid Coordinates used for Predictions*** <br />
*The Grid Data is a grid of inputs from 0 to 10 microns in both the X and Z directions that attaches material information, wavelength information, etc., to allow the model to make predictions and fill the design space of a given material. The normal size of this is 10000 datapoints, as we go from 0.1 to 10 microns in intervals of 0.1, meaning a 100 x 100 grid of X/Z data with a fixed substrate thickness. The particular size and scope of this dataset does not matter, it is simply another way of predicting from the model and generating new data for a particular material*

**GridCoords** (X,Z values used in a 2 x Grid Size Matrix) <br /> 
**GridData**   (Full 304 Input values for each set of grid coordinates (X and Z)) <br /> 

***Static Prediction Dataset*** <br />
*All of these properties pertain to a set of Nickel simulations that are used to test the accuracy of the generated neural network. The inputs/outputs follow the same 304/200 configuration. This dataset is pre-normalized using the same condition values as the ARInput dataset. This dataset contains ~256 points before points are removed due to the maximum AR allowed*

**PredictCon** (Normalization Conditions used in the static prediction dataset) <br />
**PredictInput** (Static Prediction Input - 304 neurons) <br />
**PredictOutput** (Static Prediction Output - 304 neurons) <br />
**ActInp** (Actual Input -- unnormalized input)<br />

Each of these categories contains sets of "sub structures" that allow the datasets to use a different threshold for the maximum AR used. The appearance of these vectors is: 
(Category).AR_LinN_100. The category refers to one of the bolded categories described above, the LinN to the type of normalization (Linear) and the number (100) to the maximum aspect ratio allowed in the dataset. To minimize the size of the .mat file, we currently only have AR_LinN_100 as a subvector, but if a study requires a different AR we can include as many as we need to: calling them only requries changing the number from 100 to the other aspect ratio dataset generated. 

The .mat files include in their structure a "cond" vector which has the min/max values for the X/Z, AR, t_sub, and wavelength. 

### Normalization Factors
Contained in the "Cond.AR_LinN_100" (or whatever AR is used) vector in the mat file. 

These condition values can be used to denormalize their respective neuron groupings following the linear equation: Z = Z_norm* (max_cond - min_cond) + min_cond

Normalization Condition Factors:<br />
(1)/(2) Min/Max wavelength values<br />
(3)/(4) Min/Max X/Z (geometric) values <br />
(5)/(6) Min/Max Aspect Ratio in the dataset <br />
(7)/(8) Min/Max Substrate Thickness <br />
(9)/(10) Min/Max n values across all materials used <br />
(11)/(12) Min/Max k values across all materials used <br />

(1/2) are used to denormalize neuron inputs 5-104 (the wavelength vector), (3/4) are used to denormalize neuron inputs 1-2, (5/6) for neuron input 3, (7/8) for neuron input 4, (9/10) for neuron input 105-204, and (11/12) for neuron input 205-304. 

### .mat File for Predicting Optical Properties for Materials not Included in the Training Dataset
An additional .mat file is included that is automatically run in the Neural Network file and database which follows the format "Normalized Unseen Data - 20210426.mat". This dataset includes simulations for materials not included in the training dataset (ARInput/AROutput). This data has the standard 304 Input/200 Output format and is normalized using the condition factors in AR_LinN_100. 

The matrices contained in this .mat file are:  <br />

**'Unseen_Input'** (Titanium Micropyramid simulation dataset input, 304 Input Neurons)  <br />
**'Unseen_Output'** (Titanium Micropyramid simulation dataset output, 200 output emissivity/reflectivity points)  <br />
**'Unseen_Input2'** (Alumina Micropyramid simulation dataset input, 304 Input Neurons)  <br />
**'Unseen_Output2'** (Alumina Micropyramid simulation dataset input, 200 output emissivity/reflectivity points)  <br />

These datasets contain the actual simulation outputs and are organized just like the training data, but have not been included in the training process. Thus, we can use this dataset to evaluate how accurate our model is in the prediction of materials that are unseen by the neural network. Generally, we use the inputs to predict the output (model.predict the unseen inputs) and evaulate the inputs vs. the ouputs to determine the error (MAE) score. A lower score indicates that the neural network is accurately predicting optical properties for materials it has not been exposed to. 

Please note that if you want more information on how to call/use this in python, the matfile and all of the associated vector calls are already done in the .py files given here, so minimial work is required if you want to build and explore a new model using the mat files. 

# Notes on the Neural Network Files
The neural network is tasked with taking all of the available inputs that represent the simulations (geometry, wavelength, and material properties) and replicate their outputs (emissivity and reflectivity) via a deep neural network approach or a convolutional neural network (or a combination of the two). The goal of this being to predict the behavior of materials and geometries not included in the training process. The files included here allow you to either make your own new network or use the one that has already been trained in order to predict for new data that follows the style as shown above. 

The deep-neural network approach is contained in the The Neural Network .py file and is a comprehensive code file that includes optimizaiton methods for the neural network, ways to run and build new neural networks for the dataset. The preincluded model (Iter 15, or Model_V3_Ni15FD) is an already optimized and generated model that can be used to process new data or to reprocess old data. To load and use this dataset, simply make sure that the "Model Run" option is set to "No" and that the "Iter Number" is set to 15. If you do want to run new models, simply change the Iter Number and set "Model Run" to "Yes". Hyperparameter optimizaiton is also included and can be turned on, but this is not recommended due to the time involved. 

The CNN File takes in a 256 x 256 image of the 10 um x 10 um simulation domain and makes predictions based on the binary image input for what the simulation output should be. The included .mat file has a 256 x 256 image for each geometry simulated as well as the emissivity/reflectivity outputs. This file has a similar execution style to the Neural Network .py file, simply change the "Model Run", "HyperP" and other parameters according to what you want to do. The CNN work is ongoing and less optimized, though the goal of this is to eventually demonstrate that we can take an image of the simulation domain and have a model that can predict what the optical properties of that simulation domain are. 
