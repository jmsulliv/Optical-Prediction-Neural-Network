# Optical-Prediction-Neural-Network
Optical Data Prediction Tool for micropyramid style surface structures, with a neural network predicting the emissivity and reflectivity of the domain. 
# General Description

Optical predictions (Emissivity and Reflectivity) based on inputs from finite difference time domain (FDTD) simulations. The inputs of the neural network have a geometric component (X, Z, substrate thickness, and Aspect Ratio (AR)), a component that depends on the plane wave source used in the simulation (linearly spaced wavelength points from a min and max simulation injection wavelength), and material properties defined by the complex refractive index (n and k). 

# Background -- Simulations
FDTD Simulations are Finite Element Model (FEM) style simulations, where a mesh is applied over an area and equations are solved over the domain. In this case, the solutions are of Maxwell's equations for electromagnetic radiation, and the solutions to these equations over the mesh allow us to compute optical properties for the simulated structure/geometry such as the reflectivity, transmissivity, and the emissivity. For our simulations, we simulate pyramids on a substrate in 2-D (i.e, triangles on a rectangular substrate) with periodic boundary conditions. This simulation setup predicts the optical properties for a triangular unit cell that infinitely extends along the x-axis in both directions (the y-axis, or 3rd dimension, is not considered for these simulations). The inputs we use are as follows: X span of the pyramid, Z span of the pyramid, thickness of the underlying substrate, and the wavelength (min/max) of the injected wave source into the simulation. 

For all material simulations, X/Z spans are selected from a randomly generated matrix of predefined values that range from a minimum value of ~0 to maximum of ~10 microns. 

The substrate thickness and wavelength properties depend on the selected material. For metals, the max/min values for the substrate thickness are 5/1 microns respectively. For non-metals, it is 100/5. Some select materials we use 40/1. These are somewhat arbitrarily defined, but generally rely on the material data as defined by the complex refractive index for the material and a knowledge of what materials are transmissive. Metals have minimal transmission due to a high extinction coefficient (k) value that limits tranmission thorugh the medium. Thus, simuating a thick substrate (> 5 um) would be a waste of simulation time as most metals need much less than 1 um to have a tranmission that is ~ 0 throughout the electromagnetic spectrum. Some metals/non-metals are more transmissive, so we define the max/min to capture the full range of tranmissivity behavior for the given material. Some materials do not require a large range substrate thickness range to capture the full range of behavior. The precise value for the substrate thickness is a singular value for a given simulation, and these values are selected from a randomly generated matrix of values constrained by the min/max for a given material. 

This logic is also true for the wavelength range. We chose minimum and maxium values based on the expected behavior of the material. For transmissive materials such as PDMS (a polymer) simulating in the visible light spectrum (~300 - 800 nm) is useless as the material is transmissive and the added range only serves to increase simulation time. For each material, we have a predefined min/max wavelength starting/ending point. From there, the simulation creates a linearly spaced vector of wavelengths to be simulated over. 

The simulation output (emissivity, reflectivity) match the wavelength points simulated one-for-one. From a mathmatical prospective, we are asking the simulation to simulate the equations across the mesh at particular frequencies, and each frequency point has a solution for the power transmitted through the domain and reflected from the domain (tranmission/reflection respectively). According to Kirchhoff's Law, The emission is the difference between the reflectivity and transmission. That is, the power that is not tranmitted or reflected is the emitted power. These values occur from 0 - 1 (ratios of input to output), and are defined as the emissivity, reflectivity, and transmissivity of the domain. More details on the process and simulation structure can be found here: 

https://support.lumerical.com/hc/en-us/articles/360042089573-Reflection-and-transmission-calculations-using-a-planewave
https://support.lumerical.com/hc/en-us/articles/360042706493-Thermal-emission-from-a-micro-hole-array
https://doi.org/10.1364/OE.14.008785

At each frequency point of the solution, the material properties have to be considered for the simulation to work. For optical simulations, the permittivity/refractive index are how materials are differentiated from one another and what the simualtion runs on. The simulation draws from a library of material data; we add to the material library for materials that do not exist or require refinement for our purposes (such as the wavelengths we want to simulate are not included in the library data). The data we use and import is based on literature values of n,k (refractive index) values from experimental measurements. For each frequency point of solution there is a matching n and k value used in the simulation. It should be noted that a material model is constructed (i.e, a curve-fit) based on existing literature values for frequency/wavelength points that do not match the literature values precisely. 

Examples of n,k complex refractive index data that we draw upon can be found here for several example materials: 
Silver:  https://refractiveindex.info/?shelf=main&book=Ag&page=Yang
Aluminum: https://refractiveindex.info/?shelf=main&book=Al&page=Ordal

# Understanding the Inputs in the Datafiles -- Neural Network Input
In total, there are 304 inputs included for each simulation datafile. Those are the independent variables: X, Z, and t_sub. These are the randomly generated values discussed previously that were used in the simulation. Each simulation has a single X, Z, and t_sub. Each simulation has an output of emissivity/reflectivity based on these input parameters. The dependent parameter is the aspect ratio (AR) which is defined in this case as Z/X. This is an important geometric parameter that captures much of the  behavior of the micropyramids' optical property output. 

The wavelength points are generated via a linspace (linearly spaced) vector as defined by the minimum and maximum wavelength. The importance of this vector is shown in the simulation, but as the outputs match one to one, we include the wavelength vector in the neural network inputs. The simulations use 100 datapoints linearly spaced from the min/max wavelength point. We use the exact same vector as an input into our neural network. 

Accordingly, we also match the material properties to the wavelength vector. This means that as our wavelength vector is 100 points linearly spaced, we find the material n/k (complex refractive index) values that match the given wavelength points. So, if lambda #1 (wavelength) = 0.3 um, then n/k #1 are the n/k values for the material at 0.3 um, if lambda #2 = 0.5 um, then n/k values for the material at 0.5 um, and so on until the maximum (lambda #100) is reached. For our n,k values we use material data from literature and  build a curve-fit model for the data as we seldom match the exact wavelength point that the n,k values were measured at. 

The ordering is as follows for the input: <br />
**Neuron 1 (X) <br />
Neuron 2 (Z) <br />
Neuron 3 (AR) <br />
Neuron 4 (t_sub) <br />
Neuron 5-104 (linearly spaced wavelength points with min at Neuron 5 and Max at Neuron 104) <br />
Neuron 105-204 (linearly spaced n refractive index value, one-to-one matching the material data at the wavelengths in Neurons 5-104) <br />
Neuron 205-304 (linearly spaced k extinction coefficient value, one-to-one matching the material data at the wavelengths in Neurons 5-104) <br />**

The outputs of these simualations are emissivity and reflectivity. These points -- like n and k -- are a linearly spaced vector that one-to-one match the wavelength point used in simulation. 

Output Neurons: <br />
**Neuron 1-100 (emissivity) <br />
Neuron 101-200 (reflectivity) <br />**

We do not include tranmissivity in the output files as that property can be computed from the other two properties at each wavelength point as a function of emissivity and reflectivty, assuming Kirchoff's law holds. 

# Included Datafiles
The included Excel .csv Datafiles show an annotated version of the data, with X, Z, AR, t_sub, the linearly spaced wavelength points, and n/k values that are linearly spaced corresponding to the wavelength point (i.e, if wavelength point #2 is at 1 um, then n/k #2 will be material data at 1 um as well). All data shown for the inputs is already normalized between 0 and 1 in the "Input" and "Output" .csv files, and unnormalized or "raw" data is included in the "Raw_Input" and "Raw_Output" excel file. 

Normalization occurs with the following equation: Z_norm = (Z - Z_min)/(Z_max - Z_min)

It should be noted that for the "normalized" datasets, the normalization occurs per grouping. X/Z are normalized together as the "geometric" properties, and AR, t_sub, wavelength, n, and k are all normalized separately from one another. To be more specific, Neuron 1/2 are normalized by their min/max values, Neuron 3 is normalized by its min/max, Neuron 4, Neuron 5-104, Neuron 105-204, and Neuron 205-304 follow the same group normalization strategy. For example, we find the mimimum and maximum k (extinction coefficient) across all the materials/simulations compiled in neurons 205-304, and then normalize across all the datasets using these min/max values for the "k" group of neuron 205-304. 

The .mat files included also have the normalized data but without the annotations. These files are put into structures and are classified by the aspect ratio used to limit the dataset. As the X/Z inputs are from a randomized matrix of values, there will be combinations that yield extremely high aspect ratio structures (several are AR > 2000). These points can lead to misleading results, so we eliminate the outliers by eliminating datapoints based on a certain aspect ratio. Over time, we have settled on AR < 100 being an appropriate threshold, but other ARs can also be used as a cutoff if so desired. All simulation results that have Z/X inputs exceeding 100 are removed from the larger dataset. For the 31775 dataset (31775 simulations of different materials combined together) around 300 points were eliminated for having an AR > 100. For the sake of ease of import into the system, each mat file shows the number of simulations included (i.e, "31775 datapoints") and the day the file was generated in year-month-day (i.e, "20210426"). The file is organized by structures that contain substructures for each apsect ratio that was used to limit the overall dataset. 

Each .mat file of this format has the following structures:  <br />

***Simulation Dataset Input/Ouptut*** <br />
**ARInput**  (Combination of all normalized datasets, 304 Inputs for each simulation) <br />
**AROutput**  (Output of each simulation corresponding to the input of geometry, material, wavelength) <br />

***Simulation Input Normalization Conditions*** <br />
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

Normalization Condition Factors:<br />
(1)/(2) Min/Max wavelength values<br />
(3)/(4) Min/Max X/Z (geometric) values <br />
(5)/(6) Min/Max Aspect Ratio in the dataset <br />
(7)/(8) Min/Max Substrate Thickness <br />
(9)/(10) Min/Max n values across all materials used <br />
(11)/(12) Min/Max k values across all materials used <br />

These condition values can be used to denormalize their respective neuron groupings following the linear equation: Z = Z_norm* (max_cond - min_cond) + min_cond

An additional .mat file is included that is automatically run in the Neural Network file and database which follows the format "Normalized Unseen Data - 20210426.mat". This dataset includes simulations for materials not included in the training dataset (ARInput/AROutput). This data has the standard 304 Input/200 Output format and is normalized using the condition factors in AR_LinN_100. 

The matrices contained in this .mat file are:  <br />

**'Unseen_Input'** (Titanium Micropyramid simulation dataset input, 304 Input Neurons)  <br />
**'Unseen_Output'** (Titanium Micropyramid simulation dataset output, 200 output emissivity/reflectivity points)  <br />
**'Unseen_Input2'** (Alumina Micropyramid simulation dataset input, 304 Input Neurons)  <br />
**'Unseen_Output2'** (Alumina Micropyramid simulation dataset input, 304 Input Neurons)  <br />

Please note that if you want more information on how to call/use this in python, the matfile and all of the associated vector calls are already done in the .py files given here, so minimial work is required if you want to build and explore a new model using the mat files. 

# Notes on the Neural Network Files
The neural network is tasked with taking all of the available inputs that represent the simulations (geometry, wavelength, and material properties) and replicate their outputs (emissivity and reflectivity) via a deep neural network approach or a convolutional neural network (or a combination of the two). The goal of this being to predict the behavior of materials and geometries not included in the training process. The files included here allow you to either make your own new network or use the one that has already been trained in order to predict for new data that follows the style as shown above. 

The deep-neural network approach is contained in the The Neural Network .py file and is a comprehensive code file that includes optimizaiton methods for the neural network, ways to run and build new neural networks for the dataset. The preincluded model (Iter 15, or Model_V3_Ni15FD) is an already optimized and generated model that can be used to process new data or to reprocess old data. To load and use this dataset, simply make sure that the "Model Run" option is set to "No" and that the "Iter Number" is set to 15. If you do want to run new models, simply change the Iter Number and set "Model Run" to "Yes". Hyperparameter optimizaiton is also included and can be turned on, but this is not recommended due to the time involved. 

The CNN File takes in a 256 x 256 image of the 10 um x 10 um simulation domain and makes predictions based on the binary image input for what the simulation output should be. The included .mat file has a 256 x 256 image for each geometry simulated as well as the emissivity/reflectivity outputs. This file has a similar execution style to the Neural Network .py file, simply change the "Model Run", "HyperP" and other parameters according to what you want to do. The CNN work is ongoing and less optimized, though the goal of this is to eventually demonstrate that we can take an image of the simulation domain and have a model that can predict what the optical properties of that simulation domain are. 
