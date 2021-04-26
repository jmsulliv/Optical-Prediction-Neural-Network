# Optical-Prediction-Neural-Network
Optical Data Prediction Tool for micropyramid style surface structures, with a neural network predicting the emissivity and reflectivity of the domain. 
# General Description

Optical predictions (Emissivity and Reflectivity) based on inputs from finite difference time domain (FDTD) simulations. The inputs of the neural network have a geometric component (X, Z, substrate thickness, and Aspect Ratio (AR)), a component that depends on the plane wave source used in the simulation (linearly spaced wavelength points from a min and max simulation injection wavelength), and material properties defined by the complex refractive index (n and k). 

# Background -- Simulations
FDTD Simulations are Finite Element Model (FEM) style simulations, where a mesh is applied over an area and equations are solved over the domain. In this case, the solutions are of Maxwell's equations for electromagnetic radiation, and the solutions to these equations over the mesh allow us to compute optical properties for the simulated structure/geometry such as the reflectivity, transmissivity, and the emissivity. For our simulations, we simulate pyramids on a substrate in 2-D (i.e, triangles on a rectangular substrate) with periodic boundary conditions. This simulation setup predicts the optical properties for a triangular unit cell that infinitely extends along the x-axis in both directions (the y-axis, or 3rd dimension, is not considered for these simulations). The inputs we use are as follows: X span of the pyramid, Z span of the pyramid, thickness of the underlying substrate, and the wavelength (min/max) of the injected wave source into the simulation. 

For all material simulations, X/Z spans are selected from a randomly generated matrix of predefined values that range from a minimum value of ~0 to maximum of ~10 microns. 

The substrate thickness and wavelength properties depend on the selected material. For metals, the max/min values for the substrate thickness are 5/1 microns respectively. For non-metals, it is 100/5. Some select materials we use 40/1. These are somewhat arbitrarily defined, but generally rely on the material data as defined by the complex refractive index for the material and a knowledge of what materials are transmissive. Metals have minimal transmission due to a high extinction coefficient (k) value that limits tranmission thorugh the medium. Thus, simuating a thick substrate (> 5 um) would be a waste of simulation time as most metals need much less than 1 um to have a tranmission that is ~ 0 throughout the electromagnetic spectrum. Some metals/non-metals are more transmissive, so we define the max/min to capture the full range of tranmissivity behavior for the given material. Some materials do not require a large range substrate thickness range to capture the full range of behavior. The precise value for the substrate thickness is a singular value for a given simulation, and these values are selected from a randomly generated matrix of values constrained by the min/max for a given material. 

This logic is also true for the wavelength range. We chose minimum and maxium values based on the expected behavior of the material. For transmissive materials such as PDMS (a polymer) simulating in the visible light spectrum (~300 - 800 nm) is useless as the material is transmissive and the added range only serves to increase simulation time. For each material, we have a predefined min/max wavelength starting/ending point. From there, the simulation creates a linearly spaced vector of wavelengths to be simulated over. 

The simulation output (emissivity, reflectivity) match the wavelength points simulated one-for-one. From a mathmatical prospective, we are asking the simulation to simulate the equations across the mesh at particular frequencies, and each frequency point has a solution for the power transmitted through the domain and reflected from the domain (tranmission/reflection respectively). According to Kirchhoff's Law, The emission is the difference between the reflectivity and transmission. That is, the power that is not tranmitted or reflected is the emitted power. These values occur from 0 - 1 (ratios of input to output), and are defined as the emissivity, reflectivity, and transmissivity of the domain. More details can be found here: 

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

The .mat files included also have the normalized data but without the annotations. Calls to this data are done in the .py files already, so minimial work is required if you want to build and explore a new model. 

# Notes on the Neural Network Files
The Neural Network .py file is comprehensive code that includes optimizaiton methods for the neural network, ways to run and build new neural networks for the dataset. The preincluded model (Iter 15, or Model_V3_Ni15FD) is an already optimized and generated model that can be used to process new data or to reprocess old data. To load and use this dataset, simply make sure that the "Model Run" option is set to "No" and that the "Iter Number" is set to 15. If you do want to run new models, simply change the Iter Number and set "Model Run" to "Yes". Hyperparameter optimizaiton is also included and can be turned on, but this is not recommended due to the time involved. 

The CNN File takes in a 256 x 256 image of the 10 um x 10 um simulation domain and makes predictions based on the binary image input for what the simulation output should be. The included .mat file has a 256 x 256 image for each geometry simulated as well as the emissivity/reflectivity outputs. This file has a similar execution style to the Neural Network .py file, simply change the "Model Run", "HyperP" and other parameters according to what you want to do. 
