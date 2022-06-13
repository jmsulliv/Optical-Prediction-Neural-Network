from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np


# Load Data for Preprocessing/Normalization
data = loadmat('Preprocessing Data/40500 Sims Preprocessing Data - 20211124.mat', squeeze_me=True, struct_as_record=False)

# G -- LinN normalization, (uniform)
# C -- LinN normalization, (semi-uniform)
# TS -- normal distribution, quantile
# n -- log10,
# k -- normal distribution, quantile
# ereal -- normal distribution, quantile
# eim -- normal distribution, quantile
# Output -- log10 distribution - tested but not used

# ---- only going to process TS, k, ereal, eim ----
# ---- export data to MATLAB for processing ------

# ============ SECTION 1: Normalization for Whole Dataset =============
# --------------- Go by variable ---------------------------

# TS (Substrate Thickness)
TS = data['TS_Total']
TSdata = np.reshape(TS, (-1, 1))

# Apply a transform/normalization on the TS data
quantile_transformer = QuantileTransformer(random_state=0,  output_distribution='uniform')
'''power_transformer = PowerTransformer(method='yeo-johnson')'''
TSDataOut = quantile_transformer.fit_transform(TSdata)
'''TSDataOut = power_transformer.fit_transform(TSdata)'''

# Experimented with yeo-johnson and power-transform methods, but the quantile transformation yields a much more
# consistent result, especially with the negative number(s) for the ereal value (ereal = n^2 - k^2)

# Apply a transform/normalization on the k data
k = data['k_Total']
kdata = np.reshape(k, (-1, 1))
kDataOut = quantile_transformer.fit_transform(kdata)


# Apply a transform/normalization on the ereal data
ereal = data['e_real_Total']
erealdata = np.reshape(ereal, (-1, 1))
erealDataOut = quantile_transformer.fit_transform(erealdata)


# Apply a transform/normalization on the eim data
eim = data['e_im_Total']
eimdata = np.reshape(eim, (-1, 1))
eimDataOut = quantile_transformer.fit_transform(eimdata)


# Apply transform/normalization on the Output -- R
R = data['OutputR']
Rdata = np.reshape(R, (-1, 1))
ROut = quantile_transformer.fit_transform(Rdata)


# Apply transform/normalization on the Output -- T
T = data['OutputT']
Tdata = np.reshape(T, (-1, 1))
TOut = quantile_transformer.fit_transform(Tdata)


# Test Output
print(TSdata.shape)

# Group these for post-processing and export into MATLAB
mat_ids = dict(TS_Data=TSDataOut,
               k_Data=kDataOut,
               erealData=erealDataOut,
               eimData=eimDataOut,
               TS_Check=TSdata,
               k_Check=kdata,
               ereal_Check=erealdata,
               eim_Check=eimdata,
               RCheck=Rdata,
               TCheck=Tdata,
               TOut=TOut,
               ROut=ROut)

# Export MATLAB file
filename_mat = 'Preprocessing Data\PreprocessDataOut.mat'
savemat(filename_mat, mat_ids)


# ============ SECTION 2: Normalization by Material for Classification =============
# # --------------- Go by variable ---------------------------

newdata = loadmat('Preprocessing Data/MaterialdataIn', squeeze_me=True, struct_as_record=False)

# Apply a transform/normalization on the TS data
TS = newdata['TS_values']

TSdata_new = np.reshape(TS, (-1, 1))
TSDataOut = quantile_transformer.fit(TSdata).transform(TSdata_new)

# Apply Normalization to the k data
k = newdata['k_values']

nummat = k.shape[1]
kdata_new = np.reshape(k, (-1, 1))
kDataOut = quantile_transformer.fit(kdata).transform(kdata_new)

# Apply a transform/normalization on the ereal data
ereal = newdata['e_real']
erealdata_new = np.reshape(ereal, (-1, 1))
erealDataOut = quantile_transformer.fit(erealdata).transform(erealdata_new)

'''erealDataOut = power_transformer.fit_transform(erealdata)'''

# Apply a transform/normalization on the eim data
eim = newdata['e_im']
eimdata_new = np.reshape(eim, (-1, 1))
eimDataOut = quantile_transformer.fit(eimdata).transform(eimdata_new)

print(kdata)
print(kDataOut)

# Reshape Data to the number of materials in the dataset
kDataOut = np.reshape(kDataOut, (-1, nummat))
erealDataOut = np.reshape(erealDataOut, (-1, nummat))
eimDataOut = np.reshape(eimDataOut, (-1, nummat))

# Check Inputs/Outputs
print('Confirm Same Size As Input: Columns', kDataOut.shape[1])
print('Confirm Same Size As Input: Rows', kDataOut.shape[0])

# Package data for export into MATLAB processing
mat_ids = dict(norm_k=kDataOut,
               k_check=k,
               norm_ereal=erealDataOut,
               ereal_check=ereal,
               norm_eim=eimDataOut,
               eim_check=eim,
               norm_TS=TSDataOut)

# Export MATLAB File
filename_mat = 'Preprocessing Data\MaterialDataOut.mat'
savemat(filename_mat, mat_ids)
