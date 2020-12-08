Prediction techniques implementation: (Pavan)
The following three prediction techniques are implemented using MATLAB.
Vectorized Linear Regression - file name: vectorized_linear_ridge_regression.m
Kernel Ridge Regression - file name: kernel_ridge_regression.m
Fully Connected Neural Networks - file name: FCNN_MATLAB.m
In addition a function to vectorize the lower triangular matrices in the data is added as well.
That function file is named as vectorize_data.m

The input for all three techniques is in the folder 'Data'. Also, the inputs are in the form of mat files. 
While loading mat files into MATLAB, the name of the variable in the mat file is important. 
Since we used various embedding techniques, these variable names differ and could not be coded directly. 

I have also implemented Kernel regression using python, but there were issues with consistency and matrix inversion.
So, I switched to MATLAB. The python code is attached as well. 