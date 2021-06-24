sketch of readme

nowhere ready to be understandable yet

#**Optical Polarimetry**

In this project analysis of optical microscope with 
compensator (rotating wave plates) could be found. Multivariate model
of the optical microscope was realized using Mueller calculus. Inverse 
solution to the model represents optical properties of the
sample. 


##**Contents**

`Mueller Calculations`

In this section Mueller matrix for standard elements like `Wavepallet`,
`Linear Polarizer` and operators like `Rotation` and numerical operations
like `computing transfer matrix` for stoke vector could be found. 
Digital model of the setup could be changed to add additional manipulations
to the signal.

`Interpolation Methods`

In this section discrete measurements obtained from the instrument could be
approximated by various `Interpolation` methods. (_sinc interpolation_, _Polynomial
interpolations_, _etc_)

`Maximum Likelihood`

Yet to be realized... this section is necessary to represent the error bounds
for the inverse solution representing Mueller matrix. Due to systematic uncertainties
in the experimental setup and noise interpretation of the signal is the most
important part of the project.

`Fourier Transforms`

Harmonics of the signal 

`Optimizers`

Several optimizers were adapted from **scipy** library:

They can be classified as follwing:

**Basinhop:** Heuristic method for obtaining ...
