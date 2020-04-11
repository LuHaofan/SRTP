# SRTP
This is the data/code repository of Our SRTP project
## Model specification
---
Using neural network to predict the shape factor p and conductivity
* nnmodelFP_v3: predict the value of p
	param: 
	V -- volume of the particle
	S -- Surface area of the particle
	Pro_1 -- area of orthographic projection 
	Pro_2 -- area of lateral projection
	Pro_3 -- area of upward projection
	AvgR -- average radius of the particle: calculated as the radius of the sphere of the same volume
* FindPscaler_v3: standardize the parameters. The param specification is the same as the nnmodelFP_v3

* nnmodel4f_predK: predict the value of conductivity
	param:
	Kp -- Conductivity of particle phase
	Km -- Conductivity of the matrix phase
	f -- volume fraction 
	p -- shape factor
* 4fscaler: standardize the parameters. The param specification is the same as the nnmodel4f_predK

## Plot specification
* Include the data plot and error plot
* The basic setting diagram: the shape of neural network and Finite Element model

## Data specification
* Include the test file and results.

## Document specification
* Include the midterm report, final report and starting report