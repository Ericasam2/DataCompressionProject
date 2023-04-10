# DataCompressionProject
## Implementation
* download the reprositry
* run the **networked_opinion_manipulation.m** file

## Networked Opinion Control
The control input  ̃u can be estimated using a piece-wise sparse recovery algorithm.

## POMP
The Piecewise Orthogonal Matching Pursuit (POMP) algorithm uses the same framework as the regular Orthogonal Matching Pursuit (OMP) algorithm. POMP behaves like OMP whenever the current support in a piece si is lower than the predefined piecewise sparsity Ki of the block. The main difference arises when the limit Ki of the block is reached, the last index gets added and all other columns of the block get disabled.

## Results
### Convergence
![State over steps](https://github.com/Ericasam2/DataCompressionProject/blob/main/figures/state%20over%20steps.jpg)
### Entry limitation
![Entry over steps](https://github.com/Ericasam2/DataCompressionProject/blob/main/figures/entry%20over%20steps.jpg)
### Relation between sparsity and blocksize
![Relation between sparsity and blocksize](https://github.com/Ericasam2/DataCompressionProject/blob/main/figures/relation%20between%20sparsity%20and%20time%20steps.jpg)
### Reconstruction Error
![Reconstruction Error](https://github.com/Ericasam2/DataCompressionProject/blob/main/figures/sparsity%20and%20according%20estimation.jpg)
### Control Effort
![Control effor](https://github.com/Ericasam2/DataCompressionProject/blob/main/figures/control%20effort.jpg)
