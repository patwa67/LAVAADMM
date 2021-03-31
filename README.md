# LAVAADMM contains Julia code for the proximal ADMM version of the LAVA regularizer presented in Waldmann (2021; submitted to Bioinformatics).
The code reads data from the QTLMAS2010ny012.csv file available in the AUTALASSO directory (which needs to be downloaded to your woking directory and extracted). The y-variable (phenotype) is in the first column and the x-variables (SNPs; coded 0,1,2) are in the following columns (comma separated). The data is partitioned into training (generation 1-4) and test data (generation 5). The output is the optimal regularization parameters lambda1 and lambda2 for the test MSE minimizer.
