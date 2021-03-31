using BayesianOptimization, GaussianProcesses,Statistics
using DelimitedFiles,LinearAlgebra
using Random
using ProximalOperators

# Read QTLMAS2010 data
X = readdlm("C://Users//pwaldman20//Documents//QTLMAS2010ny012.csv",',')
#X = readdlm("QTLMAS2010ny012.csv`",',')
ytot = (X[:,1].-mean(X[:,1]))
ytrain = ytot[1:2326]
Xtest= X[2327:size(X)[1],2:size(X)[2]]
ytest = ytot[2327:size(X)[1]]
Xtrain = X[1:2326,2:size(X)[2]]
p = size(Xtrain)[2]

# One hot encoding training data
Xtrain0 = copy(Xtrain)
Xtrain1 = copy(Xtrain)
Xtrain2 = copy(Xtrain)
Xtrain0[Xtrain0.==1] .= 2
Xtrain0[Xtrain0.==0] .= 1
Xtrain0[Xtrain0.==2] .= 0
Xtrain1[Xtrain1.==2] .= 0
Xtrain2[Xtrain2.==1] .= 0
Xtrain2[Xtrain2.==2] .= 1
Xtrain = hcat(Xtrain0,Xtrain1,Xtrain2)
# Set unimportant allocations to zero
Xtrain0 = 0
Xtrain1 = 0
Xtrain2 = 0
X = 0
p = size(Xtrain)[2]

# One hot encoding test data
Xtest0 = copy(Xtest)
Xtest1 = copy(Xtest)
Xtest2 = copy(Xtest)
Xtest0[Xtest0.==1] .= 2
Xtest0[Xtest0.==0] .= 1
Xtest0[Xtest0.==2] .= 0
Xtest1[Xtest1.==2] .= 0
Xtest2[Xtest2.==1] .= 0
Xtest2[Xtest2.==2] .= 1
Xtest = hcat(Xtest0,Xtest1,Xtest2)
# Set unimportant allocations to zero
Xtest0 = 0
Xtest1 = 0
Xtest2 = 0

# Calculate covariances for inits to ADMM
covar = cov(Xtrain,ytrain)

# Convergence tolerance of ADMM
tol=5e-4
# Maximum number of ADMM iterations
maxit=5000

# The LAVA ADMM function for BO
lava_admm_bo(lam::Vector) = lava_admm_bo(lam[1],lam[2]);
function lava_admm_bo(lam1,lam2)
  cu = zero(Xtrain[1,:])
  cu = covar[:,1]*0.0001
  du = zero(Xtrain[1,:])
  du = covar[:,1]*0.0001
  cuvcurr = zero(Xtrain[1,:])
  cv = zero(Xtrain[1,:])
  dv = zero(Xtrain[1,:])
  cz = zero(cv)
  lam1w = lam1
  lam2w = lam2
  gradL = zero(cv)
  hL = LeastSquares(Xtrain, ytrain) # Loss function L1
  fL = Translate(hL, du) # Translation function L1
  gL = NormL1(lam1w) # Regularization function L1
  dz = zero(dv)
  gradR = zero(dv)
  hR = LeastSquares(Xtrain, ytrain) # Loss function L2
  fR = Translate(hR, cu) # Translation function L2
  gR = SqrNormL2(lam2w) # Regularization function L2
  # Initial values for line search
  con = 0.5
  lrL = 0.9
  lrR = 0.9
  gamL = 0.9
  gamR = 0.9
  loss(d) = 0.5*norm(Xtrain*d-ytrain)^2 # Loss function for line search
  for it = 1:maxit
    # Line search L1
    gradL = Xtrain'*(Xtrain*cv-ytrain)
    while  loss(cu) > (loss(cv) +
      gradL'*(-cv) +
      (1.0/(2.0*lrL))*norm(-cv)^2)
      lrL = lrL * con
    end
    gamL = lrL
    cuvcurr = cu + du
    # ADMM perform f-update step L1
    prox!(cv, fL, cu - cz, gamL)
    # ADMM perform g-update step L1
    prox!(cu, gL, cv + cz, gamL)
    # Dual update L1
    cz .+= cv - cu
    # Line search L2
    gradR = Xtrain'*(Xtrain*dv-ytrain)
    while  loss(du) > (loss(dv) +
      gradR'*(-dv) +
      (1.0/(2.0*lrR))*norm(-dv)^2)
      lrR = lrR * con
    end
    gamR = lrR
    # ADMM perform f-update step L2
    prox!(dv, fR, du - dz, gamR)
    # ADMM perform g-update step L2
    prox!(du, gR, dv + dz, gamR)
    # Stopping criterion for ADMMM
    dualres = (cu + du) - cuvcurr
    reldualres = dualres/(norm(((cu + du) + cuvcurr)/2))
    if it % 5 == 2 && (norm(reldualres) <= tol)
      break
    end
    # Dual update L2
    dz .+= dv - du
  end
  Ytestpred = Xtest*(cu+du) # Test predictions
  MSEtest = (norm(Ytestpred.-ytest)^2)/length(ytest) # Test MSE
  return MSEtest
end

# optimize lambda 1 and 2 using BO

optlava = BOpt(lam->lava_admm_bo(lam[1],lam[2]),
  ElasticGPE(2, mean = MeanConst(0.), kernel = SEArd([2.,3.], 1.),
  logNoise = 4., capacity = 500),
  MutualInformation(),
  MAPGPOptimizer(every = 20, noisebounds = [-1.,10.],
    kernbounds = [[-3., -3., 0.], [6., 8., 8.]],
    maxeval = 40),
  [10.0,5000.0], [2000.0,300000.0], repetitions = 4, maxiterations = 250,
  sense = Min,
  verbosity = Progress)
reslavabo = boptimize!(optlava)
