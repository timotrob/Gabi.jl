
#=0 -- C-SVC
1 -- nu-SVC
2 -- one-class SVM
3 -- epsilon-SVR
4 -- nu-SVR=#
baremodule SvmType
   const CSVC = 0 # Classification
   const NuSVC = 1 #Classification
   const OneClassSVM = 2 #Classification
   const EpsilonSVR = 3 # Regression
   const NuSVR =4  # Regression
end

#=0 -- linear: u'*v
1 -- polynomial: (gamma*u'*v + coef0)^degree
2 -- radial basis function: exp(-gamma*|u-v|^2)
3 -- sigmoid: tanh(gamma*u'*v + coef0)=#
baremodule SvmKernel
   const Linear = 0
   const Polynomial = 1
   const RBF = 2
   const Sigmoid = 3
end

type Svm
  nativeModel
  x
  y
  _svmType::Int32
  _kernel::Int32
  _degree::Integer
  _gamma::Float64
  _coef0::Float64
  _C::Float64
  _nu::Float64
  _p::Float64
  _cache_size::Float64
  _eps::Float64
  _shrinking::Bool
  _probabilityEstimates::Bool
  _weights::Union{Dict{Any, Float64}, Void}
  _verbose::Bool
  Svm() = new(nothing,# NaoviteModel
                nothing,# X
                nothing,# Y
                SvmType.CSVC, # SvmType
                SvmKernel.RBF, #Kernel
                Integer(3), # Degrre
                typemin(Float64),#gamma
                0.0, # coef0
                Float64(1.0), # C (Cost)
                Float64(0.5), # nu
                Float64(0.1),#p
                Float64(100.00), # cache_size
                Float64(0.001), # eps,
                true, #shrinking
                false,#probability_estimates
                nothing,#weights
                false) # verbosity
end
#=
options:
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC
	1 -- nu-SVC
	2 -- one-class SVM
	3 -- epsilon-SVR
	4 -- nu-SVR
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
=#
function train!{U<:Real,T}(model::Svm, x::Matrix{U}, y::Vector{T})
  model.y = y
  model.x = x
  if (model._svmType==nothing)
    if (eltype(y)<:Real)
      model._svm_type = SvmType.EpsilonSVR # eps-SVR
    else
      model._svm_type = SvmType.CSVC # C-SVR
    end
    if (model._gamma==typemin(Float64))
      model._gamma=1.0/size(x, 1)
    end
  end
  model.nativeModel= svmtrain(y, transpose(x);
          svm_type=model._svmType,
          kernel_type=model._kernel, degree=model._degree,
          gamma=model._gamma, coef0=model._coef0,
          C=model._C, nu=model._nu, p=model._p,
          cache_size=model._cache_size, eps=model._eps, shrinking=model._shrinking,
          probability_estimates=model._probabilityEstimates,
          weights=model._weights,
          verbose=model._verbose)
end

function predict{U<:Real}(model::Svm,x::Matrix{U})
  (predicted_labels, decision_values) = svmpredict(model.nativeModel,transpose(x))
  return predicted_labels
end
