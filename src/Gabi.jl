module Gabi


#Included Files
include(joinpath(Pkg.dir(),"Gabi", "src","models","svm","LibSvm.jl"))
include(joinpath(Pkg.dir(),"Gabi","src","models","svm","SvmModel.jl"))
# Exports
export Svm
export predict
export train!
export SvmType
export SvmKernel
end # module
