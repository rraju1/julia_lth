using Flux
using FluxTraining
using ParameterSchedulers
using MLDatasets
using MLDatasets: FileDataset
using DataLoaders: DataLoader
using DataAugmentation
using CoordinateTransformations
using Metalhead
using MLUtils
using BSON
using BitSAD, NNlibBitSAD

import MLDataPattern

# bugfix for mapobs
Base.getindex(data::MLUtils.MappedData, idx::Integer) = data.f(getobs(data.data, idx))
Base.getindex(data::MLUtils.MappedData, idxs::AbstractVector) =
    batch(map(Base.Fix1(getindex, data), idxs))

# bugfix for CachedDataset
Base.getindex(data::CachedDataset, idxs::AbstractVector) = batch([data[i] for i in idxs])

# bugfix for one hot arrays
Flux._indices(x::Base.ReshapedArray{<:Any, <:Any, <:Flux.OneHotVector}) =
  reshape([parent(x).indices], x.dims[2:end])

# hack for DataLoaders.jl + MLUtils.jl
MLDataPattern.LearnBase.getobs(x, i) = MLUtils.getobs(x, i)
MLDataPattern.LearnBase.nobs(x) = MLUtils.numobs(x)

# bug fix for ADAMW + FluxTraining
FluxTraining.setlearningrate!(os::Flux.Optimiser, lr) = foreach(os) do o
    if hasproperty(o, :eta)
        o.eta = lr
    end
end

# bug fix for ADAMW
ADAMW(η = 0.001, β = (0.9, 0.999), decay = 0) =
    Flux.Optimiser(ADAM(η, β), WeightDecay(decay))

include("vww.jl")
include("augmentation.jl")
include("mobilenet.jl")
