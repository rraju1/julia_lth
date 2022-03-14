using DataLoaders: DataLoader
using MLDataPattern: splitobs
using Flux
using FluxTraining
using FluxTraining: Callback, HyperParameter, StepBegin, AbstractTrainingPhase, Read, Write
using ParameterSchedulers
using MLDatasets
using TensorBoardLogger
using ValueHistories
using DataAugmentation
using Functors
using Metalhead: conv_bn
using MLUtils   

include("scheduler.jl")
# include("mobilenet.jl")