include("src/setup.jl")
include("src/mobilenet.jl")

## defining the data

xs = MLDatasets.CIFAR10.traintensor(Float32)
# xs = Flux.unsqueeze(MLDatasets.CIFAR10.traintensor(Float32), 3)
ys = Float32.(Flux.onehotbatch(MLDatasets.CIFAR10.trainlabels(), 0:9))

# split into training and validation sets
# 70% train, 30% val
traindata, valdata = splitobs((xs, ys); at = 0.7)

## Data Augmentation
# how to do data augmentation
# rotate_range 10 degrees
# width_shift_range 0.05
# height_shift_range 0.05
# zoom range 0.1
# horizontal flip True
# rescale 1/255
m = MobileNetv2(nclasses = 10)

## 

# create iterators
trainiter, valiter = DataLoader(traindata, 50, buffered=false), DataLoader(valdata, 50, buffered=false);

## defining the model
model = Chain(
    Conv((3, 3), 1 => 16, relu, pad = 1, stride = 2),
    Conv((3, 3), 16 => 32, relu, pad = 1),
    GlobalMeanPool(),
    Flux.flatten,
    Dense(32, 10),
)

## loss function and optimizer
lossfn = Flux.Losses.logitcrossentropy
# define schedule
es = length(trainiter)
schedule = Interpolator(Step(0.001, 0.5, 10), es)


optim = Flux.ADAM(0.001);
# log hyperparams
logcb = LogHyperParams(TensorBoardBackend("tblogs"))
## send to learner object
learner = Learner(m, (trainiter, valiter), optim, lossfn, Scheduler(LearningRate => schedule), Metrics(accuracy), ToGPU(), logcb)

## train model
FluxTraining.fit!(learner, 50)
