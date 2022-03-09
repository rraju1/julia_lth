
using DataLoaders: DataLoader
using MLDataPattern: splitobs
using Flux
using FluxTraining

## defining the data

xs, ys = (
    # convert each image into h*w*1 array of floats 
    [Float32.(reshape(img, 28, 28, 1)) for img in Flux.Data.MNIST.images()],
    # one-hot encode the labels
    [Float32.(Flux.onehot(y, 0:9)) for y in Flux.Data.MNIST.labels()],
)

# split into training and validation sets
traindata, valdata = splitobs((xs, ys))

## Data Augmentation
# how to do data augmentation
# rotate_range 10 degrees
# width_shift_range 0.05
# height_shift_range 0.05
# zoom range 0.1
# horizontal flip True
# rescale 1/255

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
schedule = Schedule([0, 20es, 30es], [0.001, 0.0005, 0.00025])


optim = Flux.ADAM(0.001);
# log hyperparams
logcb = LogHyperParams(TensorBoardBackend("tblogs"))
## send to learner object
learner = Learner(model, (trainiter, valiter), optim, lossfn, Scheduler(LearningRate => schedule), Metrics(accuracy), ToGPU(), logcb)

## train model
FluxTraining.fit!(learner, 50)