include("src/setup.jl")

## defining the data

dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vw_coco2014_96/"
traindata = VisualWakeWords(dataroot; subset = :train) |> shuffleobs
valdata = VisualWakeWords(dataroot; subset = :val)

## data augmentation

# how to do data augmentation
# rotate_range 10 degrees (i.e. random [-10, 10])
# width_shift_range 0.05
# height_shift_range 0.05
# zoom range 0.1 (i.e random [0.9, 1.1])
# horizontal flip True
# rescale 1/255
augmentations = Rotate(10) |>
                RandomTranslate((96, 96), (0.05, 0.05)) |>
                Zoom((0.9, 1.1)) |>
                ScaleFixed((96, 96)) |>
                Maybe(FlipX()) |>
                CenterCrop((96, 96)) |>
                ImageToTensor()
trainaug = map_augmentation(augmentations, traindata)
valaug = map_augmentation(ImageToTensor(), valdata)
;

## model definition

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 1)
# Flux.loadmodel!(m, BSON.load(joinpath(artifact"mobilenet", "mobilenet.bson"))[:m])

## data loaders

bs = 32
trainloader = DataLoader(BatchView(trainaug; batchsize = bs), nothing; buffered = true)
valloader = DataLoader(BatchView(valaug; batchsize = 2 * bs), nothing; buffered = true)
;

## training setup

lossfn = Flux.Losses.logitbinarycrossentropy
accfn(ŷ, y) = mean((ŷ .> 0) .== y)

# define schedule and optimizer
es = length(trainloader)
initial_lr = 0.01
schedule = Interpolator(Step(initial_lr, 0.5, [25, 45]), es)
optim = Momentum(initial_lr)

# callbacks
logger = TensorBoardBackend("tblogs")
# schcb = Scheduler(LearningRate => schedule)
logcb = (LogMetrics(logger),)# LogHyperParams(logger))
valcb = Metrics(Metric(accfn; phase = TrainingPhase, name = "train_acc"),
                Metric(accfn; phase = ValidationPhase, name = "val_acc"))

# setup learner object
learner = Learner(m, lossfn;
                  data = (trainloader, valloader),
                  optimizer = optim,
                  callbacks = [ToGPU(), logcb..., valcb])

## train model

FluxTraining.fit!(learner, 50)
# make sure to close logger due to network fs
close(logger.logger)

## save model

m = learner.model |> cpu
BSON.@save "mobilenet-relu.bson" m
