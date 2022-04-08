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

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)

## data loaders

bs = 32
trainloader = DataLoader(BatchView(trainaug; batchsize = bs), nothing; buffered = true)
valloader = DataLoader(BatchView(valaug; batchsize = 2 * bs), nothing; buffered = true)
;

## training setup

lossfn = Flux.Losses.logitcrossentropy

# define schedule and optimizer
es = length(trainloader)
initial_lr = 0.01
schedule = Interpolator(Step(initial_lr, 0.5, [35, 10, 5]), es)
# this is a patched ADAMW not Flux.ADAMW
optim = Momentum(initial_lr)

# callbacks
logger = TensorBoardBackend("tblogs")
schcb = Scheduler(LearningRate => schedule)
hlogcb = LogHyperParams(logger)
mlogcb = LogMetrics(logger)
valcb = Metrics(Metric(accuracy; phase = TrainingPhase, name = "train_acc"),
                Metric(accuracy; phase = ValidationPhase, name = "val_acc"))

# setup learner object
learner = Learner(m, lossfn;
                  data = (trainloader, valloader),
                  optimizer = optim,
                  callbacks = [schcb, ToGPU(), hlogcb, mlogcb, valcb])

## train model

FluxTraining.fit!(learner, 50)
# make sure to close logger due to network fs
close(logger.logger)

## save model

BSON.@save "mobilenet-relu.bson" m
