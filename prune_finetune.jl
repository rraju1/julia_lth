include("src/setup.jl")
using FluxPrune
using Random: shuffle

## defining the data

dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vww-hackathon/"
# dataroot = joinpath(artifact"vww", "vww-hackathon")
traindata = VisualWakeWords(dataroot; subset = :train)
testdata = VisualWakeWords(dataroot; subset = :val)

## set data augmentation

augmentations = Rotate(10) |>
                RandomTranslate((96, 96), (0.05, 0.05)) |>
                Zoom((0.9, 1.1)) |>
                ScaleFixed((96, 96)) |>
                Maybe(FlipX()) |>
                CenterCrop((96, 96)) |>
                ImageToTensor()
trainset = map_augmentation(augmentations, traindata)


testset = map_augmentation(ImageToTensor(), testdata)
;

## model definition and select a random sample

modelpath = joinpath(artifact"mobilenet", "mobilenet.bson")
m = BSON.load(modelpath)[:m] |> gpu


## data loaders

bs = 32
# trainloader = DataLoader(BatchView(trainset; batchsize = bs), nothing; buffered = true)
valloader = DataLoader(BatchView(testset; batchsize = bs), nothing; buffered = true)
;

## training setup

lossfn = Flux.Losses.logitbinarycrossentropy
accfn(ŷ::AbstractArray, y::AbstractArray) = mean((ŷ .> 0) .== y)
accfn(data, m) = mean(accfn(m(gpu(x)), gpu(y)) for (x, y) in data)

## score the testset to see accuracy

@time println("Validation Accuracy of unpruned model: ", accfn(valloader, m))

## set pruning
# just do channel pruning for now
stages = [ChannelPrune(0.1),
          ChannelPrune(0.2),
          ChannelPrune(0.3)]

## check model
# set arbitary lr for now as well as target acc
target_acc = 0.78
nepochs = 5
m̄ = iterativeprune(stages, m) do m̄
    opt = Momentum(0.01)
    ps = Flux.params(m̄)
    subset = ObsView(trainset, shuffle(1:numobs(trainset))[1:5000])
    trainloader = DataLoader(BatchView(subset; batchsize = bs), nothing; buffered = true)
    for epoch in 1:nepochs
        @info "Epoch $epoch"
        @time for (x, y) in trainloader
            _x, _y = gpu(x), gpu(y)
            gs = Flux.gradient(ps) do
                lossfn(m̄(_x), _y)
            end
            Flux.update!(opt, ps, gs)
        end
    end
    GC.gc()
    Flux.CUDA.reclaim()
    @show current_accuracy = accfn(valloader, m̄)
    return current_accuracy > target_acc
end