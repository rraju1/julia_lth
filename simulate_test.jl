include("src/setup.jl")

## defining the data

dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vww-hackathon/"
testdata = VisualWakeWords(dataroot; subset = :test)

## data augmentation

testset = map_augmentation(ImageToTensor(), testdata)
;

## model definition and select a random sample

m = BSON.load("mobilenet-relu.bson")[:m]
x, y = randobs(testset)
x = Flux.unsqueeze(x, 4)

## convert to bitstream

mbit, scaling = scale_parameters!(merge_conv_bn(m))
mbit = mbit |> tosbitstream
xbit = SBitstream.(x)

##

ybit = mbit(xbit)

##

@time msim = simulatable(mbit, xbit)

##
