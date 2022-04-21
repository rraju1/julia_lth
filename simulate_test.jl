include("src/setup.jl")

## defining the data

dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vww-hackathon/"
testdata = VisualWakeWords(dataroot; subset = :test)

## data augmentation

testset = map_augmentation(ImageToTensor(), testdata)
;

## model definition and select a random sample

m = BSON.load("mobilenet.bson")[:m]
x, y = randobs(testset)
x = Flux.unsqueeze(x, 4)

## prepare for bitstream

mbit, scaling = prepare_bitstream_model(m)
total_scaling = prod(prod.(scaling))

## test error

m(x) .- mbit(x) .* total_scaling

## convert to bitstream

BitSAD.set_saturation_verbosity(:none)
mbit = mbit |> tosbitstream
xbit = SBitstream.(x)

## test error

ybit = mbit(xbit)
y = m(x)
ybit_scaled = float.(ybit) .* total_scaling
mean(abs.(y .- ybit_scaled))

## make simulatable

@time msim = make_simulatable(mbit, size(xbit))

## test simulatable

ysim = msim(xbit)
