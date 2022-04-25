include("src/setup.jl")

BitSAD.set_saturation_verbosity(:none)

## defining the data

# dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vww-hackathon/"
dataroot = joinpath(artifact"vww", "vww-hackathon")
testdata = VisualWakeWords(dataroot; subset = :test)

## data augmentation

testset = map_augmentation(ImageToTensor(), testdata)
;

## model definition and select a random sample

modelpath = joinpath(artifact"mobilenet", "mobilenet.bson")
m = BSON.load(modelpath)[:m]

# m = BSON.load("mobilenet-relu.bson")

# apply scaling
mbit, scaling = prepare_bitstream_model(m)
total_scaling = prod(prod.(scaling))

## test error

## m(x) .- mbit(x) .* total_scaling

# apply conversion error

bslen = 10000

morigbit = deepcopy(mbit)
mconverror = add_conversion_error!(mbit, bslen)

## loop through test set
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
trueposce = 0
truenegce = 0
falseposce = 0
falsenegce = 0
for (x, y) in testset
    x = Flux.unsqueeze(x, 4)
    yconverror = mconverror(x) * total_scaling
    ybit = morigbit(x) * total_scaling
    if y[1] == 1
        if ybit[1] > 0.5
            global truepos += 1
        else
            global falseneg += 1
        end
        if yconverror[1] > 0.5
            global trueposce += 1
        else
            global falsenegce += 1
        end
    else
        if ybit[1] < 0.5
            global trueneg += 1
        else
            global falsepos += 1
        end
        if yconverror[1] < 0.5
            global truenegce += 1
        else
            global falseposce += 1
        end
    end
    println("ybit ", ybit[1], " TP: ", truepos, " FP: ", falsepos, " TN: ", trueneg, " FN: ", falseneg)
    println("yce  ", yconverror[1], " TP: ", trueposce, " FP: ", falseposce, " TN: ", truenegce, " FN: ", falsenegce)
end

println("Final TP: ", truepos, " FP: ", falsepos, " TN: ", trueneg, " FN: ", falseneg)
println("Accuracy:", truepos+trueneg)

println("Final TP w/conv error: ", trueposce, " FP: ", falseposce, " TN: ", truenegce, " FN: ", falsenegce)
println("Accuracy:", trueposce+truenegce)




## x, y = randobs(testset)
## x = Flux.unsqueeze(x, 4)

## prepare for bitstream

##mbit, scaling = prepare_bitstream_model(m)
##total_scaling = prod(prod.(scaling))

## test error

## m(x) .- mbit(x) .* total_scaling

## convert to bitstream


## test error

##ybit = mbit(xbit)
##y = m(x)
##ybit_scaled = float.(ybit) .* total_scaling
##mean(abs.(y .- ybit_scaled))

## make simulatable

## @time msim = make_simulatable(mbit, size(xbit))

## test simulatable

##ysim = msim(xbit)
