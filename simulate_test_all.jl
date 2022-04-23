include("src/setup.jl")

## defining the data

##dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vww-hackathon/"
dataroot = joinpath(artifact"vww", "vww-hackathon")
testdata = VisualWakeWords(dataroot; subset = :test)

## data augmentation

testset = map_augmentation(ImageToTensor(), testdata)
;

## model definition and select a random sample
##modelpath = joinpath(artifact"mobilenet", "mobilenet.bson")
##m = BSON.load(modelpath)[:m]

## mbit = BSON.load("mobilenet-bitsad.bson")[:m]
## mbit = mbit |> tosbitstream

BitSAD.set_saturation_verbosity(:none)

## loop through test set
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
for (x, y) in testset
    local xu = Flux.unsqueeze(x, 4)
    local xbit = SBitstream.(xu)
    local ybit = msim(xbit)
    for t in 1:100
        push!.(ybit, pop!.(msim(xbit)))
        println("y: ", y[1], "yestimate[", t, "]: ", estimate(ybit[1]))
    end
    yestimate = estimate(ybit[1])
    if (y[1] > 0)
        if (yestimate > 0.5)
            global truepos += 1
        else
            global falseneg += 1
        end
    else
        if (yestimate <= 0.5)
            global trueneg += 1
        else
            global falsepos += 1
        end
    end
    println("yest: ", yestimate, " TP: ", truepos, " FP: ", falsepos, " TN: ", trueneg, " FN: ", falseneg)
end

println("Final TP: ", truepos, " FP: ", falsepos, " TN: ", trueneg, " FN: ", falseneg)
println("Accuracy:", truepos+trueneg)





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
