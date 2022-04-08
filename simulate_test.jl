include("src/setup.jl")

##

m = BSON.load("mobilenet-relu-nowd.bson")[:m]
x = rand(Float32, 96, 96, 3, 1)

##

mbit = m |> tosbitstream
xbit = SBitstream.(x)

##

ybit = mbit(xbit)
