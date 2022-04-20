include("src/setup.jl")
using Statistics: mean
using Random: shuffle!
using Base: @nexprs, @ntuple
using FileIO

## stratified obs hack
# copied from MLDataPattern

function _splitobs(lm::Dict{T,Vector{I}}, at::AbstractFloat) where {T,I<:Integer}
    0 < at < 1 || throw(ArgumentError("the parameter \"at\" must be in interval (0, 1)"))
    n = mapreduce(length, +, values(lm))
    k = length(keys(lm))
    # preallocate the indices vectors
    idx1 = Vector{I}()
    idx2 = Vector{I}()
    # sizehint will save us a few heavy memory allocations
    # we specify "+ k" to deal with trailing observations when
    # the number of observations from a class isn't divideable
    # by "at" or "1-at"
    sizehint!(idx1, ceil(Int, n * at     + k))
    sizehint!(idx2, ceil(Int, n * (1-at) + k))
    # loop through all label indices
    for indices in values(lm)
        i1, i2 = splitobs(indices; at = at)
        append!(idx1, i1)
        append!(idx2, i2)
    end
    idx1, idx2
end

@generated function _splitobs(lm::Dict{T,Vector{I}}, at::NTuple{N,AbstractFloat}) where {T,I<:Integer,N}
    quote
        n = mapreduce(length, +, values(lm))
        k = length(keys(lm))
        # preallocate the indices vectors
        @nexprs $(N+1) i -> idx_i = Vector{I}()
        # sizehint will save us a few heavy memory allocations
        # we specify "+ k" to deal with trailing observations when
        # the number of observations from a class isn't divideable
        # by "at" or "1-at"
        @nexprs $(N) i -> sizehint!(idx_i, ceil(Int, n*at[i] + k))
        sizehint!($(Symbol(:idx_, Symbol(N+1))), ceil(Int, n*(1-sum(at)) + k))
        # loop through all label indices
        for indices in values(lm)
            tup = splitobs(indices; at = at)
            @nexprs $(N+1) i -> append!(idx_i, tup[i])
        end
        # return a tuple of all indices vectors
        @ntuple $(N+1) idx
    end
end

function stratifiedobs(data, labels = [x[2] for x in eachobs(data)]; p, shuffle::Bool = true)
    # The given data is always shuffled to qualify as performing
    # stratified sampling without replacement.
    data_shuf = shuffleobs(data)
    idx_tup = _splitobs(group_indices(labels), p)
    # Setting the parameter "shuffle = false" specifies that the
    # classes are ordered in the resulting subsets respectively.
    shuffle && foreach(x->isempty(x) || shuffle!(x), idx_tup)

    return map(idx -> obsview(data_shuf, idx), idx_tup)
end

## defining the data

dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vw_coco2014_96/"
valdata = VisualWakeWords(dataroot; subset = :val)

## split into three random subsets
# this is non-deterministic
# (i.e. you cannot exactly reproduce the data split on disk)

frac = 100 / numobs(valdata)
testset, hiddenset, valset = stratifiedobs(valdata, valdata.labels;
                                           p = (frac, frac), shuffle = true)

## check statistics

testperson = mean(mapobs(x -> Flux.onecold(x.label, [0, 1]), testset))
hiddenperson = mean(mapobs(x -> Flux.onecold(x.label, [0, 1]), hiddenset))
valperson = mean(mapobs(x -> Flux.onecold(x.label, [0, 1]), valset))

@info """
      Test set: $(numobs(testset)) samples, $(testperson * 100)% person class
      Validation set: $(numobs(valset)) samples, $(valperson * 100)% person class
      Hidden set: $(numobs(hiddenset)) samples, $(hiddenperson * 100)% person class
      """

##

sets = [("test-hackathon", testset), 
        ("val-hackathon", valset),
        ("hidden-hackathon", hiddenset)]
for (set, data) in sets
    @info "Writing subset $set to disk"
    labeltxt = open(joinpath(dataroot, "$set.txt"), "w")
    imagefolder = joinpath(dataroot, set)
    isdir(imagefolder) || mkpath(imagefolder)
    for (i, (img, label)) in enumerate(eachobs(data))
        save(joinpath(imagefolder, "img-$i.jpeg"), img)
        write(labeltxt, "img-$i.jpeg $(Flux.onecold(label, [0, 1]))\n")
    end
    close(labeltxt)
end
