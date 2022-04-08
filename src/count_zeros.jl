## import stuff
using Flux
using LinearAlgebra
using FluxPrune
using MaskedArrays
## 
function compute_dot_prods(model::Chain, input_size)
    num_multiplies_total = 0
    num_accumulates_total = 0
    intermediate_size = input_size

    for i in 1:length(model)
        num_multiplies, num_accumulates, output_size = compute_dot_prods(model[i], intermediate_size)
        num_multiplies_total += num_multiplies
        num_accumulates_total += num_accumulates
        println("num_multiplies in layer ", i, ": ", num_multiplies)
        println("num_accumulates in layer ", i, ": ", num_accumulates)
        println("output_size: ", output_size)  
        intermediate_size = output_size
    end
    return num_multiplies_total, num_accumulates_total, intermediate_size
end

# default case
function compute_dot_prods(m, input_size)
    output_size = Flux.outputsize(m, input_size) # wxhxcinxcout
    return 0, 0, output_size
end


# count(iszero.(m[1].weight))
function compute_dot_prods(m::Dense, input_size)
    output_size = Flux.outputsize(m, input_size) # wxhxcinxcout
    num_rows_zero = sum(iszero(w) for w in eachrow(m.weight)) # find the number of zero rows
    total = count(iszero, m.weight) 
    num_input_features = size(m.weight)[2] # input features
    unstruct = total - num_rows_zero * num_input_features # total zeros - number of zeros from pruned neurons
    num_multiplies = prod(size(m.weight)) - unstruct - num_rows_zero * num_input_features
    num_accumulates = (num_input_features - 1) * (size(m.weight, 1) - num_rows_zero)
    return num_multiplies, num_accumulates, output_size
end

# tuple size
function compute_dot_prods(m::Conv, input_size)
    size_tuple = size(m.weight)
    conv_weight = reshape(m.weight, (prod(size_tuple[1:3]), size_tuple[4])) # reshape conv2d tensor into 2D matrix
    output_size = Flux.outputsize(m, input_size) # WxHxCinxCout
    num_patches = output_size[1] * output_size[2] # height (rows) of input matrix
    dense_baseline_mults = num_patches * prod(size(conv_weight)) # assume no pruning
    
    # get number of total zeros in weight and count number of pruned channels and unstructured zeros in weight
    total = count(iszero, m.weight)
    num_col_zeros = sum(iszero(w) for w in eachcol(conv_weight)) # how many pruned channels
    unstruct = total - num_col_zeros * size(conv_weight)[1] # don't double count zero cols
    
    # Number of multiplies in layer
    structured_macs_num_reduction = num_col_zeros * num_patches * size(conv_weight)[1]
    num_multiplies = dense_baseline_mults - unstruct - structured_macs_num_reduction

    # Number of adds 
    dense_baseline_adds = num_patches * (size(conv_weight)[1] - 1) * (size(conv_weight)[2])
    structured_baseline_adds = num_patches * (size(conv_weight)[1] - 1) * num_col_zeros
    num_accumulates = dense_baseline_adds - structured_baseline_adds

    return num_multiplies, num_accumulates, output_size
end
