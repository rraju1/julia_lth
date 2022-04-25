using BitSAD

function conversion_error(a,blen)
    b = SBitstream(a)
    generate!(b,blen)
    retval = estimate(b)
    ##println("a ",a," aest", retval)
    return retval
end

function add_conversion_error!(layer::Union{Dense, Conv}, blen)

    #newlayer.bias = layer.bias
    #newlayer.weight = layer.weight
    #newlayer = deepcopy(layer)
    # convert each parameter to bitstream and back to introduce quantization add_conversion_error
    layer.bias .= conversion_error.(layer.bias, blen)
    layer.weight .= conversion_error.(layer.weight, blen)
 
    return layer
end

function add_conversion_error!(model::Chain, blen)
    for layer in model
        _ = add_conversion_error!(layer, blen)
    end

    return model
end

add_conversion_error!(layer, blen) = layer

