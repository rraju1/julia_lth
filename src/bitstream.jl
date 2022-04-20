function merge_conv_bn(model)
    backbone_layers = []
    i = 1
    while i <= length(model[1])
        if (model[1][i] isa Conv) && (model[1][i + 1] isa BatchNorm)
            c = model[1][i]
            bn = model[1][i + 1]

            # merge conv + bn
            sz = (1, 1, 1, length(bn.γ))
            w = c.weight .* reshape(bn.γ, sz...) ./ sqrt.(reshape(bn.σ², sz...))
            b = bn.γ .* (c.bias .- bn.μ) ./ sqrt.(bn.σ²) .+ bn.β

            # check for saturation
            nsaturated = count(x -> abs(x) > 1, w) + count(x -> abs(x) > 1, b)
            percent_saturated = nsaturated / (length(w) + length(b)) * 100
            @info "Detected $(round(percent_saturated; digits = 2))% saturated parameters"

            # create a new instance of Conv
            push!(backbone_layers, Conv(w, b, relu;
                                        stride = c.stride,
                                        pad = c.pad,
                                        dilation = c.dilation,
                                        groups = c.groups))
            i += 2
        else
            # for non-conv+bn, just copy the layer
            push!(backbone_layers, deepcopy(model[1][i]))
            i += 1
        end
    end

    fc_layers = []
    for layer in model[2]
        if layer isa Dense
            # no need to do anything special for Dense
            # we only have this loop to check for saturation
            w = copy(layer.weight)
            b = copy(layer.bias)

            # check saturation
            nsaturated = count(x -> abs(x) > 1, w) + count(x -> abs(x) > 1, b)
            percent_saturated = nsaturated / (length(w) + length(b)) * 100
            @info "Detected $(round(percent_saturated; digits = 2))% saturated parameters"

            push!(fc_layers, Dense(w, b, layer.σ))
        else
            push!(fc_layers, deepcopy(layer))
        end
    end

    return Chain(Chain(backbone_layers...), Chain(fc_layers...))
end

function scale_parameters!(layer::Union{Dense, Conv}, input_scaling = 1)
    # first scale the bias based on the existing input_scaling
    layer.bias ./= input_scaling

    # now compute the additional scaling for this layer
    n = max(maximum(abs, layer.weight), maximum(abs, layer.bias), 1)
    layer.weight ./= n
    layer.bias ./= n

    # check that nothing is saturated
    nsaturated = count(x -> abs(x) > 1, layer.weight) + count(x -> abs(x) > 1, layer.bias)
    percent_saturated = nsaturated / (length(layer.weight) + length(layer.bias)) * 100
    @info "Detected $(round(percent_saturated; digits = 2))% saturated parameters"

    # return the modified layer, additional scaling, and accumulated scaling
    return layer, n, input_scaling * n
end
function scale_parameters!(model::Chain, input_scaling = 1)
    scaling = []
    for layer in model
        _, n, input_scaling = scale_parameters!(layer, input_scaling)
        push!(scaling, n)
    end

    return model, scaling, input_scaling
end
scale_parameters!(layer, input_scaling = 1) = layer, 1, input_scaling

function get_activations(layer::Conv, input)
    _layer = Conv(layer.weight, false, identity;
                  stride = layer.stride,
                  pad = layer.pad,
                  dilation = layer.dilation,
                  groups = layer.groups)
    preact = _layer(input)
    _layer = Conv(layer.weight, layer.bias, identity;
                  stride = layer.stride,
                  pad = layer.pad,
                  dilation = layer.dilation,
                  groups = layer.groups)
    act = _layer(input)

    return layer(input), preact, act
end
function get_activations(layer::Dense, input)
    _layer = Dense(layer.weight, false, identity)
    preact = _layer(input)
    act = preact .+ layer.bias

    return layer.σ.(act), preact, act
end
function get_activations(m::Chain, input)
    preacts = []
    acts = []
    for layer in m
        input, preact, act = get_activations(layer, input)
        push!(preacts, preact)
        push!(acts, act)
    end

    return input, preacts, acts
end
get_activations(m, input) = m(input), [], []

function compute_input_scaling(model, data; device = cpu)
    model = device(model)
    act_vals = Float32[]
    for (x, _) in data
        _, preacts, acts = get_activations(model, device(x))
        append!(act_vals, mapreduce(vec, vcat, preacts[1]))
        append!(act_vals, mapreduce(vec, vcat, preacts[2]))
        append!(act_vals, mapreduce(vec, vcat, acts[1]))
        append!(act_vals, mapreduce(vec, vcat, acts[2]))
    end

    return act_vals
end

function prepare_bitstream_model(model)
    @info "Merging conv + bn"
    model = merge_conv_bn(model)
    @info "Scaling weights + biases"
    _, scaling, _ = scale_parameters!(model)

    return model, scaling
end
