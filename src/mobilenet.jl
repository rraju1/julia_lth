relu1(x) = min(relu(x), 1)
relu1(x::SBitstream) = relu(x)
relu1_scale(x) = relu1(x / 4)
relu1_scale(x::SBitstream) = relu1(x)
relu2(x) = min(relu(x), 2)
relu_scale(x, n) = relu(x / n)
relu_scale(n) = Base.Fix2(relu_scale, n)

function MobileNet(activation = relu, width_mult = 1;
                   revbn = false, bias = true, affine = true, kwargs...)
    base = Metalhead.mobilenetv1(width_mult, Metalhead.mobilenetv1_configs;
                                 activation = activation, kwargs...)

    if revbn || !bias || !affine
        backbone_layers = []
        i = 1
        while i <= length(base[1])
            if revbn && (base[1][i] isa Conv) && (base[1][i + 1] isa BatchNorm)
                c = base[1][i]
                bn = base[1][i + 1]
                b = bias ? c.bias : false
                push!(backbone_layers, Conv(c.weight, b, bn.λ;
                                            stride = c.stride,
                                            pad = c.pad,
                                            dilation = c.dilation,
                                            groups = c.groups))
                push!(backbone_layers,
                      BatchNorm(identity, bn.β, bn.γ, bn.μ, bn.σ², bn.ϵ,
                                bn.momentum, affine, bn.track_stats, bn.active, bn.chs))
                i += 2
            elseif !bias && (base[1][i] isa Conv)
                c = base[1][i]
                push!(backbone_layers, Conv(c.weight, false, c.σ;
                                            stride = c.stride,
                                            pad = c.pad,
                                            dilation = c.dilation,
                                            groups = c.groups))
                i += 1
            elseif !affine && (base[1][i] isa BatchNorm)
                bn = base[1][i]
                push!(backbone_layers,
                      BatchNorm(bn.λ, bn.β, bn.γ, bn.μ, bn.σ², bn.ϵ,
                                bn.momentum, false, bn.track_stats, bn.active, bn.chs))
                i += 1
            else
                push!(backbone_layers, base[1][i])
                i += 1
            end
        end

        return Chain(Chain(backbone_layers...), base[2])
    else
        return base
    end
end
