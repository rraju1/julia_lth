relu1(x) = min(relu(x), 1)

MobileNet(activation = relu, width_mult = 1; kwargs...) =
  Metalhead.mobilenetv1(width_mult, Metalhead.mobilenetv1_configs;
                        activation = activation, kwargs...)
