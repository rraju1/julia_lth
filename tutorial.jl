using Flux
using Flux.Zygote

## ground truth definition

# Define the ground truth model. We aim to recover W_truth and b_truth using
# only examples of ground_truth()
W_truth = [1 2 3 4 5;
            5 4 3 2 1]
b_truth = [-1.0; -2.0]
ground_truth(x) = W_truth*x .+ b_truth

## training data

# Generate the ground truth training data as vectors-of-vectors
x_train = [ 5 .* rand(5) for _ in 1:10_000 ]
y_train = [ ground_truth(x) + 0.2 .* randn(2) for x in x_train ]

## model

# Define and initialize the model we want to train
model(x) = W*x .+ b
W = rand(2, 5)
b = rand(2)

## loss

# Define pieces we need to train: loss function, optimiser, examples, and params
function loss(x, y)
  ŷ = model(x)
  sum((y .- ŷ).^2)
end
opt = Descent(0.01)
train_data = zip(x_train, y_train)
ps = params(W, b)

## training loop

# Execute a training epoch
for (x,y) in train_data
  gs = gradient(ps) do
    loss(x,y)
  end
  Flux.Optimise.update!(opt, ps, gs)
end

# An alternate way to execute a training epoch
# Flux.train!(loss, params(W, b), train_data, opt)

# Print out how well we did
@show W
@show maximum(abs, W .- W_truth)