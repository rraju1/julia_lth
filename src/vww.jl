struct VisualWakeWords <: MLUtils.AbstractDataContainer
    path::String
    images::FileDataset
    labels::Vector{Int}
end

function VisualWakeWords(dir; subset = :train)
    (subset in (:train, :val)) ||
        throw(ArgumentError("VisualWakeWords keyword subset must be :train or :val."))

    paths = String[]
    labels = Int[]
    for line in readlines(joinpath(dir, "$subset.txt"))
        file, label = split(line, " ")
        push!(paths, joinpath(dir, string(subset), file))
        push!(labels, parse(Int, label))
    end

    VisualWakeWords(joinpath(dir, string(subset)), FileDataset(paths), labels)
end

Base.show(io::IO, ::MIME"text/plain", data::VisualWakeWords) =
    print(io, "VisualWakeWords(path = $(data.path), # of samples = $(length(data)))")

Base.length(data::VisualWakeWords) = length(data.labels)
Base.getindex(data::VisualWakeWords, i::Integer) =
    (image = data.images[i], label = Flux.onehot(data.labels[i], [0, 1]))
Base.getindex(data::VisualWakeWords, is::AbstractVector) =
    (image = batch(data.images[is]), label = Flux.onehotbatch(data.labels[is], [0, 1]))
