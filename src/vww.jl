struct VisualWakeWords{T<:FileDataset} <: MLUtils.AbstractDataContainer
    path::String
    images::T
    labels::Vector{Int}
end

function VisualWakeWords(dir; subset = :train)
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
    (image = data.images[i], label = [data.labels[i]])
Base.getindex(data::VisualWakeWords, is::AbstractVector) =
    (image = batch(data.images[is]), label = Flux.unsqueeze(data.labels[is], 1))
