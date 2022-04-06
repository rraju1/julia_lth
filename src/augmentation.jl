struct Pad{N} <: DataAugmentation.Transform
    size::NTuple{N, Int}
end
Pad(size::Int, N::Int) = Pad(ntuple(i -> size, N))

function DataAugmentation.apply(pad::Pad{M}, item::Image{N}; randstate) where {M, N}
    @assert M == 2*N "For an Image{N}, supply a Pad{2*N}."

    output = NNlib.pad_zeros(item.data, pad.size)

    return Image(output, item.bounds)
end

struct RandomTranslate{N} <: DataAugmentation.ProjectiveTransform
    shift::NTuple{N, Int}
end
RandomTranslate(sz, shift::NTuple{2, <:AbstractFloat}) =
    RandomTranslate(map((s, scale) -> round(Int, s * scale), sz, shift))

DataAugmentation.getrandstate(tfm::RandomTranslate) = map(s -> rand(-s:s), tfm.shift)

DataAugmentation.getprojection(tfm::RandomTranslate, bounds::Bounds;
                               randstate = DataAugmentation.getrandstate(tfm)) =
    Translation(randstate...)

# RandomTranslate(sz, shift::NTuple{2, Int}) =
#     Pad((shift[1], shift[1], shift[2], shift[2])) |> RandomCrop(sz)


apply_augmenation(pipeline, x) = itemdata(DataAugmentation.apply(pipeline, Image(x)))
apply_augmenation(pipeline, x::AbstractVector) =
    map(Base.Fix1(apply_augmenation, pipeline), x)
apply_augmenation(pipeline, x::NamedTuple{(:image, :label)}) =
    (image = apply_augmenation(pipeline, x.image), label = x.label)
map_augmentation(pipeline, data) =
    MLUtils.mapobs(Base.Fix1(apply_augmenation, pipeline), data)
