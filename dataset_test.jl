import DataLoaders.LearnBase: getobs, nobs
using Images

struct ImageDataset
    files::Vector{String} # holds image name and label
    root_dir::String 
    subset::String 
end

ImageDataset(root_dir::String, subset::String, txt_path::String) = ImageDataset(readlines(txt_path), root_dir, subset)

nobs(data::ImageDataset) = length(data.files)
getobs(data::ImageDataset, i::Int) = Images.load(joinpath(data.root_dir, data.subset, split(data.files[i], " ")[1])), parse(Float64, split(data.files[i], " ")[2]) # load image and label

# 
rootdir = "/group/ece/ececompeng/lipasti/libraries/datasets/vw_coco2014_96"
subset = "train"
train_txt_path = joinpath(rootdir, "train.txt")

train_dataset = ImageDataset(rootdir, subset, train_txt_path)
train_im, train_lbl = getobs(train_dataset, 9) # random test case

val_subset = "val"
val_txt_path = joinpath(rootdir, "val.txt")
val_dataset = ImageDataset(rootdir, val_subset, val_txt_path)
val_im, val_lbl = getobs(val_dataset, 9) # random test case