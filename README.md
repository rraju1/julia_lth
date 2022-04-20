# Bitstream hackathon prototype repo

This repo allows us to prototype scripts for the bitstream hackathon before converting them into tutorials.

All scripts are contained in the root directory of the repo. The `src/` folder should contain a "library" of functions that we expect to provide the participants. If you are defining a utility function that isn't hyper-specific to a particular script, then it should go under `src/`. The `src/setup.jl` file should be the only entry point into the "library" (i.e. it is responsible for importing any packages and additional source files). Using the "library" should be as simple as
```julia
include("src/setup.jl")
```

When you are satisfied with a script and you are ready to port it to the website, follow these steps:
1. Open the Julia REPL and enter Pkg mode (press `]`). Make sure you see `julia_lth` as the environment at the prompt. If not, run `activate .` to the activate the environment at the current working directory (assuming you launched the REPL from the root of this repo).
2. Run `status` in Pkg mode to see the currently installed packages. Make sure that the package versions appear correct. This step should just be a sanity check before copying things over. Ideally, you wouldn't need to do Steps 1 and 2.
3. Copy the contents of `Project.toml`, `Manifest.toml`, and `Artifacts.toml` to the `_tutorials/` folder in the website repo.
4. Copy the contents of _all_ the files in `src/` to the `_tutorials/src/` folder in the website.
5. Write the new tutorial under `_tutorials/` and add the tutorial to the website following the instructions on the website repo README.
6. Push the changes to the website repo.
