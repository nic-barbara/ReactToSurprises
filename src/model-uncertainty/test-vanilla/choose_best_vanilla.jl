using BSON
using CairoMakie
using LinearAlgebra
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "../models.jl"))
include(joinpath(@__DIR__, "../functions.jl"))
include(joinpath(@__DIR__, "../setup_mass.jl"))


fpath = joinpath(@__DIR__, "../../../results/model-uncertainty/batch-vanilla/")
get_fname(lr, id) = string(fpath, "lcp_vanilla_mass_154nvl_177nvm_lr$(lr)_v$(id).bson")
load_data(lr, id, key) = BSON.load(get_fname(lr, id))[key]


data = BSON.load(get_fname(1e-3, 1))

data["vanilla_lstm"] = load_data(5e-3, 1, "vanilla_lstm")
data["costs_vl"] = load_data(5e-3, 1, "costs_vl")

data["vanilla_mlp"] = load_data(1e-2, 1, "vanilla_mlp")
data["costs_vm"] = load_data(1e-2, 1, "costs_vm")

savedir = string(fpath, "showcase/")
if !isdir(savedir)
    mkdir(savedir)
end

bson(string(savedir, "lcp_vanilla_mass_154nvl_177nvm_best.bson"), data)
