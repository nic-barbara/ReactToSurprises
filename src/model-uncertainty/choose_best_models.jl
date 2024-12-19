using BSON
using CairoMakie
using LinearAlgebra
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "functions.jl"))
include(joinpath(@__DIR__, "setup_mass.jl"))


fpath = joinpath(@__DIR__, "../../results/model-uncertainty/batch/")
get_fname(id) = string(fpath, "lcp_mass_84nx_128nv_v$(id).bson")
load_data(id, key) = BSON.load(get_fname(id))[key]


data = BSON.load(get_fname(1))

data["youla_ren"] = load_data(2, "youla_ren")
data["costs_yr"] = load_data(2, "costs_yr")

data["youla_lren"] = load_data(2, "youla_lren")
data["costs_lr"] = load_data(2, "costs_lr")

data["youla_ren_nf"] = load_data(2, "youla_ren_nf")
data["costs_yr_nf"] = load_data(2, "costs_yr_nf")

data["fdbak_ren"] = load_data(2, "fdbak_ren")
data["costs_fr"] = load_data(2, "costs_fr")

data["fdbak_lstm"] = load_data(2, "fdbak_lstm")
data["costs_fl"] = load_data(2, "costs_fl")


savedir = string(fpath, "showcase/")
if !isdir(savedir)
    mkdir(savedir)
end

bson(string(savedir, "lcp_mass_84nx_128nv_best.bson"), data)
