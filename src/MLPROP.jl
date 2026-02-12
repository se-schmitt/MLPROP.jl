module MLPROP

using JLD2, ConcreteStructs, LinearAlgebra, Random
using Clapeyron, Lux, ChemBERTa

const CL = Clapeyron

const kB = 1.380649e-23
const NA = 6.02214076e23
const R = kB*NA

const DB_PATH = normpath(Base.pkgdir(MLPROP),"database")

BERT = nothing

# Models
include("models/models.jl")


end
