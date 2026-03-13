module MLThermoProperties

using JLD2, ConcreteStructs, LinearAlgebra, Random
using Clapeyron, Lux, ChemBERTa, JLD2, EntropyScaling

const CL = Clapeyron
const ES = EntropyScaling

const kB = 1.380649e-23
const NA = 6.02214076e23
const R = kB*NA

const DB_PATH = normpath(Base.pkgdir(MLThermoProperties),"database")

BERT = nothing

# Layers
include("layers/layers.jl")

# Models
include("models/models.jl")

end
