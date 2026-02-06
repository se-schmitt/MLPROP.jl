module MLPROP

using Clapeyron, Lux, ConcreteStructs, ChemBERTa, JLD2

const CL = Clapeyron

const DB_PATH = normpath(Base.pkgdir(MLPROP),"database")

include("models/models.jl")

end
