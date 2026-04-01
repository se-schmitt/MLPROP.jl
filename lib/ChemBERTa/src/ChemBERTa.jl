"""
    ChembBERTa

Small sub-package for the ChemBERTa model applied in the MLPROP models.

## Usage

```julia
using ChemBERTa
smiles = "CCCO"

# Loading the ChemBERTa model
bert = ChemBERTa.load()

# Calculating the embedding for a given SMILES
embedding = bert(smiles)
```

## Documentation

The ChemBERTa model is a customized model based on [ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR).
The tokenizer is different from the original version as the it fixes some errors.
See XY for details.
"""
module ChemBERTa

using DataStructures: OrderedDict
using ConcreteStructs, JSON, Random, SafeTensors

using Lux, NNlib

# Init
const DATADIR = joinpath(pkgdir(@__MODULE__), "data")
rng = Random.default_rng()

include("utils.jl")
include("api.jl")

include("tokenizer/tokenizer.jl")

include("model/bert.jl")
include("model/transformer_encoder.jl")

end # module ChemBERTa
