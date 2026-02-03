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

CANONICALIZE_COUNT = 0

using DataStructures, ConcreteStructs, JSON, Random, SafeTensors

using Lux, NNlib

# External packages
using BangBang, Tricks, FuncPipelines, StructWalk
using OneHotArrays: OneHotArray

# Text processing packages
using TextEncodeBase
using TextEncodeBase: WordTokenization, nested2batch, nestedcall, with_head_tail, tokenize, join_text,
    trunc_and_pad, trunc_or_pad, Batch, Sentence, Document, peek_sequence_sample_type,
    BaseTokenization, WrappedTokenization, MatchTokenization, Splittable, CodeNormalizer,
    CodeMap, CodeUnMap, ParentStages, TokenStages, SentenceStage, SubSentenceStage,
    WordStage, SubWordStage, TokenStage, DocumentStage, getvalue, getmeta,
    SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm, AbstractTokenizer, AbstractTokenization,
    EachMatchTokenization, EachSplitTokenization, MatchSplitsTokenization, RuRegex

# Neural attention library
using NeuralAttentionlib: LengthMask, RevLengthMask

# Init
const DATADIR = joinpath(pkgdir(@__MODULE__), "data")
rng = Random.default_rng()

include("utils.jl")
include("api.jl")

include("tokenizer/textencoder.jl")
include("tokenizer/tokenizer.jl")
include("tokenizer/utils.jl")

include("model/bert.jl")
include("model/transformer_encoder.jl")

end # module ChemBERTa
