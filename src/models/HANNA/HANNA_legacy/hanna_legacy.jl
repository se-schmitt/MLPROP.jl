abstract type HANNAModel <: CL.ActivityModel end

struct HANNAParam{T,P,S} <: CL.EoSParam
    emb::Matrix{T}
    scaler_T::AbstractScaler{T}
    nn::LuxHANNA
    ps::P
    st::S
    Mw::SingleParam{T}
end

# Constants for Lux model
const N_EMB = 384
const N_NODES = 96

function CL.split_model(param::HANNAParam, splitter)
    return [CL.each_split_model(param, i) for i ∈ splitter]
end

function CL.each_split_model(param::HANNAParam, i)
    Mw = CL.each_split_model(param.Mw, i)
    
    emb_subset = param.emb_scaled[:,i:i]

    return HANNAParam(emb_subset, param.T_scaler, param.model, param.ps, param.st, Mw)
end

struct HANNA{c<:CL.EoSModel,T,P,S} <: HANNAModel
    components::Array{String,1}
    params::HANNAParam{T,P,S}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
end

"""
    HANNA <: ActivityModel

    HANNA(components;
    puremodel = nothing,
    userlocations = String[],
    pure_userlocations = String[],
    verbose = false,
    reference_state = nothing)

## Input parameters
- `SMILES`: canonical SMILES (using RDKit) representation of the components
- `Mw`: Single Parameter (`Float64`) (Optional) - Molecular Weight `[g·mol⁻¹]`

## Input models
- `puremodel`: model to calculate pure pressure-dependent properties

## Description
Hard-Constraint Neural Network for Consistent Activity Coefficient Prediction (HANNA v1.0.0).
The implementation is based on [this](https://github.com/tspecht93/HANNA) Github repository.
HANNA was trained on all available binary VLE data (up to 10 bar) and limiting activity coefficients from the Dortmund Data Bank. HANNA was only tested for binary mixtures so far. The extension to multicomponent mixtures is experimental.

To use the model, the package `ClapeyronHANNA` must be installed and loaded (see example below).

## Example
```julia
using MLPROP, Clapeyron

components = ["water","isobutanol"]
Mw = [18.01528, 74.1216]
smiles = ["O", "CC(C)CO"]

model = HANNA(components,userlocations=(;Mw=Mw, SMILWS=smiles))
# model = HANNA(components) # also works if components are in the database 
```

## References
1. Specht, T., Nagda, M., Fellenz, S., Mandt, S., Hasse, H., Jirasek, F., HANNA: Hard-Constraint Neural Network for Consistent Activity Coefficient Prediction. Chemical Science 2024. [10.1039/D4SC05115G](https://doi.org/10.1039/D4SC05115G).
"""
HANNA

CL.default_locations(::Type{HANNA}) = ["properties/identifiers.csv", "properties/molarmass.csv"]
get_model_path(::Type{HANNA}) = joinpath(DB_PATH, "HANNA_legacy")

function HANNA(components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing
)
    _components = CL.format_components(components)
    
    _params = CL.getparams(components,CL.default_locations(HANNA);
        userlocations,ignore_headers=["dipprnumber","inchikey","cas"], ignore_missing_singleparams=["canonicalsmiles", "Mw"])

    length(_components) > 2 && error("`HANNA` is not suited for multicomponent systems. Use `HANNA` instead.")
    smiles = [
        _params["canonicalsmiles"].ismissingvalues[i] ?
        ChemBERTa.canonicalize.(_params["SMILES"].values[i]) :
        _params["canonicalsmiles"].values[i]
    for i in eachindex(_components)]

    # Create model
    nn = LuxHANNA(
        Dense(N_EMB, N_NODES, silu),
        Chain(Dense(N_NODES + 2, N_NODES, silu), Dense(N_NODES, N_NODES, silu)),
        Chain(Dense(N_NODES, N_NODES, silu), Dense(N_NODES, 1))
    )
    
    # Load model parameters and scalers
    ps, st = load(joinpath(get_model_path(HANNA),"hanna_legacy.jld2"), "ps", "st")
    scaler_T =   load_scaler(joinpath(get_model_path(HANNA), "scaler_T.jld2"))
    scaler_emb = load_scaler(joinpath(get_model_path(HANNA), "scaler_emb.jld2"))

    # Calc embeddings
    if isnothing(BERT)
        global BERT = ChemBERTa.load()
    end
    emb = hcat(BERT.(smiles; is_canonical=true)...)

    params = HANNAParam(scale(scaler_emb, emb), scaler_T, nn, ps, st, _params["Mw"])
    
    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String["10.1039/D4SC05115G"]
    model = HANNA(components, params, _puremodel, references)
    CL.set_reference_state!(model,reference_state,verbose = verbose)

    return model
end

function CL.excess_gibbs_free_energy(model::HANNA, p, T, z)
    x = z ./ sum(z)
    
    params = model.params
    Ts = scale(params.scaler_T, T)
    gE = params.nn((Ts,x,params.emb), params.ps, params.st)
    
    return gE * Rgas(model) * T * sum(z)
end

export HANNA