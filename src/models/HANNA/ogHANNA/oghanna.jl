abstract type ogHANNAModel <: CL.ActivityModel end

struct ogHANNAParam{T,P,S} <: CL.EoSParam
    emb::SingleParam{Vector{T}}
    scaler_T::AbstractScaler{T}
    nn::ogHANNALux
    ps::P
    st::S
    Mw::SingleParam{T}
end

struct ogHANNA{c<:CL.EoSModel,T,P,S} <: ogHANNAModel
    components::Array{String,1}
    params::ogHANNAParam{T,P,S}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
end

"""
    ogHANNA <: ActivityModel

    ogHANNA(components;
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
`ogHANNA` was trained on all available binary VLE data (up to 10 bar) and limiting activity coefficients from the Dortmund Data Bank. `ogHANNA` was only developed for binary mixtures. Use `HANNA` for multicomponent mixtures.

## Example
```julia
using MLThermoProperties, Clapeyron

components = ["water","isobutanol"]
Mw = [18.01528, 74.1216]
smiles = ["O", "CC(C)CO"]

model = ogHANNA(components,userlocations=(;Mw=Mw, SMILWS=smiles))
# model = ogHANNA(components) # also works if components are in the database 
```

## References
1. Specht, T., Nagda, M., Fellenz, S., Mandt, S., Hasse, H., Jirasek, F., HANNA: Hard-Constraint Neural Network for Consistent Activity Coefficient Prediction. Chemical Science 2024. [10.1039/D4SC05115G](https://doi.org/10.1039/D4SC05115G).
"""
ogHANNA

CL.default_locations(::Type{ogHANNA}) = ["properties/identifiers.csv", "properties/molarmass.csv"]
get_model_path(::Type{ogHANNA}) = joinpath(DB_PATH, "ogHANNA")

function ogHANNA(components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing
)
    _components = CL.format_components(components)
    
    _params = CL.getparams(components,CL.default_locations(ogHANNA);
        userlocations,ignore_headers=["dipprnumber","inchikey","cas"], ignore_missing_singleparams=["canonicalsmiles", "Mw"])

    length(_components) > 2 && error("`ogHANNA` is not suited for multicomponent systems. Use `HANNA` instead.")
    smiles = [
        _params["canonicalsmiles"].ismissingvalues[i] ?
        ChemBERTa.canonicalize.(_params["SMILES"].values[i]) :
        _params["canonicalsmiles"].values[i]
    for i in eachindex(_components)]

    # Create model
    N_EMB = 384
    N_NODES = 96
    nn = ogHANNALux(
        Dense(N_EMB, N_NODES, silu),
        Chain(Dense(N_NODES + 2, N_NODES, silu), Dense(N_NODES, N_NODES, silu)),
        Chain(Dense(N_NODES, N_NODES, silu), Dense(N_NODES, 1))
    )
    
    # Load model parameters and scalers
    ps, st = load(joinpath(get_model_path(ogHANNA),"parameters_states.jld2"), "ps", "st")
    scaler_T =   load_scaler(joinpath(get_model_path(ogHANNA), "scaler_T.jld2"))
    scaler_emb = load_scaler(joinpath(get_model_path(ogHANNA), "scaler_emb.jld2"))

    # Calc embeddings
    if isnothing(BERT)
        global BERT = ChemBERTa.load()
    end
    emb = SingleParam("ChemBERTa embedding", _components, scale.(scaler_emb, BERT.(smiles; is_canonical=true)))

    params = ogHANNAParam(emb, scaler_T, nn, ps, st, _params["Mw"])
    
    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String["10.1039/D4SC05115G"]
    model = ogHANNA(components, params, _puremodel, references)
    CL.set_reference_state!(model,reference_state,verbose = verbose)

    return model
end

function CL.excess_gibbs_free_energy(model::ogHANNA, p, T, z)
    x = z ./ sum(z)
    
    params = model.params
    Ts = scale(params.scaler_T, T)
    gE = params.nn((Ts,x,params.emb.values), params.ps, params.st)
    
    return gE * Rgas(model) * T * sum(z)
end

export ogHANNA