abstract type multHANNAModel <: CL.ActivityModel end

struct multHANNAParam{T,P,S} <: CL.EoSParam
    emb::Matrix{T}
    scaler_T::AbstractScaler{T}
    nn::multHANNALux      
    ps::P                 
    st::S                 
    Mw::SingleParam{T}
    gamma::T              # for RBF-equation
end

function CL.split_model(param::multHANNAParam, splitter)
    return [CL.each_split_model(param, i) for i ∈ splitter]
end

function CL.each_split_model(param::multHANNAParam, i)
    Mw = CL.each_split_model(param.Mw, i)
    
    emb_subset = param.emb[:,i:i]   # check this if emb or emb_scaled

    return multHANNAParam(emb_subset, param.scaler_T, param.nn, param.ps, param.st, Mw, param.gamma)
end


struct multHANNA{c<:CL.EoSModel,T,P,S} <: multHANNAModel
    components::Array{String,1}
    params::multHANNAParam{T,P,S}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
end

"""
## Explanation
## Example
"""

CL.default_locations(::Type{multHANNA}) = ["properties/identifiers.csv", "properties/molarmass.csv"]
get_model_path(::Type{multHANNA}) = joinpath(DB_PATH, "multHANNA")

function multHANNA(components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing
)

    # loading SMILES und Parameter
    _components = CL.format_components(components)
    
    _params = CL.getparams(components,CL.default_locations(multHANNA);
        userlocations,ignore_headers=["dipprnumber","inchikey","cas"], ignore_missing_singleparams=["canonicalsmiles", "Mw"])

    length(_components) < 3 && @warn "`ogHANNA` might be more recommended for binary systems. Proceeding with `multHANNA`..."
    smiles = [
        _params["canonicalsmiles"].ismissingvalues[i] ?
        ChemBERTa.canonicalize.(_params["SMILES"].values[i]) :
        _params["canonicalsmiles"].values[i]
    for i in eachindex(_components)]

    # Create model
    N_EMB = 384
    N_NODES = 96
    
    nn = multHANNALux(
        # theta
        Chain(LipschitzLinear(N_EMB, N_NODES), silu),
        # alpha
        Chain(LipschitzLinear(N_NODES + 2, N_NODES), silu, 
            LipschitzLinear(N_NODES, N_NODES), silu),
        # phi
        Chain(LipschitzLinear(N_NODES, N_NODES), silu, 
            LipschitzLinear(N_NODES, 1))
    )
    
    # load parameters and scalers
    ps, st = load(joinpath(get_model_path(multHANNA),"parameters_states_all.jld2"), "ps", "st")
    scaler_T =   load_scaler(joinpath(get_model_path(multHANNA), "scaler_T.jld2"))
    scaler_emb = load_scaler(joinpath(get_model_path(multHANNA), "scaler_emb.jld2"))

    # Calc embeddings
    if isnothing(BERT)
        global BERT = ChemBERTa.load()
    end
    emb = hcat(BERT.(smiles; is_canonical=true)...)

    params = multHANNAParam(scale(scaler_emb, emb), scaler_T, nn, ps, st, _params["Mw"], 100.0)
    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String["10.48550/arXiv.2509.06484"]

    model = multHANNA(components, params, _puremodel, references)
    CL.set_reference_state!(model, reference_state, verbose = verbose)

    return model
end


function CL.excess_gibbs_free_energy(model::multHANNAModel, p, T, z)
    x = z ./ sum(z) 
    
    params = model.params
    # Embeddings and RBF-Gamma
    embs = params.emb
    gamma = params.gamma
    
    # all_ps and all_st contains all parameters of all 10 ensemble models
    all_ps = params.ps
    all_st = params.st
    
    T_scaled = scale(params.scaler_T, T) 
    
    # loop over all ensemble models
    gE_sum = zero(eltype(x)) 
    num_models = length(all_ps)
    
    for i in 1:num_models
        gE_sum += model.params.nn((T_scaled, x, embs), gamma, all_ps[i], all_st[i])
    end
    
    gE_mean_dim_less = gE_sum / num_models
    
    return gE_mean_dim_less * Rgas(model) * T * sum(z)
end

export multHANNA
