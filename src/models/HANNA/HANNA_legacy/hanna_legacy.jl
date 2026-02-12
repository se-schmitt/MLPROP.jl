# Help Functions ---------------------------------------------------------------------------
silu(x) = @. x/(1+exp(-x))

struct TScaler <: Function
    u::Float64
    sk::Float64
end

function (t::TScaler)(x)
    (x - t.u) / t.sk
end

struct EmbeddedScaler{D} <: Function
    u::D
    s::D
    k::D
end

function (t::EmbeddedScaler)(x)
    (x .- @view(t.u[2:end])) ./  @view(t.s[2:end]) .*  @view(t.k[2:end])
end

function load_scalers()
    db_path = joinpath(@__DIR__, "data", "scaler")
    path = joinpath(db_path, "scaler.csv")
    data = CSV.File(path; header=1) |> CSV.Tables.matrix
    (u,s,k) = [data[:,i][:] for i in 1:3]
    T_scaler = TScaler(u[1],s[1]/k[1])
    emb_scaler = EmbeddedScaler(u,s,k)
    return T_scaler, emb_scaler
end

function cosine_similarity(x1,x2;eps=1e-8)
    ∑x1 = sqrt(dot(x1,x1))
    ∑x2 = sqrt(dot(x2,x2))
    return dot(x1,x2)/(max(∑x1,eps*one(∑x1))*max(∑x2,eps*one(∑x2)))
end

struct LuxNetwork
    model::Lux.AbstractLuxLayer
    ps::NamedTuple
    st::NamedTuple
end

function build_lux_dense(in_d, out_d, path_w, path_b)
    rng = Random.default_rng()
    model = Dense(in_d, out_d, silu)
    ps, st = Lux.setup(rng, model)
    
    # Laden & Konvertieren zu Float64 (Thermodynamik braucht Präzision!)
    w = CSV.File(path_w; header=0) |> CSV.Tables.matrix
    b = vec(readdlm(path_b, ',', Float64))
    
    # Wir erstellen ein neues NamedTuple für die Parameter
    ps_loaded = (weight=Float64.(w), bias=Float64.(b))
    
    return LuxNetwork(model, ps_loaded, st)
end

function build_lux_chain(layers, paths_w, paths_b)
    rng = Random.default_rng()
    model = Chain(layers...)
    ps, st = Lux.setup(rng, model)
    
    ps_dict = Dict{Symbol, Any}()
    
    for (i, (pw, pb)) in enumerate(zip(paths_w, paths_b))
        w = CSV.File(pw; header=0) |> CSV.Tables.matrix
        b = vec(readdlm(pb, ',', Float64))
        key = Symbol("layer_$i")
        ps_dict[key] = (weight=Float64.(w), bias=Float64.(b))
    end
    
    keys_sorted = sort(collect(keys(ps_dict)))
    vals_sorted = [ps_dict[k] for k in keys_sorted]
    ps_loaded = NamedTuple{Tuple(keys_sorted)}(Tuple(vals_sorted))

    return LuxNetwork(model, ps_loaded, st)
end

# Help Functions - END------------------------------------------------------------

# Parameter Structure
struct HANNAParam <: CL.EoSParam
    smiles::SingleParam{String}        
    emb_scaled::Vector{Vector{Float64}}
    T_scaler::Function
    # Lux architecture
    theta::LuxNetwork
    alpha::LuxNetwork
    phi::LuxNetwork

    Mw::SingleParam{Float64}            
end

# Constants for Lux model
const N_EMB = 384
const N_NODES = 96

function CL.split_model(param::HANNAParam, splitter)
    return [CL.each_split_model(param, i) for i ∈ splitter]
end

function CL.each_split_model(param::HANNAParam, i)
    Mw = CL.each_split_model(param.Mw, i)
    smiles = CL.each_split_model(param.smiles, i)
    
    emb_subset = param.emb_scaled[i]
    if !isa(emb_subset, Vector{Vector{Float64}})
        emb_subset = [emb_subset]
    end

    return HANNAParam(smiles, emb_subset, param.T_scaler, 
        param.theta, param.alpha, param.phi, Mw)
end

# Model definition
abstract type HANNAModel <: CL.ActivityModel 
end

# Model Structure
struct HANNA{c<:CL.EoSModel} <: HANNAModel
    components::Array{String,1}
    params::HANNAParam
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
end

CL.default_locations(::Type{HANNA}) = ["properties/identifiers.csv", "properties/molarmass.csv"]
# Main HANNA function --------------------------------------------
function HANNA(components::Vector{String};
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing)

    DB_PATH = joinpath(@__DIR__, "data") 

    # Get parameters (Mw and smiles)
    params = CL.getparams(components,CL.default_locations(HANNA);userlocations=userlocations,ignore_headers=["dipprnumber","inchikey","cas"])

    # Load ChemBERTa model
    bert = ChemBERTa.load()
    embs = Vector{Vector{Float64}}(undef, length(components))
    loaded_smiles = params["canonicalsmiles"].values
    for i in eachindex(components)
        smi = loaded_smiles[i]
        embs[i] = bert(smi)
    end
    
    # Load scalers and scale embeddings
    T_scaler, emb_scaler = load_scalers()
    emb_scaled = emb_scaler.(embs)

    theta = build_lux_dense(N_EMB, N_NODES, 
                            joinpath(DB_PATH, "HANNA", "theta_1_w.csv"), 
                            joinpath(DB_PATH, "HANNA", "theta_1_b.csv"))

    alpha = build_lux_chain(
        [Dense(N_NODES + 2, N_NODES, silu), Dense(N_NODES, N_NODES, silu)],
        [joinpath(DB_PATH, "HANNA", "alpha_1_w.csv"), joinpath(DB_PATH, "HANNA", "alpha_2_w.csv")],
        [joinpath(DB_PATH, "HANNA", "alpha_1_b.csv"), joinpath(DB_PATH, "HANNA", "alpha_2_b.csv")]
    )

    phi = build_lux_chain(
        [Dense(N_NODES, N_NODES, silu), Dense(N_NODES, 1)],
        [joinpath(DB_PATH, "HANNA", "phi_1_w.csv"), joinpath(DB_PATH, "HANNA", "phi_2_w.csv")],
        [joinpath(DB_PATH, "HANNA", "phi_1_b.csv"), joinpath(DB_PATH, "HANNA", "phi_2_b.csv")]
    )

    params = HANNAParam(params["canonicalsmiles"], emb_scaled, T_scaler, theta, alpha, phi, params["Mw"])
    
    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String["10.1039/D4SC05115G"]
    model = HANNA(components, params, _puremodel, references)
    CL.set_reference_state!(model,reference_state,verbose = verbose)

    #println("SMILES im Modell: ", model.params.smiles.values)
    return model
end


function CL.excess_gibbs_free_energy(model::HANNA, p, T, z)
    x = z ./ sum(z)
    #Alias
    params = model.params
    # Scale input (T and embs)
    T_s = params.T_scaler(T)
    # Fine tuning of the component embeddings
    θ_i = [first(params.theta.model(emb, params.theta.ps, params.theta.st)) for emb in params.emb_scaled]
    gE = zero(Base.promote_eltype(T_s,x))
    n = length(model)
    for i in 1:n 
        for j in (i+1):n
            # Calculate cosine similarity and distance between the two components
            cosine_sim_ij = cosine_similarity(θ_i[i],θ_i[j])
            cosine_dist_ij = 1.0 - cosine_sim_ij

            # Concatenate embeddings with T and x
            input_i = vcat(T_s, x[i], θ_i[i]) 
            input_j = vcat(T_s, x[j], θ_i[j])

            out_a_i = first(params.alpha.model(input_i, params.alpha.ps, params.alpha.st))
            out_a_j = first(params.alpha.model(input_j, params.alpha.ps, params.alpha.st))
            
            c_mix = out_a_i .+ out_a_j
            
            gE_NN = first(params.phi.model(c_mix, params.phi.ps, params.phi.st))[1]

            # Apply cosine similarity adjustment
            gE += x[i]*x[j]*gE_NN*cosine_dist_ij
        end
    end
    return gE * Rgas(model) * T * sum(z)
end

export HANNA