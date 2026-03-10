abstract type ESEModel <: ES.AbstractTransportPropertyModel end

struct ESEParam{T,P,S}
    b_ij::Matrix{T}
    nn::multHANNALux      
    ps::P                 
    st::S                 
    Mw::SingleParam{T}
end

struct ESE{M,T,P,S} <: ESEModel
    components::Array{String,1}
    params::ESEParam{T,P,S}
    vismodel::M
    references::Array{String,1}
end

"""
    ESE <: AbstractTransportPropertyModel

    ESE(components;
    vismodel = nothing,
    userlocations = String[],
    pure_userlocations = String[],
    verbose = false)

## Input parameters
- `SMILES`: canonical SMILES (using RDKit) representation of the components
- `Mw`: single parameter (`Float64`) (Optional) - Molecular Weight `[g·mol⁻¹]`
- `vismodel`: viscosity model 

## Description

ESE model for calculating diffusion coefficients at infinite dilution.
The diffusion coefficient at infinite dilution can be calculated by calling [`inf_diffusion_coefficient`](https://se-schmitt.github.io/EntropyScaling.jl/stable/).

If no viscosity model is specified, a `GCESModel` from `EntropyScaling.jl` is constructed (if possible).
A constant viscosity model can also be used if the viscosity η is knwon as `ESE(...; vismodel=ConstantModel(Viscosity(), η))`.

## Examples
```julia
using MLPROP, EntropyScaling

model = ESE(["ethanol", "acetonitrile"])
D_matrix = inf_diffusion_coefficient(model, 1e5, 300.)
D_eth = inf_diffusion_coefficient(model, 1e5, 300.; solute="ethanol", solvent="acetonitril")
```
"""
function ESE(SMILE_i::AbstractString, SMILE_j::AbstractString, eta_fun)
    #TODO use Clapeyron style
    # Loading weights and bias from nn_parameters.jld2
    path_nn_parameters = joinpath(DB_PATH, "ESE", "weights_bias_true.jld2")
    nn_parameters = load(path_nn_parameters)["Weights_Bias_ESE"]
    
    # Processing SMILES to get molecular descriptors used in neural net
    desc_i, desc_j = get_descriptors(SMILE_i), get_descriptors(SMILE_j)

    mw_i = desc_i["exactmw"] * 1e-3
    mw_j = desc_j["exactmw"] * 1e-3

    is_water_i = SMILE_i == "O" || SMILE_i == "[2H]O[2H]"
    is_water_j = SMILE_j == "O" || SMILE_j == "[2H]O[2H]"

    X_i_ini = [
        mw_i;
        is_water_i ? 0.5 : desc_i["NumHBA"] / desc_i["NumHeavyAtoms"];
        is_water_i ? 0.5 : desc_i["NumHBD"] / desc_i["NumHeavyAtoms"];
        desc_i["NumHeteroatoms"] / desc_i["NumHeavyAtoms"];
        desc_i["NumHalogens"] / desc_i["NumHeavyAtoms"];
        desc_i["NumRings"] != 0;
    ]
    X_j_ini = [
        mw_j;
        is_water_j ? 0.5 : desc_j["NumHBA"] / desc_j["NumHeavyAtoms"];
        is_water_j ? 0.5 : desc_j["NumHBD"] / desc_j["NumHeavyAtoms"];
        desc_j["NumHeteroatoms"] / desc_j["NumHeavyAtoms"];
        desc_j["NumHalogens"] / desc_j["NumHeavyAtoms"];
        desc_j["NumRings"] != 0;
    ]

    input = vcat(X_i_ini, X_j_ini)
    MW = mw_i
    b_ij_mean = 0.0

    #Initializing Neural Net using Lux
    NN = Chain(Dense(12 => 32, relu), Dense(32 => 16, relu), Dense(16 => 1, softplus))

    # Looping over parameter-sets to determine mean b_ij
    for (_, wb) in nn_parameters

        #Setting up the weights and bias of the neural net using Lux-Synatx
        st = (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple())
        ps = (
            (layer_1 = (weight = wb[2], bias = vec(wb[1]))),
            (layer_2 = (weight = wb[4], bias = vec(wb[3]))),
            (layer_3 = (weight = wb[6], bias = vec(wb[5]))),
        )
        #applying neural net with given weights and bias to calculate b_ij
        b_ij, st = NN(input, ps, st)
        b_ij_mean += only(b_ij)
    end
    b_ij_mean /= length(nn_parameters)

    # Constructing ESEParam-datastructure
    paramESE = ESEParam(MW, b_ij_mean)

    return ESE([String(SMILE_i), String(SMILE_j)], paramESE, eta_fun)
end

Base.broadcastable(x::ESE) = Ref(x)

function ES._inf_diffusion_coefficient(model::ESE, p, T, (idx_i,idx_j); phase=:unknown)
    TT = Base.promote_eltype(p,T)

    # Initialitzing constants required for Stokes-Einstein-equation
    f = 0.64
    ϱ_ref = 1050.
    M_i = model.params.Mw
    x_j = setindex!(zeros(TT,length(model)), one(TT), idx_j)
    η_j = viscosity(model.vismodel, p, T, x_j)

    r_i = cbrt(f * 3 * M_i / (4π * ϱ_ref * NA))
    Dᵢⱼ_SE = (kB*T)/(6π*η_j*r_i)
    return Dᵢⱼ_SE * model.param.b_ij[idx_i,idx_j]
end


export ESE