abstract type ESEModel <: ES.AbstractTransportPropertyModel end

struct ESEParam{T}
    b_ij::Matrix{T}            
    Mw::SingleParam{T}
end

struct ESE{T,M} <: ESEModel
    components::Array{String,1}
    params::ESEParam{T}
    vismodel::M
    references::Array{String,1}
end

@kwdef @concrete struct ESELux <: AbstractLuxWrapperLayer{:layer}
    layer = Chain(Dense(12 => 32, relu), Dense(32 => 16, relu), Dense(16 => 1, softplus))
end

"""
    ESE <: AbstractTransportPropertyModel

    ESE(components;
    vismodel = nothing,
    userlocations = String[],
    vis_userlocations = String[],
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
ESE 

CL.default_locations(::Type{ESE}) = ["properties/identifiers.csv", "properties/molarmass.csv"]
get_model_path(::Type{ESE}) = joinpath(DB_PATH, "ESE")

function ESE(components;
        vismodel = RefpropRESModel,             # TODO switch to GC model,
        userlocations = String[],
        vis_userlocations = String[],
        verbose = false,
)

    # loading SMILES und Parameter
    _components = CL.format_components(components)
    N_comps = length(_components)
    
    _params = CL.getparams(components,CL.default_locations(ESE);
        userlocations,ignore_headers=["dipprnumber","inchikey","cas","canonicalsmiles","Mw"])

    smiles = _params["SMILES"].values
    descs = get_descriptors.(smiles)

    nn = ESELux()
    ps, st = load(joinpath(get_model_path(ESE),"parameters_states_ensemble.jld2"), "ps", "st")
    st = Lux.testmode.(st)
    N_ensemble = length(ps)

    Xs = get_ese_X.(smiles)
    _X2 = zeros(12,1)
    b_ij = zeros(N_comps, N_comps)

    for i in 1:N_comps, j in 1:N_comps
        if i != j
            _X2[1:6] .= Xs[i]
            _X2[7:12] .= Xs[j]
            for (psᵢ, stᵢ) in zip(ps, st)
                b_ij[i,j] += only(first(nn(_X2, psᵢ, stᵢ))) ./ N_ensemble
            end
        end
    end

    params = ESEParam(b_ij, SingleParam("Mw",_components,first.(Xs)*1e3))
    _vismodel = PureModelContainer(Viscosity(), vismodel, _components; userlocations=vis_userlocations, verbose)
    references = String["10.48550/arXiv.2603.02761"]

    return ESE(_components, params, _vismodel, references)
end

function get_ese_X(smiles)
    desc = get_descriptors(smiles)
    is_water = smiles in ["O", "[2H]O[2H]"]

    X = [
        desc["exactmw"] * 1e-3,
        is_water ? 0.5 : desc["NumHBA"] / desc["NumHeavyAtoms"],
        is_water ? 0.5 : desc["NumHBD"] / desc["NumHeavyAtoms"],
        desc["NumHeteroatoms"] / desc["NumHeavyAtoms"],
        desc["NumHalogens"] / desc["NumHeavyAtoms"],
        desc["NumRings"] != 0,
    ]

    return X
end

Base.broadcastable(x::ESE) = Ref(x)

function ES._inf_diffusion_coefficient(model::ESE, p, T, (idx_i,idx_j); phase=:liquid)
    TT = Base.promote_eltype(p,T)

    # Initialitzing constants required for Stokes-Einstein-equation
    f = 0.64
    ϱ_ref = 1050.
    M_i = model.params.Mw.values[idx_i]*1e-3
    η_j = viscosity(model.vismodel, p, T, idx_j; phase)

    r_i = cbrt(f * 3 * M_i / (4π * ϱ_ref * NA))
    Dij_SE = (kB*T)/(6π*η_j*r_i)
    return Dij_SE * model.params.b_ij[idx_i,idx_j]
end

export ESE