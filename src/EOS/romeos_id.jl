abstract type RoMEoSIdealModel <: IdealModel end

@concrete struct RoMEoSIdealParam{SingleParam} <: EoSParam
    Es
    reference_state::ReferenceState
    Mw::SingleParam
end

@concrete struct RoMEoSIdeal{RoMEoSIdealParam, StatefulLuxLayer} <: RoMEoSIdealModel 
    components::Vector{<:AbstractString}
    params::RoMEoSIdealParam
    smodel::StatefulLuxLayer
    scaler
    references::Vector{<:AbstractString}
end

"""
    RoMEoSIdeal <: IdealModel

    RoMEoSIdeal(components;
    userlocations = String[],
    reference_state = nothing,
    verbose = false)

## Input parameters
None

## Description
Ideal model based on data from quantum mechanical (QM) simulations from *Gond et al. (2025)*.

## Model Construction Examples
```
idealmodel = RoMEoSIdeal("water")
idealmodel = RoMEoSIdeal(["water","carbon dioxide"])
```

## References
1. Gond, D., ...
"""
RoMEoSIdeal

export RoMEoSIdeal 

CL.Rgas(::RoMEoSIdeal) = R̄32

function RoMEoSIdeal(fn::String, smiles, E; ref_state=nothing)

    lmodel, ps, st, _, _ = load_model(fn)
    smodel = StatefulLuxLayer{true}(lmodel, ps, Lux.testmode(st))

    return RoMEoSIdeal(smodel, smiles, E; ref_state)
end

function RoMEoSIdeal(smodel::StatefulLuxLayer, smiles, E; ref_state=nothing)

    scaler = get_scaler(smodel)

    # Get parameters (Mw and smiles)
    mol = RDK.get_mol(smiles)
    descs = RDK.get_descriptors(mol)
    params = RoMEoSIdealParam(
        scale(scaler.emb, E),
        Clapeyron.__init_reference_state_kw(ref_state),
        SingleParam("molecular weight",[smiles],Float32[descs["amw"]]),
    )

    # Create model
    model = RoMEoSIdeal([smiles], params, smodel, scaler, String[])

    return model
end

function CL.a_ideal(model::RoMEoSIdeal, V, T, z)
    ∑z = sum(z)
    V⁻¹ = 1/V
    ϱ = ∑z * V⁻¹ 

    X = [
        scale(model.scaler.ϱ,ϱ);
        scale(model.scaler.T,T);
        model.params.Mw[1].*1f-3;
        model.params.Es;;
    ]

    return ∑z*first(model.smodel(X))/Rgas(model)/T
end