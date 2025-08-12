abstract type RoMEoSModel <: EoSModel end

@concrete struct RoMEoSParam{SingleParam} <: EoSParam
    Es
    reference_state::ReferenceState
    Mw::SingleParam
end

@concrete struct RoMEoS{RoMEoSParam, StatefulLuxLayer} <: RoMEoSModel 
    components::Vector{<:AbstractString}
    params::RoMEoSParam
    smodel::StatefulLuxLayer
    idealmodel
    scaler
    references::Vector{<:AbstractString}
end

"""
    RoMEoS <: EoSModel

    RoMEoS(components;
    userlocations = String[],
    reference_state = nothing,
    verbose = false)

## Input parameters
None

## Description
Robust Machine learned Equation of State (RoMEoS).

## Model Construction Examples
```
model = RoMEoS("water")
model = RoMEoS(["water","carbon dioxide"])
```

## References
1. ...
"""
RoMEoS

export RoMEoS

CL.Rgas(::RoMEoS) = R̄32

function RoMEoS(fn, smiles, E; ref_state = nothing, verbose = false)

    lmodel, ps, st, _, _ = load_model(fn)

    # Get parameters (Mw and smiles)
    mol = RDK.get_mol(smiles)
    descs = RDK.get_descriptors(mol)
    params = RoMEoSIdealParam(
        scale(lmodel.scaler.emb, E),
        Clapeyron.__init_reference_state_kw(ref_state),
        SingleParam("molecular weight",[smiles],Float32[descs["amw"]]),
    )


end