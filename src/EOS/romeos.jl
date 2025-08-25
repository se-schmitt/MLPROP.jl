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
    scaler = MLPROP_EOS.get_scaler(lmodel)

    # Ideal model
    smodel_id = StatefulLuxLayer{true}(lmodel.id, ps.id, st.id)
    model_id = RoMEoSIdeal(smodel_id, smiles, E; ref_state)

    # Get parameters (Mw and smiles)
    params = RoMEoSParam(
        scale(scaler.emb, E),
        Clapeyron.__init_reference_state_kw(ref_state),
        SingleParam("molecular weight",[smiles],Float32[descs["amw"]]),
    )

    # Create model
    smodel_res = StatefulLuxLayer{true}(lmodel.res, ps.res, st.res)
    model = RoMEoS([smiles], params, smodel_res, model_id, scaler, String[])

    return model
end

function CL.a_res(model::RoMEoS, V, T, z)
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