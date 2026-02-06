abstract type GRAPPAModel{T} <: CL.SaturationModel end

struct GRAPPAParam{T} <: CL.ParametricEoSParam{T} 
    Tc::SingleParam{T}
    A::SingleParam{T}
    B::SingleParam{T}
    C::SingleParam{T}
end

struct GRAPPA{T} <: GRAPPAModel{T}
    components::Array{String, 1}
    params::GRAPPAParam{T}
    references::Array{String, 1}
end

"""
    GRAPPA{T} <: SaturationModel
    
    GRAPPA(
        components;
        userlocations = String[],
        verbose::Bool=false
    )

## Description

GRAPPA model for calculating vapor pressure of pure components based on the Antoine equation.
On model construction, the Antoine parameters are predicted using a Python implementation the GRAPPA model.

!!! info "Requires loading the package `PythonCall.jl`"
    `GRAPPA` uses a modified Python implementation taken from https://github.com/marco-hoffmann/GRAPPA.
    Therefore to use the GRAPPA model, you need to install and load the package `PythonCall.jl` by
    ```julia
    using Pkg; Pkg.add("PythonCall")    # Installation
    using PythonCall                    # Loading
    ```
For predicting the Antoine parameters, only the smiles of the molecule is required.
It will automatically be retrieved from the `Clapeyron.jl` database.
The smiles can also be provided by the `userlocations` keyword (see example below). 

## Example

```julia
using Clapeyron, PythonCall

model = GRAPPA("propanol")
model = GRAPPA("propanol"; userlocations=(; smiles="CCCO"))

ps, _, _ = saturation_pressure(model, 300.)         # Vapor pressure at 300 K
```

## References

1.  M. Hoffmann, H. Hasse, and F. Jirasek: GRAPPA—A Hybrid Graph Neural Network for Predicting Pure Component Vapor Pressures, Chemical Engineering Journal Advances 22 (2025) 100750, DOI: https://doi.org/10.1016/j.ceja.2025.100750.

"""
GRAPPA

CL.default_locations(::Type{GRAPPA}) = ["properties/critical.csv","properties/identifiers.csv",]

function _GRAPPA_error(args...; kwargs...)
    error("""
    To use GRAPPA, `PythonCall` needs to be installed and loaded! This can be done by:
        using Pkg; Pkg.add("PythonCall")
        using PythonCall
    """)
    return nothing
end

const _GRAPPA = Ref{Function}(_GRAPPA_error)

function GRAPPA(components; userlocations = String[], reference_state = nothing, verbose = false)
    _GRAPPA[](components; userlocations, reference_state, verbose)
end

function CL.crit_pure(model::GRAPPAModel)
    CL.single_component_check(crit_pure,model)
    Tc = only(model.params.Tc.values)
    Pc, _, _ = saturation_pressure(model, Tc)
    return (Tc,Pc,NaN)
end

function CL.saturation_pressure_impl(model::GRAPPAModel, T, ::CL.SaturationCorrelation)
    nan = zero(T)/zero(T)
    Tc = only(model.params.Tc.ismissingvalues) ? nan : only(model.params.Tc.values)
    A = only(model.params.A.values)
    B = only(model.params.B.values)
    C = only(model.params.C.values)

    !isnan(Tc) && T > Tc && (return nan,nan,nan)
    psat = exp(A - B/(T + C)) * 1000
    return psat,nan,nan
end

export GRAPPA