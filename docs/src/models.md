```@meta
CollapsedDocStrings = true
```

# [Models](@id models_page)

## HANNA

![](assets/HANNA_scheme.svg)

HANNA is a hard-constraint neural network model for the excess Gibbs energy ``g^E`` that predicts activity coefficients in a strictly thermodynamically consistent manner [specht_hanna_2024, hoffmann_thermodynamically_2026](@cite).
It only requires the SMILES of the components and the temperature as input.
The model satisfies thermodynamic boundary conditions by construction, ensuring consistency of the predicted activity coefficients.

Two versions of the model are available:
- **`ogHANNA`**: The original version (HANNA v1.0.0), trained on binary VLE data (up to 10 bar) and limiting activity coefficients from the Dortmund Data Bank [specht_hanna_2024](@cite). This version is limited to binary mixtures.
- **`HANNA`** (alias `multHANNA`): The latest version, trained on VLE and LLE data, and applicable to multi-component mixtures [hoffmann_thermodynamically_2026](@cite).

```@docs
MLThermoProperties.ogHANNA
MLThermoProperties.multHANNA
```

## mod. UNIFAC 2.0 and UNIFAC 2.0

![](assets/UNIFAC20_scheme.svg)

UNIFAC 2.0 and mod. UNIFAC 2.0 are enhanced versions of the classical group-contribution methods UNIFAC and mod. UNIFAC (Dortmund), respectively.
Missing interaction parameters are predicted using matrix completion, which significantly extends the applicability of the methods and leads to a higher prediction accuracy compared to the original versions [hayer_advancing_2025, hayer_modified_2025](@cite).
The methods for [UNIFAC 2.0](https://clapeyronthermo.github.io/Clapeyron.jl/stable/eos/activity/#Clapeyron.ogUNIFAC2) and [mod. UNIFAC 2.0](https://clapeyronthermo.github.io/Clapeyron.jl/stable/eos/activity/#Clapeyron.UNIFAC2) are described in the `Clapeyron.jl` documentation.

## GRAPPA

![](assets/GRAPPA_scheme.svg)

GRAPPA is a graph neural network model for predicting vapor pressures and boiling points of pure components [hoffmann_grappahybrid_2025](@cite).
The model predicts the parameters ``A``, ``B``, and ``C`` of the Antoine equation:

```math
\ln(p^s / \text{kPa}) = A - \frac{B}{T / \text{K} + C}
```

On model construction, the Antoine parameters are predicted and a [`SaturationModel`](https://clapeyronthermo.github.io/Clapeyron.jl/stable/eos/correlations/#Clapeyron.SaturationModel) is automatically created, which enables the calculation of the vapor pressure via [`saturation_pressure`](https://clapeyronthermo.github.io/Clapeyron.jl/stable/properties/single/#Clapeyron.saturation_pressure) for a given temperature.

```@docs
MLThermoProperties.GRAPPA
```

## ESE

![](assets/ESE_scheme.svg)

ESE is a hybrid model for predicting binary diffusion coefficients at infinite dilution [wagner_hybrid_2026](@cite).
The model incorporates the Stokes-Einstein equation and ensures a physically consistent temperature dependence of the predicted diffusion coefficients.
It only requires the SMILES of the solute and solvent as input, together with a viscosity model for the solvent.

```@docs
MLThermoProperties.ESE
```