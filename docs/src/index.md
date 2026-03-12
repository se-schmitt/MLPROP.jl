````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: MLPROP
  text: State-of-the-art hybrid thermodynamic models
  image:
    src: /logo.png
    alt: MLPROP.jl
  tagline: Hybrid machine learning models to predict various thermodynamic properties - from phase equilibria to transport properties 
  actions:
    - theme: alt
      text: Getting started
      link: /#Installation
    - theme: alt
      text: Examples and Tutorials
      link: /tutorials
    - theme: alt
      text: View on GitHub
      link: https://github.com/se-schmitt/MLPROP.jl

features:
  - icon: 🤝
    title: Thermodynamics + Machine Learning
    details: Predictions of thermodynamic properties for any substance based on hybrid thermodynamic models
    link: /models

  - icon: <img width="150" height="64" src="Clapeyron_logo_without_text.svg" alt="Clapeyron"/>
    title: Build on Clapeyron.jl
    details: Use the rich thermodynamics solvers from Clapeyron.jl
    link: https://github.com/ClapeyronThermo/Clapeyron.jl

  - icon: 📊
    title: Website
    details: See our website for interactive calculation of thermodynamic properties based on the MLPROP models
    link: https://ml-prop.mv.rptu.de/
---
````

## Installation

Install `MLPROP.jl` in Julia by

```julia-repl
julia> using Pkg; Pkg.add("MLPROP")
```

## Properties and Thermodynamic Solvers

`MLPROP.jl` uses **`Clapeyron.jl`** as backend for the **calculation of thermodynamic bulk properties and phase equilibria**.
The `Clapeyron.jl` documentation can be found [here](https://clapeyronthermo.github.io/Clapeyron.jl/stable/).

For **transport properties**, the methods from **`EntropyScaling.jl`** (see [documentation](https://se-schmitt.github.io/EntropyScaling.jl/stable/)) are employed.

### Quick Links:

- [Bulk properties](https://clapeyronthermo.github.io/Clapeyron.jl/stable/properties/bulk/)
- [Pure VLE methods](https://clapeyronthermo.github.io/Clapeyron.jl/stable/properties/single/#Fluid-Single-component-properties)
- [Mixture VLE methods](https://clapeyronthermo.github.io/Clapeyron.jl/stable/properties/multi/#Bubble/Dew-Points)
- [LLE/VLLE methods](https://clapeyronthermo.github.io/Clapeyron.jl/stable/properties/multi/#Azeotropes,-LLE-and-VLLE-equilibria)
- [Flash solvers](https://clapeyronthermo.github.io/Clapeyron.jl/stable/properties/multi/#TP-Flash)
- [Transport properties](https://se-schmitt.github.io/EntropyScaling.jl/stable/transport_properties/)

## Models

The MLPROP models and their usage are documented [here](@ref models_page).