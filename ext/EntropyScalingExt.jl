module EntropyScalingExt

using MLPROP
using EntropyScaling

function MLPROP.SEB(components, es_model::AbstractESModel; kwargs..., p=1e5)
    wrapper = ESModelWrapper(es_model, p)
    return SEB(components, wrapper; kwargs...)
end

struct ESModelWrapper
    model
    p
end
function (wrapper::ESModelWrapper)(T)
    return viscosity(wrapper.model, wrapper.p, T)
end

end