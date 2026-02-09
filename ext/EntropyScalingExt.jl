module EntropyScalingExt

using MLPROP
using EntropyScaling

const ES = EntropyScaling

function MLPROP.SEB(components, es_model::ES.AbstractEntropyScalingModel; p=1e5, kwargs...)
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