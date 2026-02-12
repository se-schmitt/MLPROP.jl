# scalers
abstract type AbstractScaler{T} end

struct Scaler{T} <: AbstractScaler{T}
    μ::T
    σ::T
end

scale(scaler::Scaler, v::T) where {T} = (v .- scaler.μ) ./ scaler.σ
unscale(scaler::Scaler, v::T) where {T} = v .* scaler.σ .+ scaler.μ

load_scaler(path::String; T=Float64) = load_scaler(path, Scaler; T)
function load_scaler(path::String, ::Type{Scaler}; T=Float64)
    @load joinpath(DB_PATH, path) μ σ
    return Scaler(T.(μ), T.(σ))
end
