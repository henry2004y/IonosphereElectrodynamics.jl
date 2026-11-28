# ==============================================================================
# Generic Numeric & Vector Helpers (Zero-Overhead for IEEE Floats, AD-Safe)
# ==============================================================================

struct _Default end
const _DEFAULT = _Default()

@inline _val_or(::_Default, default) = default
@inline _val_or(x, default) = x

@inline _tuple_eltype(t::Tuple{Real}) = typeof(t[1])
@inline _tuple_eltype(t::Tuple{Real, Real}) = promote_type(typeof(t[1]), typeof(t[2]))
@inline _tuple_eltype(t::Tuple{Real, Real, Real}) =
    promote_type(typeof(t[1]), typeof(t[2]), typeof(t[3]))
@inline _tuple_eltype(t::Tuple{Vararg{Real}}) = mapreduce(typeof, promote_type, t)

@inline _promote_nondefault(acc::Type, ::_Default) = acc
@inline _promote_nondefault(acc::Type, ::Nothing) = acc
@inline _promote_nondefault(acc::Type, ::Type{T}) where {T <: Real} = promote_type(acc, T)
@inline _promote_nondefault(acc::Type, ::Type{Union{}}) = acc
@inline _promote_nondefault(acc::Type, x::Real) = promote_type(acc, typeof(x))
@inline _promote_nondefault(acc::Type, v::AbstractArray{<:Real}) =
    promote_type(acc, eltype(v))
@inline _promote_nondefault(acc::Type, v::Tuple{Vararg{Real}}) =
    promote_type(acc, _tuple_eltype(v))

@inline _promote_args(acc::Type) = acc
@inline _promote_args(acc::Type, x, rest...) =
    _promote_args(_promote_nondefault(acc, x), rest...)

@inline function _infer_ft(::Nothing, args...)
    T = _promote_args(Union{}, args...)
    return T === Union{} ? Float64 : float(T)
end
@inline _infer_ft(::Type{FT}, args...) where {FT <: Real} = FT

@inline _scalar_float_type(::Type{T}) where {T <: AbstractFloat} = T
@inline _scalar_float_type(::Type{T}) where {T <: Real} = typeof(eps(float(T)))

@inline _floatmin(::Type{T}) where {T <: AbstractFloat} = floatmin(T)
@inline _floatmin(::Type{T}) where {T <: Real} = floatmin(_scalar_float_type(T))

@inline _eps(::Type{T}) where {T <: AbstractFloat} = eps(T)
@inline _eps(::Type{T}) where {T <: Real} = eps(float(T))

@inline _inf(::Type{T}) where {T <: Real} = convert(_scalar_float_type(T), Inf)

@inline function _tiny(::Type{T}) where {T <: Real}
    sT = _scalar_float_type(T)
    e = eps(sT)
    val = e < eps(Float64) ? max(floatmin(sT), e * e) : max(floatmin(sT), sT(1.0e-30))
    return convert(T, val)
end

@inline function _eps_rho(::Type{T}) where {T <: Real}
    sT = _scalar_float_type(T)
    e = eps(sT)
    val = e < eps(Float64) ? max(floatmin(sT), sqrt(e)) : max(floatmin(sT), sT(1.0e-12))
    return convert(T, val)
end

@inline function _eps_omega(::Type{T}) where {T <: Real}
    sT = _scalar_float_type(T)
    e = eps(sT)
    val = e < eps(Float64) ? max(floatmin(sT), e) : max(floatmin(sT), sT(1.0e-14))
    return convert(T, val)
end

@inline function _pi(::Type{T}) where {T <: Real}
    sT = _scalar_float_type(T)
    return convert(T, convert(sT, pi))
end

@inline _floor_int(x::Base.IEEEFloat) = unsafe_trunc(Int, floor(x))
@inline _floor_int(x::Real) = floor(Int, x)

@inline _as_svec3(x::SVector{3, T}) where {T <: Real} = x
@inline function _as_svec3(x::AbstractVector{<:Real})
    length(x) == 3 || throw(DimensionMismatch("expected length-3 vector, got $(length(x))"))
    i0 = firstindex(x)
    @inbounds return SVector{3}(x[i0], x[i0 + 1], x[i0 + 2])
end
@inline _as_svec3(x::NTuple{3, Real}) = SVector{3}(x[1], x[2], x[3])

const _Vec3Input = Union{AbstractVector{<:Real}, NTuple{3, Real}}
