import Base.getindex, Base.size, Base.*

export create_projection, *, HouseholderMatrix

struct HouseholderMatrix{T, V<:AbstractVector{T}} <: AbstractMatrix{T}
    v::V
    β::Float64
    nnz::Tuple{Int64, Int64}
end

HouseholderMatrix(v) = HouseholderMatrix(v, 2 / dot(v, v), (findfirst(!iszero, v), findlast(!iszero, v)) )
HouseholderMatrix(v, β) = HouseholderMatrix(v, β, (findfirst(!iszero, v), findlast(!iszero, v) ) )

size(H::HouseholderMatrix) = length(H.v), length(H.v)

getindex(H::HouseholderMatrix, i::Int, j::Int) = 1 * (i==j) - H.β * H.v[i]*H.v[j]

function create_projection(x, k::Int, j::Int)
    n = length(x)
    @assert(1 <= k <= j <= n, "inconsistent indices")
    @assert( sum(isapprox.(x[k:j], 0)) == 0, "values in x not correct." )
    v, m, α = spzeros(n), norm(x[k:j], Inf), 0.
    
    for i in k:j
        v[i] = x[i] / m
        α += v[i]^2
    end
    
    α = sqrt(α)
    β = 1 / (α * (α + abs(v[k])))
    v[k] += sign(v[k]) * α

    HouseholderMatrix(v, β, (k, j))
end

function *(H::HouseholderMatrix, x::AbstractVector)
    v, β = H.v, H.β
    k, j = H.nnz
    y = copy(x)

    s = β*(dot(v[k:j], x[k:j]))
    for i in k:j
        y[i] -= s*v[i]
    end
    y
end

function *(H::HouseholderMatrix, X::AbstractMatrix)
    v, β = H.v, H.β
    k, j = H.nnz
    Y = copy(X)
    n, q = size(X)

    for p in 1:q
        s = β*(dot(v[k:j], X[k:j, p]))
        for i in k:j
            Y[i, p] -= s*v[i]
        end
    end
    Y
end