import Base.getindex, Base.size, Base.*

export create_projection, *, HouseholderMatrix

struct HouseholderMatrix{T, V<:AbstractVector{T}} <: AbstractMatrix{T}
    v::V
    β::Float64
    nnz::Tuple
end

HouseholderMatrix(v) = HouseholderMatrix(v, 2 / dot(v, v), (findfirst(!iszero, v), findlast(!iszero, v)) )
HouseholderMatrix(v, β) = HouseholderMatrix(v, β, (findfirst(!iszero, v), findlast(!iszero, v) ) )

size(H::HouseholderMatrix) = length(H.v), length(H.v)

getindex(H::HouseholderMatrix, i::Int, j::Int) = 1 * (i==j) - H.β * H.v[i]*H.v[j]

function create_projection(x::AbstractVector, k::Int, j::Int)
    # see "Matrix Computations", G.H. Golub and C.F. Van Loan, Johns Hopkins University Press, 1983
    n = length(x)
    @assert(1 <= k <= j <= n, "inconsistent indices")
    m = norm(x[k:j], Inf)
    isapprox(m, 0) && return HouseholderMatrix(zeros(n), 0.)
    
    v, α = spzeros(n), 0.
    for i in k:j
        v[i] = x[i] / m
        α += v[i]^2
    end
    
    α = sqrt(α)
    β = 1 / (α * (α + abs(v[k])))
    v[k] += sign(v[k]) * α

    HouseholderMatrix(v, β, (k, j))
end

create_projection(x::AbstractVector, range::UnitRange) = create_projection(x, range.start, range.stop)

function create_projection(x::AbstractVector)
    # see "Fundamentals of Matrix Computations", D. Watkins, John Wiley & Sons, 1991
    m = norm(x, Inf)
    y = copy(x)

    isapprox(m, 0) && return HouseholderMatrix(x, 0.), 0

    y = 1/m * x
    σ = sign(y[1]) * norm(y, 2)
    y[1] += σ
    γ = 1 / (σ * y[1])
    σ = σ*m
    σ, γ, y
    HouseholderMatrix(y, γ), σ
end

function *(H::HouseholderMatrix, x::AbstractVector)
    # see "Matrix Computations", G.H. Golub and C.F. Van Loan, Johns Hopkins University Press, 1983
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
    # see "Matrix Computations", G.H. Golub and C.F. Van Loan, Johns Hopkins University Press, 1983
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