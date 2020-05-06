export cg, cg_precond

function cg(A::AbstractMatrix, b::AbstractVector; ε = 1e-12, kmax::Int=500)
    # @assert( size(A)[1] == size(A)[2] == length(b), "inconsistent dimensions")
    # @assert( norm(A - A') ≈ 0 && isposdef(A), "matrix is not positive definite")
    n = length(b)
    x, r = zeros(n), copy(b)
    ρ, η, k = dot(r, r), 0., 1
    while sqrt(ρ) > ε*norm(b) && k < kmax
        p = r
        if k > 1
            p += ρ / η * p 
        end

        w = A*p
        α = ρ / dot(p, w)
        x += α*p
        r -= α*w
        η, ρ = ρ, dot(r, r)
        k += 1
    end
    x
end

function cg_precond(A::AbstractMatrix, b::AbstractVector)
    @assert( size(A)[1] == size(A)[2] == length(b), "inconsistent dimensions")
    @assert( norm(A - A') ≈ 0 && isposdef(A), "matrix is not symmetric & positive definite")
    m = diag(A)

    n = length(b)
    r = copy(b)
    x = z = z_ = r_ = p = zeros(Float64, n)

    for k in 1:n
        isapprox(norm(r, Inf), 0, atol=1e-12) && return x
        z = r ./ m
        dzr = dot(z, r)
        β = k == 1 ? 0. : dzr / dot(z_, r_)
        p = z + β*p
        α = dzr / dot(p, A*p)
        x += α*p
        r_, z_ = r, z
        r -= α*A*p
    end
    x
end