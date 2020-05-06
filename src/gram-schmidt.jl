export gram_schmidt

"""
Given the `n`-by-`m`-dimensional matrix `V`, apply modified Gram-Schmidt orthonormalization such that `Q R = V`

Can be called also via gram_schmidt(v1, v2, ..., vm).

inspired by "Fundamentals of Matrix Computations", D. Watkins, John Wiley & Sons, 1991
"""
function gram_schmidt(V::AbstractMatrix)
    n, m = size(V)
    Q = copy(V)
    R = spzeros(m, m)
    for k  in 1:m
        R[k, k] = norm(Q[:, k])
        isapprox(R[k, k], 0, atol=1e-10) && throw(error("vectors appear to be linearly dependent."))
        Q[:, k] *= 1 / R[k, k]
        if k < m
            for j in k+1:m
                R[k, j] = dot(Q[:, j], Q[:, k])
                Q[:, j] -= R[k, j] * Q[:, k]
            end
        end
    end
    Q, UpperTriangular(R)
end

gram_schmidt(v::AbstractVector{<:AbstractVector}) = gram_schmidt(hcat(v...))
gram_schmidt(v::AbstractVector...) = gram_schmidt(hcat(v...))