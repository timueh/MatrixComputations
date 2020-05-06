export  forward_elimination,
        backward_substitution,
        gauss_elimination,
        solve

"""
Given an `n`-by-`n` nonsingular lower triangular matrix `L` and corresponding `x`, this algorithm finds `y such that `Ly = x`.

See "Matrix Computations", G.H. Golub and C.F. Van Loan, Johns Hopkins University Press, 1983
"""
function forward_elimination(L, x)
    @assert( !isapprox(prod(diag(L)), 0), "matrix appears to be singular (det = $(prod(diag(L))))" )
    n, y = length(x), similar(x)
    @assert( size(L)[1] == size(L)[2] == n, "inconsistent dimensions" )

    for i in 1:n
        y[i] = x[i]
        for j in 1:i-1
            y[i] -= L[i, j]*y[j] 
        end
        y[i] /= L[i, i]
    end
    y
end

"""
Given an `n`-by-`n` nonsingular upper triangular matrix `U` and corresponding y, this algorithm finds `x` such that `Ux = y`.

See "Matrix Computations", G.H. Golub and C.F. Van Loan, Johns Hopkins University Press, 1983
"""
function backward_substitution(U, y)
    @assert( !isapprox(prod(diag(U)), 0), "matrix appears to be singular (det = $(prod(diag(L))))" )
    n, x = length(y), similar(y)
    @assert( size(U)[1] == size(U)[2] == n, "inconsistent dimensions" )
    for i = n:-1:1
        x[i] = y[i]
        for j = i+1:n
            x[i] -= U[i, j]*x[j]
        end
        x[i] /= U[i, i]
    end
    x
end

# function gauss_elimination_basic(M::AbstractMatrix)
#     A = copy(M)
#     m, n = size(A)
#     w = spzeros(n)
#     for k in 1:min(m-1, n)
#         isapprox(A[k, k], 0) && return A
#         w[k+1:n] = A[k, k+1:n]
#         for i in k+1:m
#             η = A[i, k] / A[k, k]
#             A[i, k] = η
#             for j in k+1:n
#                 A[i, j] -= η * w[j] 
#             end
#         end
#     end
#     UnitLowerTriangular(A), UpperTriangular(A)
# end


"""
Given a nonsingular matrix `M` execute Gaussian elimination with partial pivoting.
The function returns the LU decomposition in the first two arguments, and the permutation matrix `P` as the third argument

inspired by "Fundamentals of Matrix Computations", D. Watkins, John Wiley & Sons, 1991
"""
function gauss_elimination(M::AbstractMatrix)
    m, n = size(M)
    @assert(m == n, "matrix is non-square ($m-by-$n)")
    intch = zeros(Int64, n-1)
    A = copy(M)
    for k in 1:n-1
        # find maximal pivot
        amax, imax = findmax(abs.(A[k:n, k]))
        imax += k - 1

        if amax == 0
            throw(error("Matrix appears to be singular."))
        else
            if imax != k
                A[k, 1:n], A[imax, 1:n] = A[imax, 1:n], A[k, 1:n]
            end
            intch[k] = imax
            for i in k+1:n
                A[i, k] = A[i, k] / A[k, k]
                for j in k+1:n
                    A[i, j] -= A[i, k]*A[k, j]
                end
            end
        end
    end
    UnitLowerTriangular(A), UpperTriangular(A), create_permutation_matrix(n, intch)
end

function create_permutation_matrix(n::Int, r::Vector{Int})
    P = spdiagm(0 => ones(Int64, n))
    for (k, rk) in enumerate(r)
        P = create_permutation_matrix(n, k, rk)*P
    end
    P
end

function create_permutation_matrix(n::Int, row1::Int, row2::Int)
    A = spdiagm(0 => ones(Int64, n))
    A[row1, :], A[row2, :] = A[row2, :], A[row1, :]
    A
end

function solve(A::AbstractMatrix, b::AbstractVector)
    @assert( size(A)[1] == size(A)[2] == length(b), "inconsistent dimensions")
    L, U, P = gauss_elimination(A)
    y = forward_elimination(L, P*b)
    x = backward_substitution(U, y)
end