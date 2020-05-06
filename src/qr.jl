export qr

"""
Given an `m`-by-`n`-dimensional Matrix `A` (`m >= n`), return `Q` and `R` such that `A = QR`.
`Q` is orthogonal, and `R` is upper triangular.

see "Matrix Computations", G.H. Golub and C.F. Van Loan, Johns Hopkins University Press, 1983
"""
function qr(A::AbstractMatrix)
    m, n = size(A)
    @assert(m >= n, "implemented only for matrices Nrows >= Ncolumns")

    P = Vector{HouseholderMatrix}(undef, n)
    R = copy(A)
    for k in 1:n
        v = R[k:m, k]
        P_, σ = create_projection(v)
        P[k] = HouseholderMatrix([zeros(k-1); P_.v])
        R = P[k]*R
        display(R)
    end
    get_Q(P, m), R
end

function get_Q(P::Vector{HouseholderMatrix}, m::Int)
    Q = diagm(ones(m))
    for (i, P_) in enumerate(P)
        Q *= P_
    end
    Q
end