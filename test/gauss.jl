using Test, MatrixComputations, LinearAlgebra

mat = ([0 4 1; 1 1 3; 2 -2 1.],
        [2 2 -4; 1 1 5; 1 3 6.],
        [2 10 8 8 6; 1 4 -2 4 -1; 0 2 3 2 1; 3 8 3 10 9; 1 4 1 2 1.])

# L, U = gauss_elimination_basic(A)
# @test isapprox(norm(L*U - Ahat), 0)

# L, U, P = gauss_eliminiation(A)
# @test isapprox(norm(L*U - Ahat), 0)
@testset "LU decomposition" begin
    for A in mat
        L, U, P = gauss_elimination(A)
        LUref = LinearAlgebra.lu(A)
        @test isapprox(norm(L*U - LUref.L*LUref.U), 0)
        @test isapprox(norm(L*U - P*A), 0)
    end
end

A = mat[end]
data = ([52; 14; 12; 51; 15.], [50; 4; 12; 48; 12.])
sols = ([1;2;1;2;1.], [2;1;2;1;2.])

@testset "Solve Ax=b" begin
    for (b, sol) in zip(data, sols)
        x = solve(A, b)
        @test isapprox(norm(A*x - b), 0., atol=1e-12)
        @test isapprox(norm(x - sol), 0., atol=1e-12)
    end
end