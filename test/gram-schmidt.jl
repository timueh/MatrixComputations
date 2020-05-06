using Test, MatrixComputations, LinearAlgebra

v1 = [3; 0; 4.]
v2 = [-2; 3; 4 ]

V = hcat(v1, v2)

@testset "Gram-Schmidt orthonormalization" begin
    @test gram_schmidt(V) == gram_schmidt([v1, v2]) == gram_schmidt(v1, v2)

    Q, R = gram_schmidt(v1, v2)

    @test isapprox(norm(Q'*Q - diagm(ones(size(V)[2]))), 0, atol=1e-12)
    @test isapprox(norm(Q*R - V), 0, atol=1e-12)
end