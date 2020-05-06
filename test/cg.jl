using Test, MatrixComputations, LinearAlgebra

N = 2:10:150

@testset "Conjugate gradient" begin
    for n in N
        A, b = rand(n, n), rand(n)
        A = A' + A + diagm(20*(1:n))
        x = cg(A, b)
        @test isapprox(norm(A*x - b), 0, atol = 1e-10)
    end
end

@testset "Conjugate gradient with preconditioning" begin
    for n in N
        A, b = rand(n, n), rand(n)
        A = A' + A + diagm(1000*(1:n))
        x = cg_precond(A, b)
        @test isapprox(norm(A*x - b), 0, atol = 1e-10)
    end
end