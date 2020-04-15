using Test, MatrixComputations, LinearAlgebra

n_values = 2:2:10

@testset "Householder projections" begin
    for n in n_values
        v = rand(n)
        H = diagm(ones(n)) - 2 / dot(v, v) * v*v'
        P = HouseholderMatrix(v)
        @test H == HouseholderMatrix(v) == HouseholderMatrix(v, 2/dot(v,v)) == HouseholderMatrix(v, 2/dot(v,v), (1, n))
        @test size(P) == (n, n)
        @test H[1, 1] == P[1, 1]

        I = diagm(ones(n))
        @test isapprox(norm(P*I[:, 1] - P[:, 1]), 0, atol=1e-14)
        @test isapprox(norm(P*I - P), 0, atol=1e-14)

        x = rand(n)
        for k in 1:n
            for j in k:n
            P = create_projection(x, k, j)
            p = P*x
            @test isapprox(sum(p[k+1:j]), 0, atol=1e-14)
            @test create_projection(x, k, j) == create_projection(x, k:j)
            end
        end
        @test isapprox(norm(create_projection(x, 1, n) - create_projection(x)[1]), 0, atol=1e-14)
    end
end