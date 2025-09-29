
include("simplex.jl")

using Pkg
# Pkg.add("JuMP")
# Pkg.add("GLPK")

using JuMP
using GLPK
using Random


# ================== [ Example 1 ] ==================
# c = [3.0, 1.0, 1.0, 0.0, 0.0]
# A = [2.0 1.0 1.0 1.0 0.0;
#      1.0 -1.0 -1.0 0.0 1.0]
# b = [2.0, -1.0]



# ================== [ Example 2 ] ==================
c = [-19.0, -13.0, -12.0, -17.0, -20.0, -11.0]
A = [3.0 2.0 1.0 2.0 5.0 2.0;
    1.0 1.0 1.0 1.0 2.0 5.0;
    4.0 3.0 3.0 4.0 4.0 6.0]
b = [225.0, 117.0, 420.0]



# ================== [ Example 3 ] ==================
# m = 100
# n = 200
# A = randn(m, n)              
# x0 = rand(n)                 
# b = A * x0                   
# z = rand(m)                  
# c = transpose(A) * z + rand(n)



### ================================================ ###
###                    SIMPLEX                       ###
### ================================================ ###

println("Dimensions of the problem: (m,n) = $(size(A))")
result = two_phase_simplex(c, A, b)


println("\n")
println("Simplex Solution:")
println("\tStatus: $(result.status)")
println("\tSolution: $(result.solution)")
println("\tOptimal Value: $(result.value)")
println("\tBasic Set: $(result.basic)")
println("\tNonbasic Set: $(result.nonbasic)")
println("\n")





### ================================================ ###
###                    GLPK solver                   ###
### ================================================ ###


println(repeat("=", 60))
println(repeat(" ", 20), "GLPK solver")
println(repeat("=", 60))

m, n = size(A)

model = Model(GLPK.Optimizer)

@variable(model, x[1:n] >= 0)
@objective(model, Min, sum(c[i]*x[i] for i in 1:n))

for i in 1:m
    @constraint(model, sum(A[i,j]*x[j] for j in 1:n) == b[i])
end

optimize!(model)

# Store results in your own variables
solution = value.(x)
opt_value = objective_value(model)
status = termination_status(model)

println("Status: ", status)
println("Solution: ", solution)
println("Optimal value: ", opt_value)
