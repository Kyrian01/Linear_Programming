using Pkg

# Use following Packages
using LinearAlgebra
using Printf



struct SimplexResult
    status::String              
    solution::Union{Nothing, Vector{Float64}}   
    value::Union{Nothing, Float64}      
    basic::Union{Nothing, Vector{Int}}        
    nonbasic::Union{Nothing, Vector{Int}}     
end




function revised_simplex(
    c::Vector{Float64},           
    A::Matrix{Float64},           
    xB::Vector{Float64},           
    basic::Vector{Int},            
    nonbasic::Vector{Int},         
    max_iter::Int = 100_000 ,
    tol::Float64 = 1e-12        
)

    # Get dimensions
    m, n = size(A)
    println("Dimensions: (m, n) = ($m, $n)")

    println("Cost Function: $c")
    println("Constraint Matrix: $A")
    println("\n")


    # Make local copies so modifying them here doesn't change the input arrays.
    basic = copy(basic)
    nonbasic = copy(nonbasic)
    xB = copy(xB)

    # Simplex loop
    for iter in 1:max_iter


        println("============================ [ Iteration $iter ] ============================")

        # Update (non)basis matrices B, N
        B = A[:, basic]
        N = A[:, nonbasic]

        println("Current Information:")
        println("\tBasic Set: $basic")
        println("\tBasic Matrix: $B")
        println("\tNonbasic Set: $nonbasic")
        println("\tNonbasic matrix: $N")
        println("\tCurrent Basic Feasible Point: $xB")

        # Partition cost into basic & nonbasic
        cN = c[nonbasic]
        cB = c[basic]

        # Compute Langrange Multiplier λ
        lam = B' \ cB

        # Pricing: compute reduced costs
        sN = cN - (N' * lam)

        println("Langrange Information:")
        println("\tLangrange Multipliers: $lam")
        println("\tNonbasic reduced costs: $sN")

        # Termination: nonnegative reduced costs
        if all(sN .>= -tol)
            println("Optimal solution found in $iter iterations")
            x = zeros(n)
            x[basic] = xB
            value = dot(c, x)

            return SimplexResult(
                "Optimal",
                x,
                value,
                basic,
                nonbasic
            )
        end

        # Choose entering index (most negative reduced cost)
        entering_pos = argmin(sN)
        entering_idx = nonbasic[entering_pos]

        println("Entering Information:")
        println("\tEntering position: $entering_pos")
        println("\tEntering index: $entering_idx")

        # Determine step direction d
        d = B \ A[:, entering_idx]
        if (all(d .<= tol))
            println("Optimization Problem is Unbounded!")
            return SimplexResult(
                "Unbounded", 
                nothing, 
                nothing, 
                nothing, 
                nothing
            )   
        end

        # Determine steplength & leaving index 
        ratios = [d[i] > tol ? xB[i] / d[i] : Inf for i in 1:m]
        leaving_pos = argmin(ratios)
        alpha = ratios[leaving_pos]

        println("Moving Information:")
        println("\tMoving Direction: $d")
        println("\tratios: $ratios")
        println("\tLeaving Position: $leaving_pos")
        println("\tLeaving Index: $(basic[leaving_pos])")
        println("\tstepsize: $alpha")

        # Update basic feasible iterate xB
        xB = xB - (d * alpha)
        xB[leaving_pos] = alpha

        # Update basic and nonbasic indices
        nonbasic[entering_pos] = basic[leaving_pos]
        basic[leaving_pos] = entering_idx

        println("Update Iterate Information:")
        println("\tNew Basic Feasible Point: $xB")
        println("\tNew Basic Set: $basic") 
        println("\tNew Nonbasic Set: $nonbasic") 
        println("\n")
    end

    # Fallback solution when iterations exceed max iterations
    return SimplexResult(
                "Timeout", 
                nothing, 
                nothing, 
                nothing, 
                nothing
            ) 
end







function two_phase_simplex(
    c::Vector{Float64},           
    A::Matrix{Float64},           
    b::Vector{Float64},
    ) 

    # Get dimensions of Problem
    m, n = size(A)

    # ====== [ PHASE I SIMPLEX ] ====== 
    println(repeat("=", 60))
    println(repeat(" ", 25), "PHASE I")
    println(repeat("=", 60))
    cost = vcat(zeros(n), ones(m))
    E = Diagonal(sign.(b))   
    constraint_matrix = hcat(A, Matrix(E))
    xB_phaseI = abs.(b)

    basic_phaseI= collect(n+1 : n+m)
    nonbasic_phaseI = collect(1:n) 
    
    phaseI = revised_simplex(
        cost, 
        constraint_matrix, 
        xB_phaseI,
        basic_phaseI,
        nonbasic_phaseI)
    
    # Check infeasibility
    if (phaseI.value > 0)
        println("Optimization Problem in Infeasible")

        return SimplexResult(
            "Infeasible",
            nothing,
            nothing,
            nothing,
            nothing
        )
    end
    println("Solution: $(phaseI.solution)")
    println("Optimal Value: $(phaseI.value)")
    println("Intial Basic Set: $(phaseI.basic)")
    println("Initial Nonbasic Set: $(setdiff(1:n, phaseI.basic))")
    println("\n")
    

    # ====== [ PHASE II SIMPLEX ] ======
    println(repeat("=", 60))
    println(repeat(" ", 25), "PHASE II")
    println(repeat("=", 60))
    basic_phaseII = filter(i -> i <= n, phaseI.basic)
    nonbasic_phaseII = setdiff(1:n, basic_phaseII)
    xB_phaseII = phaseI.solution[basic_phaseII]

    phaseII = revised_simplex(
        c,
        A,
        xB_phaseII,
        basic_phaseII,
        nonbasic_phaseII
    )

    return phaseII
end







