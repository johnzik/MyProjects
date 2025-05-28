using SparseArrays, LinearAlgebra;
import PrettyTables; # Remove later

"""
    Lmesh(maxSQlen::Float64)

# Arguments
 - maxSQlen::Float64 : The maximum side size of the square with which you want to create the L mesh. (This is the max length, if it doesn't 
                        divide the domain equally the function will take nearest value below maxSQlen which creates an even mesh)

# Returns
- go : Global Ordering (go) is a matrix with 3 other martices (the subdomains) inside it, those 3 matrices contain the mesh of each subdomain 
"""
function Lmesh(maxSQlen::Float64)

    domain = [-1 0 1;] # Domain

    # Func input
    domStep = abs(domain[2] - domain[1]);
    maxSQlen = min(maxSQlen, domStep); # Maximum square length (not greater than 1 so that we dont overwrite the domain)
    maxSQ = div(domStep, maxSQlen, RoundUp); # Maximum small squares inside the big square based on maxSQlen
    step1 =  domStep/maxSQ; # Final closest step so that all small squares have equal length <= maxSQlen
    
    # Domain properties
    n = length(domain);
    domMid = (n+1)/2;
    
    # ---- Square 1 & 2 Mesh --------------------------------
    
    x1 = range(domain[1], domain[Integer(domMid)], step = step1) |> collect; # Nodes vector in row 1
    rowNodes = length(x1);
    rowElem = length(x1) - 1; # Elements in 1 row
    
    # Global ordering of Square 1 & 2
    
    # Faster way than for loop, but same complexity, O(n^2):
    go1 = (rowNodes-1:-1:0).*(2*rowNodes-1).+(1:rowNodes)'; # Only use matrix operations to create the mesh
    lastNode = go1[end,end]; # Temp to hold the last node so that we can construct the next matrix
    
    go2 = (rowNodes-1:-1:0).*(2*rowNodes-1).+(lastNode:lastNode+rowNodes-1)';
    lastNode = go2[1,end];
    
    # ---- Square 3 Mesh --------------------------------
    
    go3 = zeros(Int, rowElem+1, rowElem+1);
    go3[end,:] = go1[1,:];
    go3[1:end-1, :] = (rowNodes-2:-1:0).*(rowNodes).+(lastNode+1:lastNode+rowNodes)';
    
    go = [go1, go2, go3]; # Use go[1] to get go1, go[2] to get go2 etc.

    return go
end

"""
    QuadFEM_Matrices(GO::Vector{Matrix{Int64}}, f::Function, k::Float64)

# Arguments
- GO::Vector{Matrix{Int64}} : The Global Ordering matrix which contains the 3 seperate subdomains matrices meshed (use Lmesh to create it)
- f::Function : The right hand side function of the PDE
- k::Float64 : k is the wave number (The bigger it is we expect more peaks to appear)

# Returns
- K : K is the sparse stiffness matrix of our inhomogeneous Helmholtz equation 
- F_global : F_global is the final right hand side of the equation. Now use a solver to solve the system K*u=F_global for u(x,y,t)
"""
function QuadFEM_Matrices(GO::Vector{Matrix{Int64}}, f::Function, k::Float64)

    rowNodes = length(GO[1][1,:]);
    colNodes = length(GO[1][:,1]);
    nop = 3*rowNodes^2 - 2*rowNodes; # Number of points
    rowElem = rowNodes - 1;
    colElem = colNodes - 1;
    noe = rowElem*colElem*3; # Number of elements
    step1 = 1/(rowNodes - 1); # Get step from the GO matrix

    # ---- Create local 2 global mapping ----------------------------------------------------------------------

    l2g = zeros(noe,4);

    idx = 1; # Index
    for i = 0:rowElem-1
        for j = 1:colElem
            l2g[idx, :] = [GO[1][end-i, j], GO[1][end-i, j+1], GO[1][end-i-1, j+1], GO[1][end-i-1, j]];
            l2g[idx + Int(noe/3), :] = [GO[2][end-i, j], GO[2][end-i, j+1], GO[2][end-i-1, j+1], GO[2][end-i-1, j]];
            l2g[idx + Int(2*noe/3), :] = [GO[3][end-i, j], GO[3][end-i, j+1], GO[3][end-i-1, j+1], GO[3][end-i-1, j]];
            idx = idx + 1;
        end
    end

    # ---- Point Coords ---------------------------------------------------------------------------------------

    coords = zeros(nop,2);

    # Coordinates of points inside GO[1] & GO[2]
    for i = 1:rowNodes
        xvals = collect(-1:step1:1); # x values from -1 to 1 with step1
        yvals = (-1 + step1*(i - 1))*ones(rowNodes*2-1,1); # rowNodes*2-1 cause there is 1 common point, increase from -1 with step1 once every iteration

        start_idx = (i - 1)*(rowNodes*2-1) + 1;
        end_idx = (rowNodes*2-1)*i;
        coords[start_idx:end_idx, :] = hcat(xvals, yvals);
    end

    # Coordinates of points inside GO[3]
    for i = 2:rowNodes
        xvals = collect(-1:step1:0); # x values from -1 to 1 with step1
        yvals = (0 + step1*(i - 1))*ones(rowNodes,1); # rowNodes*2-1 cause there is 1 common point, increase from -1 with step1 once every iteration

        start_idx = GO[3][end-i+1,1];
        end_idx = GO[3][end-i+1,end];
        coords[start_idx:end_idx, :] = hcat(xvals, yvals);
    end

    # ---- Set Variables & Iterate and Populate ---------------------------------------------------------------

    # Interpolation functions
    N = [
        (ksi, h) -> (1/4)*(1-ksi)*(1-h);
        (ksi, h) -> (1/4)*(1+ksi)*(1-h);
        (ksi, h) -> (1/4)*(1+ksi)*(1+h);
        (ksi, h) -> (1/4)*(1-ksi)*(1+h); 
    ]

    # Assemble Stiffness Matrix & Right Hand Side 
    Me = zeros(4, 4); # Element Me
    Te = zeros(4, 4); # Element Te

    ia = zeros(Int, 16*noe); # Row Sparse idx
    ja = zeros(Int, 16*noe); # Column Sparse idx
    va = zeros(Float64, 16*noe); # Value Sparse idx

    fe = zeros(4); # Element f
    F_global = zeros(nop, 1); # Global F

    c = 1; # idx for Sparse construction
    for e = 1:noe
        # Point Coordinates of each element
        xe = coords[Int.(l2g[e,:]), 1]; 
        ye = coords[Int.(l2g[e,:]), 2];

        # Jacobian matrix values quadrilateral elements 
        J11 = (ksi, h) -> (1/4)*(-(1-h)*xe[1] + (1-h)*xe[2] + (1+h)*xe[3] - (1+h)*xe[4]);
        J12 = (ksi, h) -> (1/4)*(-(1-h)*ye[1] + (1-h)*ye[2] + (1+h)*ye[3] - (1+h)*ye[4]);
        J21 = (ksi, h) -> (1/4)*(-(1-ksi)*xe[1] - (1+ksi)*xe[2] + (1+ksi)*xe[3] + (1-ksi)*xe[4]);
        J22 = (ksi, h) -> (1/4)*(-(1-ksi)*ye[1] - (1+ksi)*ye[2] + (1+ksi)*ye[3] + (1-ksi)*ye[4]);
        detJ = (ksi, h) -> J11(ksi,h)*J22(ksi,h) - J12(ksi,h)*J21(ksi,h); # Determinant of J matrix

        # Interpolation functions derivatives
        dNx = [
            (ksi, h) -> (1/(4*detJ(ksi, h))) * (-J22(ksi, h)*(1-h) + J12(ksi, h)*(1-ksi));
            (ksi, h) -> (1/(4*detJ(ksi, h))) * ( J22(ksi, h)*(1-h) + J12(ksi, h)*(1+ksi));
            (ksi, h) -> (1/(4*detJ(ksi, h))) * ( J22(ksi, h)*(1+h) - J12(ksi, h)*(1+ksi));
            (ksi, h) -> (1/(4*detJ(ksi, h))) * (-J22(ksi, h)*(1+h) - J12(ksi, h)*(1-ksi));
        ]

        dNy = [
            (ksi, h) -> (1/(4*detJ(ksi, h))) * ( J21(ksi, h)*(1-h) - J11(ksi, h)*(1-ksi));
            (ksi, h) -> (1/(4*detJ(ksi, h))) * (-J21(ksi, h)*(1-h) - J11(ksi, h)*(1+ksi));
            (ksi, h) -> (1/(4*detJ(ksi, h))) * (-J21(ksi, h)*(1+h) + J11(ksi, h)*(1+ksi));
            (ksi, h) -> (1/(4*detJ(ksi, h))) * ( J21(ksi, h)*(1+h) + J11(ksi, h)*(1-ksi));
        ]

        for i = 1:4 # 1:4 because we have 4 nodes per element
            for j = 1:4 
                # Assemble Stiffness Matrix
                Mfunc = (ksi, h) -> (dNx[i](ksi, h)*dNx[j](ksi, h) + dNy[i](ksi, h)*dNy[j](ksi, h)) * detJ(ksi, h);
                Me[i,j] = - DGQ_leg(Mfunc, 3);

                Tfunc = (ksi, h) -> k^2 * N[i](ksi, h) * N[j](ksi, h) * detJ(ksi, h);
                Te[i,j] = DGQ_leg(Tfunc, 3);
            end

            # Assemble right hand side
            xkh = (ksi, h) -> xe[1]*N[1](ksi, h) + xe[2]*N[2](ksi, h) + xe[3]*N[3](ksi, h) + xe[4]*N[4](ksi, h);
            ykh = (ksi, h) -> ye[1]*N[1](ksi, h) + ye[2]*N[2](ksi, h) + ye[3]*N[3](ksi, h) + ye[4]*N[4](ksi, h);
            ffunc = (ksi, h) -> N[i](ksi, h)*f(xkh(ksi,h), ykh(ksi,h)) * detJ(ksi, h);
            fe[i] = DGQ_leg(ffunc ,3);

            # Global f
            F_global[Int.(l2g[e, i])] = F_global[Int.(l2g[e, i])] + fe[i];
        end

        # Global Sparse Striffness Matrix Construction
        for i = 1:4
            for j = 1:4
                ia[c] = l2g[e,i];
                ja[c] = l2g[e,j];
                va[c] = Me[i,j] + Te[i,j];
                c += 1;
            end
        end

    end

    K = sparse(ia, ja, va, nop, nop);

    # ---- Find & Enforce Dirichlet Boundary Conditions -------------------------------------------------------
   
    # Find the nodes
    boundary_nodes = findall(
        (coords[:,1] .== -1) .| 
        (coords[:,1] .== 1) .| 
        (coords[:,2] .== -1) .| 
        (coords[:,2] .== 1) .| 
        ((coords[:,1] .== 0) .& (coords[:,2] .>= 0)) .| 
        ((coords[:,2] .== 0) .& (coords[:,1] .>= 0))
    );

    # Enforce the Boundary Conditions
    for i in boundary_nodes
        cols, _ = findnz(K[i, :])  # Ignore 2nd output of findnz (which is the actual non-zero values themselves) and keep only the columns 
                                   # indices of the non-zero elements in row i
        for j in cols
            K[i, j] = 0.0
        end
        K[i, i] = 1.0
        F_global[i] = 0.0
    end

    return K, F_global, coords;
end

"""
    QuadFEM_Solver(K::SparseMatrixCSC{Float64, Int64}, F_global::Matrix{Float64})

Returns the solution of K * u = F using the normal backslash
"""
function QuadFEM_Solver(K::SparseMatrixCSC{Float64, Int64}, F_global::Matrix{Float64})

    u = K\F_global; # Solve the system

    return u;
end

"""
    DGQ_leg(f::Function, n::Integer)

# Arguments
- f::Function : The function f(x,y) you want to double integrate from -1 to 1 (This calculates the Legendre GQ)
- n::Integer : The degree of the Gausian Quadrature. With n-point GQ you can solve for polynomials of degree 2n-1 or less

# Returns
- res : The aproximate result of the Double Integration of f(x,y) from -1 to 1
"""
function DGQ_leg(f::Function, n::Integer) # Up to 5 point Double Gaussian Quadrature (Legendre)

    # Gaussian Quadrature points and weights table
    GQ_Table = [
        ([0.0], [2.0]),
        ([+1/sqrt(3), - 1/sqrt(3)], [1.0, 1.0]),
        ([0, +sqrt(3/5), -sqrt(3/5)], [8/9, 5/9, 5/9]),
        ([+sqrt(3/7-(2/7)*sqrt(6/5)), -sqrt(3/7-(2/7)*sqrt(6/5)), +sqrt(3/7+(2/7)*sqrt(6/5)), -sqrt(3/7+(2/7)*sqrt(6/5))]
            ,[(18+sqrt(30))/36, (18+sqrt(30))/36, (18-sqrt(30))/36, (18-sqrt(30))/36]),
        ([0, +(1/3)*sqrt(5-2*sqrt(10/7)), -(1/3)*sqrt(5-2*sqrt(10/7)), +(1/3)*sqrt(5+2*sqrt(10/7)), -(1/3)*sqrt(5+2*sqrt(10/7))]
            ,[128/225, (322+13*sqrt(70))/900, (322+13*sqrt(70))/900, (322-13*sqrt(70))/900, (322-13*sqrt(70))/900])
    ]

    # Points and Weights initialization
    p, w = GQ_Table[n]; 

    # Double Gaussian Quadrature calculation
    res = 0.0;
    for i = 1:n
        for j = 1:n
            res += + w[i] * w[j] * f(p[i], p[j]);
        end
    end

    return res
end

function ndgrid(x::AbstractVector, y::AbstractVector)
    X = repeat(x', length(y), 1)  # Transpose x, repeat across rows
    Y = repeat(y, 1, length(x))   # Repeat y across columns
    return X, Y
end

"""
    Conj_Grad(A, B, tol)

Return the solution of 'A * x = B' using the Conjugate Gradient Method (CG)
"""
function Conj_Grad(A, B, tol, Nmax)
    M = size(A)
    q = zeros(M[1])

    # Residual
    r = B - A * q
    p = r

    # Norm of rbs
    norm0 = norm(r)

    it = 0;

    if norm(r) > tol * norm0
        for i in 1:Nmax
            denom = dot(p, A*p);
            @assert abs(denom) > eps() "CG: Broke down in alpha division (division by 0 = Inf)"
            a = dot(r, r) / denom; # dot(a, b) computes (a' * b)
            q += a * p;

            temp_r = r;
            r0 = r;
            r = r - a * A * p;

            if norm(r) < tol * norm0
                println("CG converged with ", i, " iterations");
                return q
            end

            denom2 = dot(temp_r, temp_r);
            @assert abs(denom2) > eps() "CG: Broke down in beta division (division by 0 = Inf)"
            beta = dot(r, r) / denom2;
            p = r + beta * p;
            it += 1;
        end
    end

    println("CG did not converge after $Nmax iterations. Final residual: ", norm(r) / norm0)
    return q
end

"""
    SubdomainOverlap_Matrices(GO::Vector{Matrix{Int64}}, K::SparseMatrixCSC{Float64, Int64} , F_global::Matrix{Float64}, H::Int)

This function takes the Global Ordering, in subdomains, and overlaps each into the other by H elements. Then 
it constructs and returns the stiffness and right hand side matrices of each subdomain.

# Arguments
- **GO::Vector{Matrix{Int64}} :** The Global Ordering matrix which contains the 3 seperate subdomains matrices meshed (use Lmesh to create it)
- **K::SparseMatrixCSC{Float64, Int64} :** K is the sparse stiffness matrix of our inhomogeneous Helmholtz equation
- **F_global::Matrix{Float64} :** F_global is the final right hand side of the equation. Now use a solver to solve the system K*u=F_global for u(x,y,t)
- **H::Int :** This variable sets the overlap number, in elements, between the subdomains

# Returns
- **K_sub :** This contains the stiffness matrices for each subdomain (outputs: K_sub[i], i=1,2,3)
- **F_global_sub :** This contains the right hand side matrices for each subdomain (outputs: F_global_sub[i], i=1,2,3)
- **idx :** The 3 subdomains node vectors with their overlaps
"""
function SubdomainOverlap_Matrices(GO::Vector{Matrix{Int64}}, 
                                   K::SparseMatrixCSC{Float64, Int64}, 
                                   F_global::Matrix{Float64}, 
                                   H::Int)

    # Catch Error: Overlap must not exceed the element length of a subdomains vertical or horizontal side
    if H > length(GO[1][:,1]) - 1
        return error("Overlap H must NOT exceed the element length of a subdomains vertical or horizontal side");
    end

    # Inner node vectors
    inGO1 = vec(GO[1]);
    inGO2 = vec(GO[2]);
    inGO3 = vec(GO[3]);

    # Overlapping node vectors
    G12 = vec(GO[2][:, 1:H+1]); # Overlap of subdomain 1 into 2 (H+1 because H is the overlap in elements)
    G13 = vec(GO[3][end-H:end, :]); # Overlap of subdomain 1 into 3
    G1 = vcat(G12, G13); # Merge the overlapping nodes of subdomain 1 into 2 & 3
    G21 = vec(GO[1][:, end-H:end]); # Overlap of subdomain 2 into 1
    G31 = vec(GO[1][1:H+1, :]); # Overlap of subdomain 3 into 1

    # Overlapping and inner node union vectors
    idx1 = unique([inGO1;G1]);
    idx2 = unique([inGO2;G21]);
    idx3 = unique([inGO3;G31]);

    # Subdomain stiffness matrices
    K_sub = [
        K[idx1, idx1],
        K[idx2, idx2],
        K[idx3, idx3]
    ]

    # Subdomain right hand side
    F_global_sub = [
        F_global[idx1],
        F_global[idx2],
        F_global[idx3]
    ]

    return K_sub, F_global_sub, [idx1, idx2, idx3]
end

"""
    pcdSAP(K::SparseMatrixCSC{Float64, Int64}, 
           r_input::Matrix{Float64}, 
           K_sub::Vector{SparseMatrixCSC{Float64, Int64}}, 
           idx::Vector{Vector{Int}})

This function is a preconditioner, utilizing Schwartz Additive Procedure, that is used
together with the preconditioned Conjugate Gradient Method (pcdCG function)

# Arguments
- **K::SparseMatrixCSC{Float64, Int64} :** K is the sparse stiffness matrix of our inhomogeneous Helmholtz equation
- **r_input::Matrix{Float64} :** This is the residual input vector of the pcdCG function 
- **K_sub::Vector{SparseMatrixCSC{Float64, Int64}} :** This contains the stiffness matrices for each subdomain (outputs: K_sub[i], i=1,2,3)
- **idx::Vector{Vector{Int}} :** The 3 subdomains node vectors with their overlaps

# Returns
 - **z :** This the solution of the system M^-1 * z = r , where M^-1 is the preconditioner matrix
"""

function pcdSAP(K::SparseMatrixCSC{Float64, Int64}, 
                r_input::Matrix{Float64}, 
                K_sub::Vector{SparseMatrixCSC{Float64, Int64}}, 
                idx::Vector{Vector{Int}} 
                )

    nop = size(K,1); # Number of nodes in the whole domain

    z = zeros(nop, 1); # Output vector initialization

    # Solve system for the union of subdomain 1 and its overlaps
    z1 = K_sub[1] \ r_input[idx[1], 1];

    # Solve system for the union of subdomain 2 and its overlaps
    z2 = K_sub[2] \ r_input[idx[2], 1];

    # Solve system for the union of subdomain 3 and its overlaps
    z3 = K_sub[3] \ r_input[idx[3], 1];

    z[idx[1], 1] .+= z1;
    z[idx[2], 1] .+= z2;
    z[idx[3], 1] .+= z3;

    return z;
end

"""
    pcdCG(K::SparseMatrixCSC{Float64, Int64}, 
               F::Matrix{Float64}, 
               tol::Float64, 
               K_sub::Vector{SparseMatrixCSC{Float64, Int64}},
               idx::Vector{Vector{Int}}, Nmax::Int)

This function is an implementation of a preconditioned CG Method (Polak-Ribière variant of CG), using SAP (pcdSAP function) as a preconditioner 

# Arguments
- **K::SparseMatrixCSC{Float64, Int64} :** K is the sparse stiffness matrix of our inhomogeneous Helmholtz equation
- **F::Matrix{Float64} :** F is the final right hand side of the equation
- **tol::Float64 :** This is the tolerance of the solver (ex. 10^-6) 
- **K_sub::Vector{SparseMatrixCSC{Float64, Int64}} :** This contains the stiffness matrices for each subdomain (outputs: K_sub[i], i=1,2,3)
- **idx::Vector{Vector{Int}} :** The 3 subdomains node vectors with their overlaps
- **Nmax::Int :** The maximum number of iterations

# Returns
 - **q :** The solution of the system K * q = F
"""
function pcdCG(K::SparseMatrixCSC{Float64, Int64}, 
               F::Matrix{Float64}, 
               tol::Float64, 
               K_sub::Vector{SparseMatrixCSC{Float64, Int64}},
               idx::Vector{Vector{Int}}, Nmax::Int
               )

    M = size(K, 1);
    q = zeros(M[1]);

    # Initial Residual
    r = F - K * q;
    norm0 = norm(r); # Norm of initial r (r0)

    # Initial preconditioned residual (z0 = M^-1 * r0)
    z = pcdSAP(K, r, K_sub, idx);
    p = z;

    for i in 1:Nmax
        w = K * p;

        a_denom = dot(p, w);
        @assert abs(a_denom) > eps() "pcdCG: Broke down in alpha division (division by 0 = Inf)"
        a = dot(r, z) / a_denom;
        q = q + a*p;
        r_new = r - a*w;

        current_norm = norm(r_new) / norm0;
        println("Iteration $i: ||r|| / ||r0|| = $current_norm");

        if current_norm < tol
            println("pcdCG converged with ", i, " iterations");
            return q
        end

        # New preconditioned residual ( z_k+1 = M^-1 * r_k+1)
        z_new = pcdSAP(K, r_new, K_sub, idx);

        denom2 = dot(r, z);
        @assert abs(denom2) > eps() "pcdCG: Broke down in beta division (division by 0 = Inf)"
        beta = dot(r_new, z_new) / denom2;

        p = z_new + beta*p;

        # Update r and z 
        r = r_new;
        z = z_new;
    end

    println("pcdCG did not converge after $Nmax iterations. Final residual: ", norm(r) / norm0)
    return q
end

struct MGLevel
    h::Float64; # Mesh step size
    GO::Vector{Matrix{Int64}}; # GO matrix for the current level
    coords::Matrix{Float64}; # Global coordinates for current level
    K::SparseMatrixCSC{Float64, Int64}; # Stiffness matrix for the current level
    D::SparseVector{Float64, Int64} # The diagonal of K for the current level
end

"""
    buildMGlevels(InitialStep::Float64, levels::Int, f::Function, k::Float64)

# Arguments
- **InitialStep::Float64** : The step of the finest grid
- **levels::Int** : The number of level used in the Multigrid method
- **f::Function :** The right hand side function of the PDE
- **k::Float64 :** k is the wave number (The bigger it is we expect more peaks to appear)

# Returns
- **A::Vector{MGLevel} :** The MGLevel object with all its variables and matrices computed 
"""
function buildMGlevels(InitialStep::Float64, levels::Int, f::Function, k::Float64, H::Int)
    A = Vector{MGLevel}(undef, levels); # Pre-allocate for a vector of 'levels' uninitialized elements
    currentStep = InitialStep;
    
    for i = 1:levels
        # Build GO for the current level
        GO_level = Lmesh(currentStep);

        # Calc K and coords for current level
        K_level, _, coo_level = QuadFEM_Matrices(GO_level, f, k);

        # Compute the diagonal of K for the Damped Jacobi
        D_lvl = diag(K_level);

        # Push the vector in 
        A[i] = MGLevel(currentStep, GO_level, coo_level, K_level, D_lvl);
        
        currentStep *= 2.0;
    end

    return A
end

function compRnP(A::Vector{MGLevel})

    levels = length(A); # Multigrid levels

    # Initialize the Restriction & Prolongation vectors for all levels
    R = Vector{SparseMatrixCSC}(undef, levels - 1);
    P = Vector{SparseMatrixCSC}(undef, levels - 1);

    for i = 1:levels-1
        nFine_lvl = size(A[i].GO[1], 1) - 1; # Fine intervals
        nCoarse_lvl = size(A[i+1].GO[1], 1) - 1; # Coarse intervals

        R1D = spzeros(nCoarse_lvl-1, nFine_lvl-1); # Pre-allocate space for the vector

        for j = 1:nCoarse_lvl-1
            R1D[j, (2*(j-1)+1):(2*j+1)] .= (1/4)*[1, 2, 1];
        end
        # Tensor product the construct the Restriction operator
        R[i] = kron(R1D, R1D);
        P[i] = 4*transpose(R[i]);
    end

    return R, P
end

function MG_Vcycle(A::Vector{MGLevel}, 
                   r, 
                   u, 
                   P, 
                   R, 
                   nPre::Int, nPost::Int, lvl::Int, maxlvl::Int, omega::Float64, gamma::Int
                   )
    if lvl == maxlvl
        u[lvl] = A[lvl].K \ r[lvl];
        return
    end

    # Pre-smoothing
    for _ = 1:nPre
        u[lvl] = u[lvl] + omega*(r[lvl] - A[lvl].K * u[lvl]) ./ A[lvl].D;
    end

    # Restrict
    r[lvl+1] = R[lvl] * (r[lvl] - A[lvl].K * u[lvl]);
    u[lvl+1] = zeros(size(A[lvl+1].K, 1)); # Initial guess for the coarse grid correction

    # Recursive call of the 2 grid cycle
    for _ = 1:gamma
        MG_Vcycle(A, r, u, P, R, nPre, nPost, lvl+1, maxlvl, omega, gamma);
    end 
    
    # Prolong the coarse grid correction
    u[lvl] = u[lvl] + P[lvl]*u[lvl+1];

    # Post-smoothing
    for _ = 1:nPost
        u[lvl] = u[lvl] + omega*(r[lvl] - A[lvl].K * u[lvl]) ./ A[lvl].D;
    end

    return
end

function MGpcdSAP(K::SparseMatrixCSC{Float64, Int64}, 
                  r_input::Matrix{Float64},
                  idx::Vector{Vector{Int}},
                  MG::Vector{MGLevel},
                  R_sub::Vector{SparseMatrixCSC},
                  P_sub::Vector{SparseMatrixCSC},
                  nPre::Int, nPost::Int, omega::Float64, gamma::Int
                 )

    nop = size(K,1); # Number of nodes in the whole domain
    z = zeros(nop, 1); # Output vector initialization

    # Iterate through the subdomains
    for i = 1:length(idx) # We have 3 subdomains
        maxlvl = length(MG); # Number of levels used in MG

        # Initialize r_lvl (RHS/Residual) for V-cycle of the current subdomain
        r_lvl = Vector{Vector{Float64}}(undef, maxlvl);
        r_lvl[1] = r_input[idx[i], 1];
        for lvl = 2:maxlvl
            r_lvl[lvl] = zeros(size(MG[lvl].K, 1)); 
        end

        # Initialize u_lvl (solution) for V-cycle of the current subdomain
        u_lvl = Vector{Vector{Float64}}(undef, maxlvl);
        u_lvl[1] = zeros(length(r_lvl[1])) # Set initial guess to 0
        for lvl = 2:maxlvl
            u_lvl[lvl] = zeros(size(MG[lvl].K, 1));
        end

        # Modifies u_lvl in place
        MG_Vcycle(MG, r_lvl, u_lvl, P_sub[i], R_sub[i], nPre, nPost, 1, maxlvl, omega, gamma);

        # Add the correction to the solution vector z
        z[idx[i], 1] .+= u_lvl[1];
    end

    return z;
end

function MG_PCG(K::SparseMatrixCSC{Float64, Int64}, 
               F::Matrix{Float64}, 
               tol::Float64,
               idx::Vector{Vector{Int}}, Nmax::Int,
               subMG::Vector{MGLevel},
               R_sub::Vector{SparseMatrixCSC},
               P_sub::Vector{SparseMatrixCSC},
               nPre::Int, nPost::Int, omega::Float64, gamma::Int
               )

    M = size(K, 1);
    q = zeros(M[1]);

    # Initial Residual
    r = F - K * q;
    norm0 = norm(r); # Norm of initial r (r0)

    # Initial preconditioned residual (z0 = M^-1 * r0)
    z = MGpcdSAP(K, r, idx, subMG, R_sub, P_sub, nPre, nPost, omega, gamma);
    p = z;

    for i in 1:Nmax
        w = K * p;

        a_denom = dot(p, w);
        @assert abs(a_denom) > eps() "MG_PCG: Broke down in alpha division (division by 0 = Inf)"
        a = dot(r, z) / a_denom;
        q = q + a*p;
        r_new = r - a*w;

        current_norm = norm(r_new) / norm0;
        println("Iteration $i: ||r|| / ||r0|| = $current_norm");

        if current_norm < tol
            println("MG_PCG converged with ", i, " iterations");
            return q
        end

        # New preconditioned residual ( z_k+1 = M^-1 * r_k+1)
        z_new = MGpcdSAP(K, r_new, idx, subMG, R_sub, P_sub, nPre, nPost, omega, gamma);

        denom2 = dot(r, z);
        @assert abs(denom2) > eps() "MG_PCG: Broke down in beta division (division by 0 = Inf)"
        beta = dot(r_new, z_new) / denom2;

        p = z_new + beta*p;

        # Update r and z 
        r = r_new;
        z = z_new;
    end

    println("pcdCG did not converge after $Nmax iterations. Final residual: ", norm(r) / norm0)
    return q
end
