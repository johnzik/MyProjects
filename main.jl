import PrettyTables;
include("functions.jl")
using SparseArrays, GLMakie, BenchmarkTools;

step1 = 0.0625;

GO = Lmesh(step1);

show(stdout, "text/plain", GO[3])
println()

go12 = [GO[1] GO[2][:,2:end]];
show(stdout, "text/plain", go12)
println()

# Define the function as an anonymous function
s = 0.05;
k = 11.0; # If k = 0 then we have the Poisson Equation (Because we have division by zero errors use eps(Float64) instead of 0). The bigger k is we have more peaks
f = (x, y) -> (1/(2*pi*s^2)) * exp(-((x+0.5).^2 + (y-0.5).^2) / (2*s^2)) + (1/(2*pi*s^2)) * exp(-((x-0.5).^2 + (y+0.5).^2) / (2*s^2));

@time "Build matrices time" begin
    K1, F_global1, all_coords = QuadFEM_Matrices(GO, f, k);
end

H = 10; # Overlap
K_sub, F_sub, idx = SubdomainOverlap_Matrices(GO, K1, F_global1, H);

# Multigrid setup phase
MGlevels = 3;
A = buildMGlevels(step1, MGlevels, f, k);
R, P = compRnP(A);
nPre = 2;
nPost = 1;
omega = 4/5;
gamma = 1;

@time "Solving time" begin
    # u = QuadFEM_Solver(K1, F_global1);
    tol = 10^-6;
    # u = Conj_Grad(K1, F_global1, tol, 1000);
    # u = pcdCG(K1, F_global1, tol, K_sub, idx, 500);
    u = MG_PCG(K1, F_global1, tol, idx, 500, A, R, P, nPre, nPost, omega, gamma);
end

# ----------------------------------------------------------------------------
#       Plotting the result
# ----------------------------------------------------------------------------

GLMakie.activate!();
rowNodes = length(GO[1][1,:]);
newStep = 1/(rowNodes - 1); # Get step from the GO matrix

# ---- Plot the Domain ----------------------------

# --- Mesh for plotting ---
X1m, Y1m = ndgrid(-1:newStep:0, -1:newStep:0);
X2m, Y2m = ndgrid(0:newStep:1, -1:newStep:0);
X3m, Y3m = ndgrid(-1:newStep:0, 0:newStep:1);

fig_mesh = GLMakie.Figure();
ax_mesh = GLMakie.Axis(fig_mesh[1,1], xlabel = "x", ylabel = "y", title = "Physical Mesh Nodes of L-Shape Domain");

GLMakie.scatter!(ax_mesh, all_coords[:,1], all_coords[:,2],
                 color = :blue, markersize = 5,
                 strokecolor = :black, strokewidth = 0.5, # Stroke is for better visibility
                 );

screen_mesh = display(GLMakie.Screen(), fig_mesh);

# ---- Set animation parameters -------------------
c = 3*10^8; # Speed of the wave
omega = k*c; # Angular Velocity
T = 2π / omega; # Period
frames = 60;
ts = range(0, T, length=frames);  # Time frames

u1 = u[GO[1]]; U1 = reshape(u1, size(GO[1]));
u2 = u[GO[2]]; U2 = reshape(u2, size(GO[2]));
u3 = u[GO[3]]; U3 = reshape(u3, size(GO[3]));

# --- Create Observable for each subdomain ---
U1_t = Observable(real.(U1));
U2_t = Observable(real.(U2));
U3_t = Observable(real.(U3));

# --- Create Figure ---
fig2 = GLMakie.Figure();
screen2 = display(GLMakie.Screen(), fig2);
ax = GLMakie.Axis3(fig2[1,1], title = "Wave Propagation", xlabel="x", ylabel="y", zlabel="u(x,y,t)");

# --- Surface Plots ------------------------------------
GLMakie.surface!(ax, X1m, Y1m, U1_t, colormap = :viridis);
GLMakie.surface!(ax, X2m, Y2m, U2_t, colormap = :viridis);
GLMakie.surface!(ax, X3m, Y3m, U3_t, colormap = :viridis);

# --- Animation ----------------------------------------
for t in ts
    phase = exp(1im * omega * t);
    U1_t[] = real.(U1 * phase);
    U2_t[] = real.(U2 * phase);
    U3_t[] = real.(U3 * phase);
    sleep(0.05)
end

nothing