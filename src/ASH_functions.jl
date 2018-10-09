
# I think we can make a general ash if we specify h_diff
function h_diff_dirichlet(x,t)
    n,r = size(x); m = length(t);
    
    function h(α)
        return -sum(lgamma.(α),2) .+ lgamma.(sum(α,2))
    end
    
    out = h(repeat(t+1, inner = (n,r))) .- h(repmat(x,m,1) + repeat(t+1, inner = (n,r)));
    return reshape(out, n, length(t))
end

#
function ash_dirichlet(x::Array{Float64,2}, h_diff::Function;
                       t::Array{Float64,1} = 1.5.^(-(1:20) + 1) - 1)
    # set sizes
    n,r = size(x);
    m = length(t);
    
    # contruct a likelihood function
    loglik = h_diff(x,t)
    lik = exp.(loglik .- maximum(loglik,2));
    
    # solve mixsqp
    out = mixsqp(lik, verbose = false, lowrank = "none")["x"];
    ind = find(out .> 0);
    
    # the posterior is a mixture of Dirichlet distributions
    # Dirichlet(x_{j1} + t_k + 1, ..., x_{jr} + t_k + 1)
    
    # component posterior probability
    L = lik[:,ind];
    cpp = L .* out[ind]';
    cpp = cpp ./ sum(cpp,2);
    
    # component posterior mean
    s = length(ind);
    cpm = repmat(x,s,1) + repeat(t[ind]+1, inner = (n,r))
    cpm = cpm./sum(cpm,2);

    # posterior mean
    pm = repmat(speye(n),1,s) * (cpm .* cpp[:]);
    
    return Dict([
                (:p, out), (:pp, out[ind]), (:L, L), (:t, t[ind]), (:lik, lik),
                (:cpp, cpp), (:cpm, cpm), (:pm, pm)
                ])
end

function h_diff_beta(x,τ,ν,t)
    function B(a,b)
        return lgamma.(a) + lgamma.(b) .- lgamma.(a + b)
    end
    return -B(ν .* t + 1, (2 - ν) .* t + 1)' .+ B(x .* τ .+ (ν .* t)' + 1, (2-x) .* τ .+ ((2-ν) .* t)' + 1)
end

# ash function
function ash_beta_binomial(x::Array{Float64,1}, τ::Array{Float64,1}, h_diff::Function;
                        ν = [zeros(7);ones(5)],
                        t = [5.0.^(-3:3);5.0.^(-2:2)])
    
    # contruct a likelihood function
    loglik = h_diff(x,τ,ν,t)
    lik = exp.(loglik .- maximum(loglik,2));
    
    # solve mixsqp
    out = mixsqp(lik, verbose = false, lowrank = "none")["x"];
    ind = find(out .> 0);
    
    # the posterior is a mixture of Beta distributions
    # Beta(x .* τ .+ (ν .* t)' + 1, (2-x) .* τ .+ ((2-ν) .* t)' + 1)
    
    # component posterior probability
    L = lik[:,ind];
    cpp = L .* out[ind]';
    cpp = cpp ./ sum(cpp,2);
    
    # component posterior mean
    cpm = (x .* τ .+ (ν[ind] .* t[ind])' + 1) ./ (2 * τ .+ 2 * t[ind]' + 2)
    
    # posterior mean
    pm = sum(cpp .* cpm, 2)
    
    
    return Dict([
                (:p, out[ind]), (:L, L), (:ν, ν[ind]), (:t, t[ind]), (:lik, lik),
                (:cpp, cpp), (:cpm, cpm), (:pm, pm)
                ])
end
