% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== % 

% GEWA (Generalized evolutionary walker algorithm)--a demo %
% by Xin-She Yang @Cambridge University 2008, modified 2010%
% Three major components in GEWA:                          %
% 1) random walk 2)randomization 3) selection/elitism      %
% -------------------------------------------------------- %
% Two algorithm-dependent parameters:                      %
%       n=population size or number of random walkers      %
%       pa=randomization probability                       %
% ------------------------------------------------------   %
% Usage: >gewa(5000) or simply >gewa;                      %

% This demo is for unconstrained optimization only, though %        
% it is easy to extend it for constrained optimization.    %     

function [bestsol,fval]=gewa(N_iter)

global initial_flag;
initial_flag = 0;

% Default number of iterations
if nargin<1, N_iter=5000; end

% Display help info
help gewa.m

% dimension or number variables
d=10;
% Lower and upper bounds
Lb=-100*ones(1,d);  Ub=100*ones(1,d);

% population size -- the number of walkers
n=10;

% Probability -- balance of local & global search
alpha=0.5;

% Random initial solutions
ns=init_sol(n,Lb,Ub);
% Evaluate all new solutions and find the best
fval=init_fval(ns);
[fbest,sbest,kbest]=findbest(ns,fval);

% Iterations begin
for j=1:N_iter,

    % Discard the worst and replace it later
    k=get_fmax(fval);

    if rand<alpha,
    % Local search by random walk
      ns(k,:)=rand_walk(sbest,Lb,Ub);

    else
    % Global search by randomization
      ns(k,:)=randomization(Lb,Ub);
    end

    % Evaluation and selection of the best
    fval(k)=fobj(ns(k,:));
    if fval(k)<fbest,
        fbest=fval(k);
        sbest=ns(k,:);
        kbest=k;
    end
end  % end of iterations

% Post-processing and show all the solutions
ns
%% Show the best and number of evaluations
Best_solution=sbest
Best_fmin=fbest
Number_Eval=N_iter+n

% ----- All subfunctions are placed here ----------------
% Initial solutions
function ns=init_sol(n,Lb,Ub);
for i=1:n,
ns(i,:)=Lb+rand(size(Lb)).*(Ub-Lb);
end

% Perform random walks around the best
function s=rand_walk(sbest,Lb,Ub)
 step=0.01;
 s=sbest+randn(size(sbest)).*(Ub-Lb).*step;

% Discard the worst solution and replace it later
function k=get_fmax(fval)
[fmax,k]=max(fval);

% Randomization in the whole search space
function s=randomization(Lb,Ub)
d=length(Lb);
s=Lb+(Ub-Lb).*rand(1,d);

% Evaluations of all initial solutions
function [fval]=init_fval(ns)
n=size(ns,1);
for k=1:n,
    fval(k)=fobj(ns(k,:));
end

% Find the best solution so far
function [fbest,sbest,kbest]=findbest(ns,fval)
n=size(ns,1);
fbest=fval(1);
sbest=ns(1,:); kbest=1;
 for k=2:n,
       if fval(k)<=fbest,
          fbest=fval(k);
          sbest=ns(k,:);
          kbest=k;
       end
 end

% Objective function
function z=fobj(u)
z = benchmark_func(u, 1);
% Rosenbrock's 3D function
%z=(1-u(1))^2+100*(u(2)-u(1)^2)^2+(1-u(3))^2;
% -------- end of the GEWA implementation -------

