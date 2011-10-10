% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    

% -------------------------------------------------------- %
% Cuckoo algorithm by Xin-She Yang and Suasg Deb           %
% Programmed by Xin-She Yang at Cambridge University 2009  %
% -------------------------------------------------------- %
% Usage: cuckoo_search(5000) or cuckoo_search;             %

function [bestsol,fval]=cuckoo_search(Ngen)
% Here Ngen is the max number of function evaluations
if nargin<1, Ngen=1500; end

% Display help info
help cuckoo_search

% d-dimensions (any dimension)
d=2;
% Number of nests (or different solutions)
n=25;

% Discovery rate of alien eggs/solutions
pa=0.25;

% Random initial solutions
nest=randn(n,d);
fbest=ones(n,1)*10^(100);   % minimization problems
Kbest=1;

for j=1:Ngen,
    % Find the current best
    Kbest=get_best_nest(fbest);
    % Choose a random nest (avoid the current best)
    k=choose_a_nest(n,Kbest);
    bestnest=nest(Kbest,:) ;
    % Generate a new solution (but keep the current best)
    s=get_a_cuckoo(nest(k,:),bestnest);

    % Evaluate this solution
    fnew=fobj(s);
    if fnew<=fbest(k),
        fbest(k)=fnew;
        nest(k,:)=s;
    end
    % discovery and randomization
    if rand<pa,
       k=get_max_nest(fbest);
       s=emptyit(nest(k,:));
       nest(k,:)=s;
       fbest(k)=fobj(s);
    end
end

%% Post-optimization processing

%% Display all the nests
nest

%% Find the best and display
[fmin,I]=min(fbest); 
best_solution=nest(I,:)
best_fmin=fmin

%% --------- All subfunctions are listed below -----------
%% Choose a nest randomly
function k=choose_a_nest(n,Kbest)
k=floor(rand*n)+1;
% Avoid the best
if k==Kbest,
 k=mod(k+1,n)+1;
end

%% Get a cuckoo and generate new solutions by ramdom walk
function s=get_a_cuckoo(s,star)
% This is a random walk, which is less efficient
% than Levy flights. In addition, the step size
% should be a vector for problems with different scales.
% Here is the simplified implementation for demo only!
stepsize=0.05;
s=star+stepsize*randn(size(s));

%% Find the worse nest
function k=get_max_nest(fbest)
[fmax,k]=max(fbest);

%% Find the current best nest
function k=get_best_nest(fbest)
[fmin,k]=min(fbest);

%% Replace some (of the worst nests)
%% by constructing new solutions/nests
function s=emptyit(s)
% Again the step size should be varied
% Here is a simplified approach
s=s+0.05*randn(size(s));

% d-dimensional objective function
function z=fobj(u)
% Rosenbrock's function (in 2D)
% It has an optimal solution at (1.000,1.000)
z=(1-u(1))^2+100*(u(2)-u(1)^2)^2;

%%%% ============== end ===================================

