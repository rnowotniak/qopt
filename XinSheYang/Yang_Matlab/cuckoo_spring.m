% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %   


% Cuckoo Search for nonlinear constrained optimization     %
% Programmed by Xin-She Yang @ Cambridge University 2009   %
% Usage: cuckoo_spring(25000) or cuckoo_spring;            %

function [bestsol,fval]=cuckoo_spring(N_iter)
format long;
% Display help info
help cuckoo_spring.m

% number of iterations
if nargin<1, N_iter=15000; end
% Number of nests
n=25;
disp('Searching ... may take a minute or so ...');
% d variables and simple bounds
% Lower and upper bounds
Lb=[0.05 0.25 2.0];
Ub=[2.0  1.3  15.0];
% Number of variables
d=length(Lb);

% Discovery rate
pa=0.25;
% Random initial solutions
nest=init_cuckoo(n,d,Lb,Ub);
fbest=ones(n,1)*10^(10);   % minimization problems
Kbest=1;

% Start of the cuckoo search
for j=1:N_iter,
    % Find the best nest
    [fmin,Kbest]=get_best_nest(fbest);
    % Choose a nest randomly
    k=choose_a_nest(n,Kbest);
    bestnest=nest(Kbest,:) ;
    % Get a cuckoo with a new solution
    s=get_a_cuckoo(nest(k,:),bestnest,Lb,Ub);

    % Update if the solution improves
    fnew=fobj(s);
    if fnew<=fbest(k),
        fbest(k)=fnew;
        nest(k,:)=s;
    end

    % Discovery and randomization
    if rand<pa,
    k=get_max_nest(fbest);
    s=emptyit(nest(k,:),Lb,Ub);
    nest(k,:)=s;
    fbest(k)=fobj(s);
    end
end

%% Find the best
[fmin,I]=min(fbest)
bestsol=nest(I,:);

% Show all the nests
nest
% Display the best solution
bestsol, fmin

% Initial locations of all n cuckoos
function [guess]=init_cuckoo(n,d,Lb,Ub)
for i=1:n,
    guess(i,1:d)=Lb+rand(1,d).*(Ub-Lb);
end

%% Choose a nest randomly
function k=choose_a_nest(n,Kbest)
k=floor(rand*n)+1;
% Avoid the best
if k==Kbest,
 k=mod(k+1,n)+1;
end

%% Get a cuckoo with a new solution via a random walk
%% Note: Levy flights were not implemented in this demo
function s=get_a_cuckoo(s,star,Lb,Ub)
s=star+0.01*(Ub-Lb).*randn(size(s));
s=bounds(s,Lb,Ub);

%% Find the worse nest
function k=get_max_nest(fbest)
[fmax,k]=max(fbest);

%% Find the best nest
function [fmin,k]=get_best_nest(fbest)
[fmin,k]=min(fbest);

%% Replace an abandoned nest by constructing a new nest
function s=emptyit(s,Lb,Ub)
s=s+0.01*(Ub-Lb).*randn(size(s));
s=bounds(s,Lb,Ub);

% Check if bounds are met
function ns=bounds(ns,Lb,Ub)
% Apply the lower bound
  ns_tmp=ns;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);
  % Apply the upper bounds
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
% Update this new move
  ns=ns_tmp;

% d-dimensional objective function
function z=fobj(u)
% The well-known spring design problem
z=(2+u(3))*u(1)^2*u(2);
z=z+getnonlinear(u);

function Z=getnonlinear(u)
Z=0;
% Penalty constant
lam=10^15;

% Inequality constraints
g(1)=1-u(2)^3*u(3)/(71785*u(1)^4);
gtmp=(4*u(2)^2-u(1)*u(2))/(12566*(u(2)*u(1)^3-u(1)^4));
g(2)=gtmp+1/(5108*u(1)^2)-1;
g(3)=1-140.45*u(1)/(u(2)^2*u(3));
g(4)=(u(1)+u(2))/1.5-1;

% No equality constraint in this problem, so empty;
geq=[];

% Apply inequality constraints
for k=1:length(g),
    Z=Z+ lam*g(k)^2*getH(g(k));
end
% Apply equality constraints
for k=1:length(geq),
   Z=Z+lam*geq(k)^2*getHeq(geq(k));
end

% Test if inequalities hold
% Index function H(g) for inequalities
function H=getH(g)
if g<=0,
    H=0;
else
    H=1;
end
% Index function for equalities
function H=getHeq(geq)
if geq==0,
   H=0;
else
   H=1;
end
% ----------------- end ------------------------------

