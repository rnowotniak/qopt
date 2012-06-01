% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    

% Differential Evolution for global optimization
% Programmed by Xin-She Yang @Cambridge University 2008

% The basic version of scheme DE/Rand/1 is implemented
% Usage: de(para) or de;

function [best,fmin,N_iter]=de(para)

global initial_flag;
initial_flag = 0;

% Default parameters
if nargin<1,
   para=[10 0.7 0.9];
   help de.m
end

n=para(1);      % Population >=4, typically 10 to 25
F=para(2);      % DE parameter - scaling (0.5 to 0.9)
Cr=para(3);     % DE parameter - crossover probability

% Iteration parameters
tol=10^(-5);    % Stop tolerance
N_iter=0;       % Total number of function evaluations

% Dimension of the search variables
d=10;

% Simple bounds 
Lb=-100 * ones(1,d);
Ub=100 * ones(1,d);

% Initialize the population/solutions
for i=1:n,
  Sol(i,:)=Lb+(Ub-Lb).*rand(size(Lb));
  Fitness(i)=Fun(Sol(i,:));
end

% Find the current best among the initial guess;
[fmin,I]=min(Fitness);
best=Sol(I,:);

% Start the iterations by differential evolution
while (N_iter < 100000) % XXX
    % Obtain donor vectors by permutation
    k1=randperm(n);     k2=randperm(n);
    k1sol=Sol(k1,:);    k2sol=Sol(k2,:);
        % Random crossover index/matrix
        K=rand(n,d)<Cr;
        % DE/RAND/1 scheme
        V=Sol+F*(k1sol-k2sol);
        V=Sol.*(1-K)+V.*K;

        % Evaluate new solutions
        for i=1:n,
           Fnew=Fun(V(i,:));
           % If the solution improves
           if Fnew<=Fitness(i),
                Sol(i,:)=V(i,:);
                Fitness(i)=Fnew;
           end
          % Update the current best
          if Fnew<=fmin,
                best=V(i,:);
                fmin=Fnew;
          end
        end
        N_iter=N_iter+n;
end

% Output/display
disp(['Number of evaluations: ',num2str(N_iter)]);
disp(['Best=',num2str(best),' fmin=',num2str(fmin)]);


% Objective function -- Rosenbrock's 3D function
function z=Fun(u)
z = benchmark_func(u, 5);
%z=(1-u(1))^2+100*(u(2)-u(1)^2)^2+100*(u(3)-u(2)^2)^2;



