% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    

% -------------------------------------------------------- %
% Bat-inspired algorithm for continuous optimization (demo)%
% Programmed by Xin-She Yang @Cambridge University 2010    %
% -------------------------------------------------------- %
% Usage: bat_algorithm([20 0.25 0.5]);                     %


function [best,fmin,N_iter]=bat_algorithm(fnum)
% Display help
 help bat_algorithm.m

rng shuffle;

global gfnum;
gfnum = fnum;
 
global initial_flag;
initial_flag = 0;
 
% Default parameters
para=[10 0.25 0.5];
n=para(1);      % Population size, typically 10 to 25
A=para(2);      % Loudness  (constant or decreasing)
r=para(3);      % Pulse rate (constant or decreasing)
% This frequency range determines the scalings
Qmin=0;         % Frequency minimum
Qmax=2;         % Frequency maximum
% Iteration parameters
tol=10^(-5);    % Stop tolerance
N_iter=0;       % Total number of function evaluations
% Dimension of the search variables
d=10;
% Initial arrays
Q=zeros(n,1);   % Frequency
v=zeros(n,d);   % Velocities
% Initialize the population/solutions
for i=1:n,
  Sol(i,:)=randn(1,d);
  Fitness(i)=Fun(Sol(i,:));
end
% Find the current best
[fmin,I]=min(Fitness);
best=Sol(I,:);

% ======================================================  %
% Note: As this is a demo, here we did not implement the  %
% reduction of loudness and increase of emission rates.   %
% Interested readers can do some parametric studies       %
% and also implementation various changes of A and r etc  %
% ======================================================  %

% Start the iterations -- Bat Algorithm
while (N_iter < 100000) % (fmin>tol)
        % Loop over all bats/solutions
        for i=1:n,
          Q(i)=Qmin+(Qmin-Qmax)*rand;
          v(i,:)=v(i,:)+(Sol(i,:)-best)*Q(i);
          S(i,:)=Sol(i,:)+v(i,:);
          % Pulse rate
          if rand>r
              S(i,:)=best+0.01*randn(1,d);
          end

     % Evaluate new solutions
           Fnew=Fun(S(i,:));
     % If the solution improves or not too loudness
           if (Fnew<=Fitness(i)) & (rand<A) ,
                Sol(i,:)=S(i,:);
                Fitness(i)=Fnew;
           end

          % Update the current best
          if Fnew<=fmin,
                best=S(i,:);
                fmin=Fnew;
          end
        end
        N_iter=N_iter+n;
        disp(sprintf('%d %g', N_iter, fmin));
end
% Output/display
disp(['Number of evaluations: ',num2str(N_iter)]);
disp(['Best =',num2str(best),' fmin=',num2str(fmin)]);
% Objective function -- Rosenbrock's 3D function
function z=Fun(u)
global gfnum;
%disp(u)
z = benchmark_func(u, gfnum);
%z=(1-u(1))^2+100*(u(2)-u(1)^2)^2+(1-u(3))^2;
%%%%% ============ end ====================================

