% ===================================================================% 
% Simulated Annealing (by X-S Yang, Cambridge University)            %
% Usage: sa_demo                                                     %
% For the constrained optimization, please see the file: sa_mincon.m %
% ------------------------------------------------------------------ %

function sa_demo
disp('Simulating ... it will take a minute or so!');
global initial_flag;
initial_flag = 0;
% Lower and upper bounds
nd=10;
Lb=-100 * ones(1,nd);
Ub=100 * ones(1,nd);

% Initializing parameters and settings
T_init =1.0;    % Initial temperature
T_min = 1e-10;  % Final stopping temperature
F_min = -1e+100;% Min value of the function
max_rej=250;    % Maximum number of rejections
max_run=150;    % Maximum number of runs
max_accept=15;  % Maximum number of accept
k = 1;          % Boltzmann constant
alpha=0.95;     % Cooling factor
Enorm=1e-2;     % Energy norm (eg, Enorm=le-8)
guess=Lb+(Ub-Lb).*rand(size(Lb));    % Initial guess
% Initializing the counters i,j etc
i= 0; j = 0; accept = 0; totaleval = 0;
% Initializing various values
T = T_init;
E_init = fobj(guess);
E_old = E_init; E_new=E_old;
best=guess; % initially guessed values

% -----------------------------------------------------------------  %
% Matlab Programs included the Appendix B in the book:               %
%  Xin-She Yang, Engineering Optimization: An Introduction           %
%                with Metaheuristic Applications                     %
%  Published by John Wiley & Sons, USA, July 2010                    %
%  ISBN: 978-0-470-58246-6,   Hardcover, 347 pages                   %
% -----------------------------------------------------------------  %
% Citation detail:                                                   %
% X.-S. Yang, Engineering Optimization: An Introduction with         %
% Metaheuristic Application, Wiley, USA, (2010).                     %
% -------------------------------------------------------------------%
% The algorithm was described in detail in Chapter 12 of the book    %
%                                                                    % 
% http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470582464.html % 
% http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0470582464.html  %
% -----------------------------------------------------------------  %
% ===== ftp://  ===== ftp://   ===== ftp:// =======================  %
% Matlab files ftp site at Wiley                                     %
% ftp://ftp.wiley.com/public/sci_tech_med/engineering_optimization   %
% ----------------------------------------------------------------   %

N_iter = 0;

% Starting the simulated annealling
%while ((T > T_min) & (j <= max_rej) & E_new>F_min) XXX
while (N_iter < 100000)
i = i+1;
% Check if max numbers of run/accept are met
if (i >= max_run) | (accept >= max_accept)
% Cooling according to a cooling schedule
T = alpha*T;
totaleval = totaleval + i;
% reset the counters
i = 1; accept = 1;
end
% Function evaluations at new locations
s=0.01*(Ub-Lb);
ns=best+s.*randn(1,nd);
E_new = fobj(ns);
% Decide to accept the new solution
DeltaE=E_new-E_old;
% Accept if improved
if (-DeltaE > Enorm)
best = ns; E_old = E_new;
accept=accept+1; j = 0;
end
% Accept with a small probability if not improved
if (DeltaE<=Enorm & exp(-DeltaE/(k*T))>rand );
best = ns; E_old = E_new;
accept=accept+1;
else
end
% Update the estimated optimal solution
f_opt=E_old;
N_iter = N_iter + 1;
end
% Display the final results
disp(strcat('Evaluations :', num2str(totaleval)));
disp(strcat('Best solution:', num2str(best)));
disp(strcat('Best objective:', num2str(f_opt)));

function z=fobj(u)
z = benchmark_func(u, 1);
% Rosenbrock's function with f*=0 at (1,1)
%z=(u(1)-1)^2+100*(u(2)-u(1)^2)^2; 
