% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    

% Harmony Search (Simple Demo) Matlab Program
% Written by X S Yang (Cambridge University)
% Usage: hs_simple
% or     hs_simple(`x^2+(y-5)^2',25000);
function [solution,fbest]=hs_simple(funstr,MaxAttempt)

global initial_flag;
initial_flag = 0;

help hs_simple.m
disp('It may take a few minutes ...');
% MaxAttempt=25000;  % Max number of Attempt
if nargin<2, MaxAttempt=25000; end
if nargin<1,
% Rosenbrock's Banana function with the
% global fmin=0 at (1,1).
funstr = '(1-x1)^2+100*(x2-x1^2)^2';
end
% Converting to an inline function
f=vectorize(inline(funstr));
ndim=2;  %Number of independent variables
% The range of the objective function
range(1,:)=[-10 10]; range(2,:)=[-10 10];
% Pitch range for pitch adjusting
pa_range=[200 200];
% Initial parameter setting
HS_size=20;        %Length of solution vector
HMacceptRate=0.95; %HM Accepting Rate
PArate=0.7;        %Pitch Adjusting rate
% Generating Initial Solution Vector
for i=1:HS_size,
   for j=1:ndim,
   x(j)=range(j,1)+(range(j,2)-range(j,1))*rand;
   end
   HM(i, :) = x;
   HMbest(i) = f(x(1), x(2));
end %% for i
% Starting the Harmony Search
for count = 1:MaxAttempt,
  for j = 1:ndim,
    if (rand >= HMacceptRate)
      % New Search via Randomization
      x(j)=range(j,1)+(range(j,2)-range(j,1))*rand;
    else
      % Harmony Memory Accepting Rate
      x(j) = HM(fix(HS_size*rand)+1,j);
      if (rand <= PArate)
      % Pitch Adjusting in a given range
      pa=(range(j,2)-range(j,1))/pa_range(j);
      x(j)= x(j)+pa*(rand-0.5);
      end
    end
  end %% for j
  % Evaluate the new solution
   fbest = f(x(1), x(2));
  % Find the best in the HS solution vector
   HSmaxNum = 1; HSminNum=1;
   HSmax = HMbest(1); HSmin=HMbest(1);
   for i = 2:HS_size,
      if HMbest(i) > HSmax,
        HSmaxNum = i;
        HSmax = HMbest(i);
      end
      if HMbest(i)<HSmin,
         HSminNum=i;
         HSmin=HMbest(i);
      end
   end
   % Updating the current solution if better
   if fbest < HSmax,
       HM(HSmaxNum, :) = x;
       HMbest(HSmaxNum) = fbest;
   end
   solution=x;  % Record the solution
end %% (end of harmony search)

