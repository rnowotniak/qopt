% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    

% The Particle Swarm Optimization
% (written by X S Yang, Cambridge University)
% Usage: pso(number_of_particles,Num_iterations)
%  eg:   best=pso_demo(20,10);
% where best=[xbest ybest zbest]  %an n by 3 matrix
%   xbest(i)/ybest(i) are the best at ith iteration

function [best]=pso_demo(n, Num_iterations)
% n=number of particles
% Num_iterations=total number of iterations
if nargin<2,   Num_iterations=20;  end
if nargin<1,   n=25;          end
% Michalewicz Function f*=-1.801 at [2.20319,1.57049]
% Splitting two parts to avoid long lines in printing
str1='-sin(x)*(sin(x^2/3.14159))^20';
str2='-sin(y)*(sin(2*y^2/3.14159))^20';
funstr=strcat(str1,str2);
% Converting to an inline function and vectorization
f=vectorize(inline(funstr));
% range=[xmin xmax ymin ymax];
range=[0 4 0 4];
% ----------------------------------------------------
% Setting the parameters: alpha, beta
% Random amplitude of roaming particles alpha=[0,1]
% alpha=gamma^t=0.7^t;
% Speed of convergence (0->1)=(slow->fast)
beta=0.5;
% ----------------------------------------------------
% Grid values of the objective function
% These values are used for visualization only
Ngrid=100;
dx=(range(2)-range(1))/Ngrid;
dy=(range(4)-range(3))/Ngrid;
xgrid=range(1):dx:range(2); ygrid=range(3):dy:range(4);
[x,y]=meshgrid(xgrid,ygrid);
z=f(x,y);
% Display the shape of the function to be optimized
figure(1);
surfc(x,y,z);
% ---------------------------------------------------
best=zeros(Num_iterations,3);   % initialize history
% ----- Start Particle Swarm Optimization -----------
% generating the initial locations of n particles
[xn,yn]=init_pso(n,range);
% Display the paths of particles in a figure
% with a contour of the objective function
 figure(2);
% Start iterations
for i=1:Num_iterations,
% Show the contour of the function
  contour(x,y,z,15); hold on;
% Find the current best location (xo,yo)
zn=f(xn,yn);
zn_min=min(zn);
xo=min(xn(zn==zn_min));
yo=min(yn(zn==zn_min));
zo=min(zn(zn==zn_min));
% Trace the paths of all roaming particles
% Display these roaming particles
plot(xn,yn,'.',xo,yo,'*'); axis(range);
% The accelerated PSO with alpha=gamma^t
  gamma=0.7; alpha=gamma.^i;
% Move all the particles to new locations
[xn,yn]=pso_move(xn,yn,xo,yo,alpha,beta,range);
drawnow;
% Use "hold on" to display paths of particles
hold off;
% History
best(i,1)=xo; best(i,2)=yo; best(i,3)=zo;
end   %%%%% end of iterations
% ----- All subfunctions are listed here -----
% Intial locations of n particles
function [xn,yn]=init_pso(n,range)
xrange=range(2)-range(1); yrange=range(4)-range(3);
xn=rand(1,n)*xrange+range(1);
yn=rand(1,n)*yrange+range(3);
% Move all the particles toward (xo,yo)
function [xn,yn]=pso_move(xn,yn,xo,yo,a,b,range)
nn=size(yn,2);  %a=alpha, b=beta
xn=xn.*(1-b)+xo.*b+a.*(rand(1,nn)-0.5);
yn=yn.*(1-b)+yo.*b+a.*(rand(1,nn)-0.5);
[xn,yn]=findrange(xn,yn,range);
% Make sure the particles are within the range
function [xn,yn]=findrange(xn,yn,range)
nn=length(yn);
for i=1:nn,
   if xn(i)<=range(1), xn(i)=range(1); end
   if xn(i)>=range(2), xn(i)=range(2); end
   if yn(i)<=range(3), yn(i)=range(3); end
   if yn(i)>=range(4), yn(i)=range(4); end
end

