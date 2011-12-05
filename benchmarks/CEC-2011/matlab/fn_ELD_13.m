%% ECONOMIC LOAD DISPATCH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Input  => Population of Row vectors (generation units' generations)
%%%  Output => Each is a column vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Total_Cost Cost Total_Penalty] = fn_ELD_13(Input_Population,Display)
%% DATA REQUIRED
[Pop_Size No_of_Units] = size(Input_Population);
Power_Demand = 1800; % in MW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % % ============= 13 unit system data ==========
        %Data1=[Pmin   Pmax     a           b       c       e       f]; 
        Data1=[ 0       680     0.00028     8.1     550     300     0.035;
                0       360     0.00056     8.1     309     200     0.042;
                0       360     0.00056     8.1     307     200     0.042;
                60      180     0.00324     7.74    240     150     0.063;
                60      180     0.00324     7.74    240     150     0.063;
                60      180     0.00324     7.74    240     150     0.063;
                60      180     0.00324     7.74    240     150     0.063;
                60      180     0.00324     7.74    240     150     0.063;
                60      180     0.00324     7.74    240     150     0.063;
                40      120     0.00284     8.6     126     100     0.084;
                40      120     0.00284     8.6     126     100     0.084;
                55      120     0.00284     8.6     126     100     0.084;
                55      120     0.00284     8.6     126     100     0.084;];
        % Data2=[Po     UR      DR      Zone1min    Zone1max     Zone2min   Zone2max];
        Data2=[];
        % Loss Co-efficients
        B1=zeros(No_of_Units,No_of_Units);
        B2=zeros(1,No_of_Units);
        B3=0; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIALIZATIONS
Pmin = Data1(:,1)'; 
Pmax = Data1(:,2)';
a = Data1(:,3)';
b = Data1(:,4)';
c = Data1(:,5)';
e = Data1(:,6)';
f = Data1(:,7)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CALCULATIONS
for i = 1:Pop_Size
    x = Input_Population(i,:);
    Power_Loss  = (x*B1*x') + (B2*x') + B3;
    Power_Loss  = round(Power_Loss *10000)/10000;
%%% Power Balance Penalty Calculation
    Power_Balance_Penalty = abs(Power_Demand + Power_Loss - sum(x));    
%%% Capacity Limits Penalty Calculation
    Capacity_Limits_Penalty = sum(abs(x-Pmin)-(x-Pmin)) + sum(abs(Pmax-x)-(Pmax-x));
%%% Total Penalty Calculation
    Total_Penalty(i,1) = 1e5*Power_Balance_Penalty + 1e3*Capacity_Limits_Penalty;
%%% Cost Calculation
    Cost(i,1) = sum( a.*(x.^2) + b.*x + c + abs(e.*sin(f.*(Pmin-x))) );
    Total_Cost(i,1) = Cost(i,1) + Total_Penalty(i,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (nargin>1)
        disp('----------------------------------------------------------------------------');
        disp(sprintf('13 UNIT SYSTEM'));
        disp(sprintf('Total_Power_Generation   : %17.8f ',sum(x))); 
        disp(sprintf('Power_Balance_Penalty    : %17.8f ',Power_Balance_Penalty));
        disp(sprintf('Capacity_Limits_Penalty  : %17.8f ',Capacity_Limits_Penalty ));
        disp(sprintf('Cost                     : %17.8f ',Cost(i,1)));
        disp(sprintf('Total_Penalty            : %17.8f ',Total_Penalty(i,1)));
        disp(sprintf('Total_Objective_Value    : %17.8f ',Total_Cost(i,1))); 
        disp('----------------------------------------------------------------------------');
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
end
end