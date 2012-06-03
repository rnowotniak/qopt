%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DYNAMIC ECONOMIC LOAD DISPATCH (10 UNIT SYSTEM)
%% Guided by : Dr. B.K.Panigrahi, V.Ravikumar Pandi, IIT Delhi
%% Coded by  : Krishnanand K.R., Santanu Kumar Nayak
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Input  => A single Row vector (Thermal generations of each unit at each interval)
%%%  Output => Each is a single value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Total_Value Total_Cost Total_Penalty] = fn_DED_10(Input_Vector,My_Action) %%[1:24*10]
%% DATA REQUIRED
No_of_Load_Hours=24;
No_of_Units=10;
Input_Generations = reshape(Input_Vector,No_of_Units,No_of_Load_Hours)';
%%
%=============10 unit system data for DED==========
Power_Demand = [1036 1110 1258 1406 1480 1628 1702 1776 1924 2072 2146 2220 2072 1924 1776 1554 1480 1628 1776 2072 1924 1628 1332 1184]; % pattern 2 in MW 
%Data1=[Pmin   Pmax     a           b       c       e       f]; 
Data1= [150     470     0.00043     21.60   958.20  450     0.041;
        135     460     0.00063     21.05   1313.6  600     0.036;
        73      340     0.00039     20.81   604.97  320     0.028;
        60      300     0.00070     23.90   471.60  260     0.052;
        73      243     0.00079     21.62   480.29  280     0.063;
        57      160     0.00056     17.87   601.75  310     0.048;
        20      130     0.00211     16.51   502.7   300     0.086;
        47      120     0.0048      23.23   639.40  340     0.082;
        20      80      0.10908     19.58   455.60  270     0.098;
        55      55      0.00951     22.54   692.4   380     0.094;];
%Data2=[Po     UR       DR      Zone1min    Zone1max     Zone2min   Zone2max];
Data2=[ NaN      80      80;
        NaN      80      80;
        NaN      80      80;
        NaN      50      50;
        NaN      50      50;
        NaN      50      50;
        NaN      30      30;
        NaN      30      30;
        NaN      30      30;
        NaN      30      30;]; 
B1=zeros(10,10);
B2=zeros(1,10);
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
Previous_Generations = Data2(:,1)';
Up_Ramp = Data2(:,2)';
Down_Ramp = Data2(:,3)';
Prohibited_Operating_Zones_POZ = Data2(:,4:end)';
No_of_POZ_Limits = size(Prohibited_Operating_Zones_POZ,1);
POZ_Lower_Limits = Prohibited_Operating_Zones_POZ(1:2:No_of_POZ_Limits,:);
POZ_Upper_Limits = Prohibited_Operating_Zones_POZ(2:2:No_of_POZ_Limits,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Power_Balance_Penalty = zeros(No_of_Load_Hours,1);
Capacity_Limits_Penalty = zeros(No_of_Load_Hours,1);
Up_Ramp_Limit = zeros(No_of_Load_Hours,No_of_Units);
Down_Ramp_Limit = zeros(No_of_Load_Hours,No_of_Units);
Ramp_Limits_Penalty = zeros(No_of_Load_Hours,1);
POZ_Penalty = zeros(No_of_Load_Hours,1);
All_Penalty = zeros(No_of_Load_Hours,1);
Current_Cost = zeros(No_of_Load_Hours,1);
Power_Loss = zeros(No_of_Load_Hours,1);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:No_of_Load_Hours
    %Input_Generations(j,1) = Power_Demand(j) - sum(Input_Generations(j,2:end));
    x = Input_Generations(j,:);
    Power_Loss(j)  = (x*B1*x') + (B2*x') + B3;
%%% Power Balance Penalty Calculation
    Power_Balance_Penalty(j) = abs(Power_Demand(j) + Power_Loss(j) - sum(x));
%%% Capacity Limits Penalty Calculation
    Capacity_Limits_Penalty(j) = sum(abs(x-Pmin)-(x-Pmin)) + sum(abs(Pmax-x)-(Pmax-x));
%%% Ramp Rate Limits Penalty Calculation
    if j>1
        Up_Ramp_Limit(j,:) = min(Pmax,Previous_Generations+Up_Ramp);
        Down_Ramp_Limit(j,:) = max(Pmin,Previous_Generations-Down_Ramp);
        Ramp_Limits_Penalty(j) = sum(abs(x-Down_Ramp_Limit(j,:))-(x-Down_Ramp_Limit(j,:))) + sum(abs(Up_Ramp_Limit(j,:)-x)-(Up_Ramp_Limit(j,:)-x));                
    end
    Previous_Generations = x;
%%% Prohibited Operating Zones Penalty Calculation
    temp_x = repmat(x,No_of_POZ_Limits/2,1);
    POZ_Penalty(j) = sum(sum((POZ_Lower_Limits<temp_x & temp_x<POZ_Upper_Limits).*min(temp_x-POZ_Lower_Limits,POZ_Upper_Limits-temp_x)));
%%% Cost Calculation
    Current_Cost(j) = sum( a.*(x.^2) + b.*x + c + abs(e.*sin(f.*(Pmin-x))) );
end
%%% All & Total Penalty Calculation
    All_Penalty = 1e3*Power_Balance_Penalty + 1e3*Capacity_Limits_Penalty + 1e3*Ramp_Limits_Penalty + 1e5*POZ_Penalty;
    Total_Penalty = sum(All_Penalty);
    Total_Cost = sum(Current_Cost);
    Total_Value = Total_Cost + Total_Penalty;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin>1)
    disp('----------------------------------------------------------------------------');
    disp(sprintf('10 UNIT SYSTEM : (24*10) SCHEDULE'));
    disp(sprintf('Power_Balance_Penalty     : %17.8f ',sum(Power_Balance_Penalty)));
    disp(sprintf('Capacity_Limits_Penalty   : %17.8f ',sum(Capacity_Limits_Penalty)));
    disp(sprintf('Ramp_Limits_Penalty       : %17.8f ',sum(Ramp_Limits_Penalty)));     
    disp(sprintf('POZ_Penalty               : %17.8f ',sum(POZ_Penalty)));    
    disp(sprintf('Cost                      : %17.8f ',Total_Cost));
    disp(sprintf('Total_Penalty             : %17.8f ',Total_Penalty));
    disp(sprintf('Total_Objective_Value     : %17.8f ',Total_Value));
    disp('----------------------------------------------------------------------------');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end