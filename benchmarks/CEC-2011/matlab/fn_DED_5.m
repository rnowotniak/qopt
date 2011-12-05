%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DYNAMIC ECONOMIC LOAD DISPATCH (5 UNIT SYSTEM)
%% Guided by : Dr. B.K.Panigrahi, V.Ravikumar Pandi, IIT Delhi
%% Coded by  : Krishnanand K.R., Santanu Kumar Nayak
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Input  => A single Row vector (Thermal generations of each unit at each interval)
%%%  Output => Each is a single value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Total_Value Total_Cost Total_Penalty] = fn_DED_5(Input_Vector,Display) %%[1:24*5]
%% DATA REQUIRED
No_of_Load_Hours=24;
No_of_Units=5;
Input_Generations = [ reshape(Input_Vector,No_of_Units,No_of_Load_Hours)'];
%%
%=============5 unit system data for DED==========
Power_Demand = [410 435 475 530 558 608 626 654 690 704 720 740 704 690 654 580 558 608 654 704 680 605 527 463]; % in MW 
%Data1=[Pmin   Pmax     a           b       c       e       f]; 
Data1=[10       75     0.0080       2.0     25      100     0.042;
       20       125    0.0030       1.8     60      140     0.040;
       30       175    0.0012       2.1     100     160     0.038;
       40       250    0.0010       2.0     120     180     0.037;
       50       300    0.0015       1.8     40      200     0.035;];   
%Data2=[Po     UR       DR      Zone1min    Zone1max     Zone2min   Zone2max];
Data2=[NaN      30       30      10          10              10      10;
       NaN      30       30      20          20              20      20;
       NaN      40       40      30          30              30      30;
       NaN      50       50      40          40              40      40;
       NaN      50       50      50          50              50      50;];
   
% Loss Co-efficients
B1=[0.000049    0.000014    0.000015    0.000015    0.000020;
    0.000014    0.000045    0.000016    0.000020    0.000018;
    0.000015    0.000016    0.000039    0.000010    0.000012;
    0.000015    0.000020    0.000010    0.000040    0.000014;
    0.000020    0.000018    0.000012    0.000014    0.000035;];
B2=zeros(1,5);
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
    x = Input_Generations(j,:);
    Power_Loss(j)  = (x*B1*x') + (B2*x') + B3;
    Power_Loss(j) = round(Power_Loss(j)*10000)/10000;
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
    All_Penalty = 1e3*Power_Balance_Penalty + 1e3*Capacity_Limits_Penalty + 1e5*Ramp_Limits_Penalty + 1e5*POZ_Penalty;
    Total_Penalty = sum(All_Penalty);
    Total_Cost = sum(Current_Cost);
    Total_Value = Total_Cost + Total_Penalty;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin>1)
    disp('----------------------------------------------------------------------------');
    disp(sprintf('5 UNIT SYSTEM : (24*5) SCHEDULE'));
    disp(sprintf('Power_Loss                : %17.8f ',sum(Power_Loss)));
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