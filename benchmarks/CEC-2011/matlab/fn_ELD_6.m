%% ECONOMIC LOAD DISPATCH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Input  => Population of Row vectors (generation units' generations)
%%%  Output => Each is a column vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Total_Cost Cost Total_Penalty] = fn_ELD_6(Input_Population,Display)
%% DATA REQUIRED
[Pop_Size No_of_Units] = size(Input_Population);
Power_Demand = 1263; %% in MW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % % ============= 6 unit system data ==========
        % Data1=  [Pmin   Pmax     a       b           c]; 
        Data1=[ 100     500     0.0070  7.0         240;
                50      200     0.0095  10.0        200;
                80      300     0.0090  8.5         220;
                50      150     0.0090  11.0        200;
                50      200     0.0080  10.5        220;
                50      120     0.0075  12.0        190;];
        % Data2=[Po     UR      DR      Zone1min    Zone1max     Zone2min   Zone2max];
        Data2=[ 440     80      120     210         240         350         380;
                170     50      90      90          110         140         160;
                200     65      100     150         170         210         240;
                150     50      90      80          90          110         120;
                190     50      90      90          110         140         150;
                150     50      90      75          85          100         105;];
        % Loss Co-efficients
        B1=[ 1.7     1.2     0.7     -0.1     -0.5    -0.2;
             1.2     1.4     0.9      0.1     -0.6    -0.1;
             0.7     0.9     3.1      0.0     -1.0    -0.6;
            -0.1     0.1     0.0      0.24    -0.6    -0.8;
            -0.5    -0.6    -0.1     -0.6     12.9    -0.2;
             0.2    -0.1    -0.6     -0.8     -0.2    15.0;];
        B1=B1.*10^-5;
        B2=[-0.3908 -0.1297 0.7047 0.0591   0.2161  -0.6635].*10^-5;
        B3=0.0056*10^-2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIALIZATIONS
Pmin = Data1(:,1)'; 
Pmax = Data1(:,2)';
a = Data1(:,3)';
b = Data1(:,4)';
c = Data1(:,5)';
Initial_Generations = Data2(:,1)';
Up_Ramp = Data2(:,2)';
Down_Ramp = Data2(:,3)';
Up_Ramp_Limit = min(Pmax,Initial_Generations+Up_Ramp);
Down_Ramp_Limit = max(Pmin,Initial_Generations-Down_Ramp);
Prohibited_Operating_Zones_POZ = Data2(:,4:end)';
No_of_POZ_Limits = size(Prohibited_Operating_Zones_POZ,1);
POZ_Lower_Limits = Prohibited_Operating_Zones_POZ(1:2:No_of_POZ_Limits,:);
POZ_Upper_Limits = Prohibited_Operating_Zones_POZ(2:2:No_of_POZ_Limits,:);
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
%%% Ramp Rate Limits Penalty Calculation
    Ramp_Limits_Penalty = sum(abs(x-Down_Ramp_Limit)-(x-Down_Ramp_Limit)) + sum(abs(Up_Ramp_Limit-x)-(Up_Ramp_Limit-x));
%%% Prohibited Operating Zones Penalty Calculation
    temp_x = repmat(x,No_of_POZ_Limits/2,1);
    POZ_Penalty = sum(sum((POZ_Lower_Limits<temp_x & temp_x<POZ_Upper_Limits).*min(temp_x-POZ_Lower_Limits,POZ_Upper_Limits-temp_x)));
%%% Total Penalty Calculation
    Total_Penalty(i,1) = 1e3*Power_Balance_Penalty + 1e3*Capacity_Limits_Penalty + 1e5*Ramp_Limits_Penalty + 1e5*POZ_Penalty;
%%% Cost Calculation
    Cost(i,1) = sum( a.*(x.^2) + b.*x + c );
    Total_Cost(i,1) = Cost(i,1) + Total_Penalty(i,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (nargin>1)
        disp('----------------------------------------------------------------------------');
        disp(sprintf('6 UNIT SYSTEM'));
        disp(sprintf('Power_Loss               : %17.8f ',Power_Loss));
        disp(sprintf('Total_Power_Generation   : %17.8f ',sum(x))); 
        disp(sprintf('Power_Balance_Penalty    : %17.8f ',Power_Balance_Penalty));
        disp(sprintf('Capacity_Limits_Penalty  : %17.8f ',Capacity_Limits_Penalty ));
        disp(sprintf('Ramp_Limits_Penalty      : %17.8f ',Ramp_Limits_Penalty));
        disp(sprintf('POZ_Penalty              : %17.8f ',POZ_Penalty));
        disp(sprintf('Cost                     : %17.8f ',Cost(i,1)));
        disp(sprintf('Total_Penalty            : %17.8f ',Total_Penalty(i,1)));
        disp(sprintf('Total_Objective_Value    : %17.8f ',Total_Cost(i,1))); 
        disp('----------------------------------------------------------------------------');
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
end
end