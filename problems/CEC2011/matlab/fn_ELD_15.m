%% ECONOMIC LOAD DISPATCH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Input  => Population of Row vectors (generation units' generations)
%%%  Output => Each is a column vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Total_Cost Cost Total_Penalty] = fn_ELD_15(Input_Population,Display)
%% DATA REQUIRED
[Pop_Size No_of_Units] = size(Input_Population);
Power_Demand = 2630; %% in MW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % % ============= 15 unit system data ==========
        % Data1=  [Pmin   Pmax     a       b           c];                
        Data1=[ 150     455     0.000299   10.1     671;
                150     455     0.000183   10.2     574;
                20      130     0.001126    8.8     374;
                20      130     0.001126    8.8     374;
                150     470     0.000205   10.4     461;
                135     460     0.000301   10.1     630;
                135     465     0.000364    9.8     548;
                60      300     0.000338   11.2     227;
                25      162     0.000807   11.2     173;
                25      160     0.001203   10.7     175;
                20      80      0.003586   10.2     186;
                20      80      0.005513    9.9     230;
                25      85      0.000371   13.1     225;
                15      55      0.001929   12.1    309;
                15      55      0.004447   12.4    323;];
        % Data2=[Po     UR      DR      Zone1min    Zone1max     Zone2min   Zone2max     Zone3min   Zone3max];
      Data2=[   400     80      120     150         150             150         150         150         150;
                300     80      120     185         255             305         335         420         450;
                105     130     130      20         20              20          20          20          20;   
                100     130     130      20         20              20          20          20          20;
                90      80      120     180         200             305         335         390         420;
                400     80      120     230         255             365         395         430         455;
                350     80      120     135         135             135         135         135         135;
                95      65      100     60          60              60          60          60          60;
                105     60      100     25          25              25          25          25          25;
                110     60      100     25          25              25          25          25          25;
                60      80      80      20          20              20          20          20          20;
                40      80      80      30          40              55          65          20          20;
                30      80      80      25          25              25          25          25          25;
                20      55      55      15          15              15          15          15          15;
                20      55      55      15          15              15          15          15          15;];
        % Loss Co-efficients
       B1=[     1.4     1.2     0.7     -0.1     -0.3    -0.1    -0.1    -0.1    -0.3    -0.5    -0.3    -0.2    0.4     0.3     -0.1;
                 1.2     1.5     1.3      0.0     -0.5    -0.2     0.0     0.1    -0.2     -0.4   -0.4    0.0     0.4     1       -0.2;
                 0.7     1.3     7.6     -0.1     -1.3    -0.9    -0.1     0.0    -0.8    -1.2    -1.7    0.0     -2.6    11.1    -2.8;
                -0.1     0.0    -0.1      3.4     -0.7    -0.4     1.1     5.0     2.9     3.2     -1.1    0.0     0.1     0.1     -2.6;
                -0.3    -0.5    -1.3     -0.7      9.0     1.4    -0.3    -1.2    -1.0    -1.3    0.7     -0.2    -0.2    -2.4    -0.3;
                -0.1    -0.2    -0.9     -0.4      1.4     1.6     0.0    -0.6    -0.5    -0.8    1.1     -0.1    -0.2    -1.7    0.3;
                -0.1     0.0    -0.1      1.1     -0.3     0.0     1.5     1.7     1.5     0.9     -0.5    0.7     0.0     -0.2    -0.8;
                -0.1     0.1     0.0      5.0     -1.2    -0.6     1.7     16.8    8.2     7.9     -2.3    -3.6    0.1     0.5     -7.8;
                -0.3    -0.2    -0.8      2.9     -1.0    -0.5     1.5     8.2    12.9    11.6    -2.1    -2.5    0.7     -1.2    -7.2;
                -0.5    -0.4    -1.2      3.2     -1.3    -0.8     0.9     7.9    11.6    20.0    -2.7    -3.4    0.9     -1.1    -8.8;
                -0.3    -0.4    -1.7     -1.1      0.7     1.1    -0.5    -2.3    -2.1    -2.7    14.0    0.1     0.4     -3.8    16.8;
                -0.2     0.0     0.0      0.0     -0.2    -0.1     0.7    -3.6    -2.5    -3.4    0.1     5.4     -0.1    -0.4    2.8;
                 0.4     0.4    -2.6      0.1     -0.2    -0.2     0.0     0.1     0.7     0.9     0.4     -0.1    10.3    -10.1   2.8;
                 0.3     1.0    11.1      0.1     -2.4    -1.7    -0.2     0.5    -1.2    -1.1    -3.8    -0.4    -10.1   57.8    -9.4;
                -0.1    -0.2    -2.8     -2.6     -0.3     0.3    -0.8    -7.8    -7.2    -8.8    16.8    2.8     2.8     -9.4    128.3;];

         B1=B1.*10^-5;
         B2=[-0.1    -0.2    2.8    -0.1    0.1     -0.3    -0.2    -0.2    0.6     3.9     -1.7    0.0     -3.2    6.7     -6.4].*10^-5;
         B3=0.0055*10^-2;       
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
        disp(sprintf('15 UNIT SYSTEM'));
        disp(sprintf('Power_Loss               : %17.8f ',Power_Loss));
        disp(sprintf('Total_Power_Generation   : %17.8f ',sum(x)));        
        disp(sprintf('Total_Power_Required     : %17.8f ',Power_Loss+Power_Demand)); 
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