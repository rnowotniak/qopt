%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Hydrothermal System : four Hydro Units & one thermal unit
%% Guided by : Pravat Kumar Rout, Silicon Institute of Technology
%% Coded by  : Krishnanand K.R., Santanu Kumar Nayak
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Input    => Single Row vector (Discharge Rates bounded by Qmin & Qmax)
%%%  Outputs  => Each one is a single value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Total_Value Total_Cost Total_Penalty] = fn_HT_ELD_Case_2(Input_Vector,My_Action) %% fn_HT_ELD([1:24*4])
%% DATA REQUIRED
No_of_Load_Hours=24;
No_of_Units=4;
Input_Discharges = reshape(Input_Vector,No_of_Units,No_of_Load_Hours);
%%
Power_Demand = [1370 1390 1360 1290 1290 1410 1650 2000 2240 2320 2230 2310 2230 2200 2130 2070 2130 2140 2240 2280 2240 2120 1850 1590]; %% in MW
C_Coefficients=[   -0.0042   -0.42    0.030    0.90   10.0  -50;
                   -0.0040   -0.30    0.015    1.14    9.5  -70;
                   -0.0016   -0.30    0.014    0.55    5.5  -40;
                   -0.0030   -0.31    0.027    1.44   14.0  -90];     
Inflow_Rate = [10	9	 8	  7	 6	7	8	9	10	11	12	10	11	12	11	10	9	8	7	6	7	8	9	10;
                8	8	 9	  9	 8	7	6	7	8	9	9	8	8	9	9	8	7	6	7	8	9	9	8	8;
               8.1	8.2	 4	  2	 3	4	3	2	1	1	1	2	4	3	3	2	2	2	1	1	2	2	1	0;
               2.8	2.4	 1.6  0	 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
Spillages = zeros(No_of_Units,No_of_Load_Hours);           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIALIZATIONS
Ptmin = [500];
Ptmax = [2500];
PHmin = [0 0 0 0];%Data1(:,1)';
PHmax = [500 500 500 500];%Data1(:,2)';
c1 = C_Coefficients(:,1)';%Data1(:,3)';
c2 = C_Coefficients(:,2)';%Data1(:,4)';
c3 = C_Coefficients(:,3)';%Data1(:,5)';
c4 = C_Coefficients(:,4)';%Data1(:,6)';
c5 = C_Coefficients(:,5)';%Data1(:,7)';
c6 = C_Coefficients(:,6)';%Data1(:,8)';
%%
Delay_Time = [2 3 4 0];
No_of_Upstreams = [0 0 2 1];
Vmin = [80 60 100 70];
Vmax = [150 120 240 160];
V_Initial = [100 80 170 120];
V_Final = [120 70 170 140];
Qmin = [5 6 10 13];
Qmax = [15 15 30 25];
%%
Prohibited_Operating_Zones_POZ = [  8   9;     %Data2(:,4:end)';
                                    7   8;
                                   22  27;
                                   16  18]';
No_of_POZ_Limits = size(Prohibited_Operating_Zones_POZ,1);
POZ_Lower_Limits = Prohibited_Operating_Zones_POZ(1:2:No_of_POZ_Limits,:);
POZ_Upper_Limits = Prohibited_Operating_Zones_POZ(2:2:No_of_POZ_Limits,:);
%%
Storage_Volume = [V_Initial' zeros(No_of_Units,No_of_Load_Hours)];
Max_Delay = max(Delay_Time);
Initial_Discharges = zeros(No_of_Units,Max_Delay);
Initial_Spillages = zeros(No_of_Units,Max_Delay);
All_Spillages = [Initial_Spillages  Spillages];
All_Discharges = [Initial_Discharges  Input_Discharges];
Upstream_Carry = zeros(No_of_Units,No_of_Load_Hours);
for i=1:No_of_Units
    for j=1:No_of_Load_Hours
        Upstream_Volume=0;
        for k=(i-No_of_Upstreams(i)):(i-1)
            Upstream_Volume=Upstream_Volume+All_Discharges(k,j+Max_Delay-Delay_Time(k))+ All_Spillages(k,j+Max_Delay-Delay_Time(k));
        end
        Upstream_Carry(i,j)=Upstream_Volume;
        Storage_Volume(i,j+1)=Storage_Volume(i,j)+Upstream_Volume-Input_Discharges(i,j)-Spillages(i,j)+ Inflow_Rate(i,j);
    end
end
%%
Discharge_Rate_Limits_Penalty = zeros(No_of_Load_Hours,1);
Storage_Volume_Limits_Penalty = zeros(No_of_Load_Hours,1);
POZ_Penalty = zeros(No_of_Load_Hours,1);
Capacity_Limits_Penalty_H = zeros(No_of_Load_Hours,1);
Capacity_Limits_Penalty_T = zeros(No_of_Load_Hours,1);
Power_Balance_Penalty = zeros(No_of_Load_Hours,1);
Current_Penalty = zeros(No_of_Load_Hours,1);
Current_Cost = zeros(No_of_Load_Hours,1);
Power_Loss = zeros(No_of_Load_Hours,1);
Hydro_Generations = zeros(No_of_Load_Hours,No_of_Units);
Thermal_Generations = zeros(No_of_Load_Hours,1);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:No_of_Load_Hours
    q = Input_Discharges(:,j)';
    v = Storage_Volume(:,j+1)';
    %%% Water Discharge Rate Limits Penalty Calculation 
    Discharge_Rate_Limits_Penalty(j) = sum(abs(q-Qmin)-(q-Qmin)) + sum(abs(Qmax-q)-(Qmax-q));
    %%% Reservoir Storage Volume Limits Penalty Calculation 
    Storage_Volume_Limits_Penalty(j) = sum(abs(v-Vmin)-(v-Vmin)) + sum(abs(Vmax-v)-(Vmax-v));
    %%% Prohibited Operating Zones Penalty Calculation
    temp_q = repmat(q,No_of_POZ_Limits/2,1);
    POZ_Penalty(j) = sum(sum((POZ_Lower_Limits<temp_q & temp_q<POZ_Upper_Limits).*min(temp_q-POZ_Lower_Limits,POZ_Upper_Limits-temp_q)));    
    %%% Hydro Plants' Generation Calculation
    Ph = c1.*(v.^2) + c2.*(q.^2) + c3.*(v.*q) + c4.*(v) + c5.*(q) + c6 ;
    Ph = Ph.*(Ph>0); %%% SPECIAL CHECKING OF LIMITS { Ph(i)=0 if -ve }
    Hydro_Generations(j,:) = Ph;
    %%% Capacity Limits Penalty Calculation of Hydro Plants
    Capacity_Limits_Penalty_H(j) = sum(abs(Ph-PHmin)-(Ph-PHmin)) + sum(abs(PHmax-Ph)-(PHmax-Ph));  
    %%% Thermal Plant's Generation Calculation
    P_Thermal = Power_Demand(j) + Power_Loss(j) - sum(Ph);
    Thermal_Generations(j) = P_Thermal;
    %%% Capacity Limits Penalty Calculation of Thermal Plants
    Capacity_Limits_Penalty_T(j)=sum(abs(P_Thermal-Ptmin)-(P_Thermal-Ptmin))+sum(abs(Ptmax-P_Thermal)-(Ptmax-P_Thermal));
    %%% Power Balance Penalty Calculation
    Power_Balance_Penalty(j) = abs(Power_Demand(j) + Power_Loss(j) - sum(Ph) - P_Thermal);
    %%% Cost Calculation %%CHANGE HERE FOR DIFFERENT CASES
    Current_Cost(j) = 5000 + 19.2*P_Thermal + 0.002*(P_Thermal^2) ;%+ abs(700*sin(0.085*(Ptmin-P_Thermal))); 
end
%%
%%% Penalty Calculation %%CHANGE HERE FOR DIFFERENT CASES
All_Penalty = 1e4*Power_Balance_Penalty + 1e4*Capacity_Limits_Penalty_H + 1e4*Capacity_Limits_Penalty_H + ...
              1e4*Discharge_Rate_Limits_Penalty + 1e5*Storage_Volume_Limits_Penalty + 1e5*POZ_Penalty;

Reservoir_End_Limits_Penalty = sum(abs(Storage_Volume(:,1)-V_Initial')) + sum(abs(Storage_Volume(:,end)-V_Final'));

Total_Penalty = sum(All_Penalty) + 1e6*Reservoir_End_Limits_Penalty;
Total_Cost = sum(Current_Cost);
Total_Value = Total_Cost + Total_Penalty;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin==2)
    disp('----------------------------------------------------------------------------');
    disp(sprintf('CASE 2 : With Prohibited Operating Zones and Without Valve-point Loading')); 
    disp(sprintf('Power_Balance_Penalty     : %17.8f ',sum(Power_Balance_Penalty)));
    disp(sprintf('Capacity_Limits_Penalty_H : %17.8f ',sum(Capacity_Limits_Penalty_H)));
    disp(sprintf('Capacity_Limits_Penalty_T : %17.8f ',sum(Capacity_Limits_Penalty_T)));
    disp(sprintf('Discharge_Limits_Penalty  : %17.8f ',sum(Discharge_Rate_Limits_Penalty)));
    disp(sprintf('Storage_Limits_Penalty    : %17.8f ',sum(Storage_Volume_Limits_Penalty)));    
    disp(sprintf('POZ_Penalty               : %17.8f ',sum(POZ_Penalty)));    
    disp(sprintf('Reservoir_Limits_Penalty  : %17.8f ',sum(Reservoir_End_Limits_Penalty)));
    disp(sprintf('Total_Penalty             : %17.8f ',Total_Penalty));
    disp(sprintf('Total_Cost                : %17.8f ',Total_Cost));
    disp(sprintf('Total_Objective_Value     : %17.8f ',Total_Value));  
    disp('----------------------------------------------------------------------------');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end