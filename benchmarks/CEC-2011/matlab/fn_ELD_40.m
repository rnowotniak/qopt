%% ECONOMIC LOAD DISPATCH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Input  => Population of Row vectors (generation units' generations)
%%%  Output => Each is a column vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Total_Cost Cost Total_Penalty] = fn_ELD_40(Input_Population,Display)
%% DATA REQUIRED
[Pop_Size No_of_Units] = size(Input_Population);
Power_Demand = 10500; % in MW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % % ============= 40 unit system data ==========
        %Data1=[Pmin   Pmax     a           b       c       e       f]; 
        Data1=[ 36      114     0.0069      6.73    94.705  100     0.084;       
                36      114     0.0069      6.73    94.705  100     0.084;       
                60      120     0.02028     7.07    309.54  100     0.084;
                80      190     0.00942     8.18    369.03  150     0.063;
                47      97      0.0114      5.35    148.89  120     0.077;
                68      140     0.01142     8.05    222.33  100     0.084;
                110     300     0.00357     8.03    287.71  200     0.042;
                135     300     0.00492     6.99    391.98  200     0.042;
                135     300     0.00573     6.6     455.76  200     0.042;
                130     300     0.00605     12.9    722.82  200     0.042;
                94      375     0.00515     12.9    635.2   200     0.042;
                94      375     0.00569     12.8    654.69  200     0.042;
                125     500     0.00421     12.5    913.4   300     0.035;
                125     500     0.00752     8.84    1760.4  300     0.035;
                125     500     0.00708     9.15    1728.3  300     0.035;
                125     500     0.00708     9.15    1728.3  300     0.035;
                220     500     0.00313     7.97    647.85  300     0.035;
                220     500     0.00313     7.95    649.69  300     0.035;
                242     550     0.00313     7.97    647.83  300     0.035;
                242     550     0.00313     7.97    647.81  300     0.035;
                254     550     0.00298     6.63    785.96  300     0.035;
                254     550     0.00298     6.63    785.96  300     0.035;
                254     550     0.00284     6.66    794.53  300     0.035;
                254     550     0.00284     6.66    794.53  300     0.035;
                254     550     0.00277     7.1     801.32  300     0.035;
                254     550     0.00277     7.1     801.32  300     0.035;
                10      150     0.52124     3.33    1055.1  120     0.077;
                10      150     0.52124     3.33    1055.1  120     0.077;
                10      150     0.52124     3.33    1055.1  120     0.077;
                47      97      0.0114      5.35    148.89  120     0.077;
                60      190     0.0016      6.43    222.92  150     0.063;
                60      190     0.0016      6.43    222.92  150     0.063;
                60      190     0.0016      6.43    222.92  150     0.063;
                90      200     0.0001      8.95    107.87  200     0.042;
                90      200     0.0001      8.62    116.58  200     0.042;
                90      200     0.0001      8.62    116.58  200     0.042;
                25      110     0.0161      5.88    307.45  80      0.098;
                25      110     0.0161      5.88    307.45  80      0.098;
                25      110     0.0161      5.88    307.45  80      0.098;
                242     550     0.00313     7.97    647.83  300     0.035;];
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
        disp(sprintf('40 UNIT SYSTEM'));
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