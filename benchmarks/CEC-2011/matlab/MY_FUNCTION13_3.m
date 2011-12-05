%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HYDROTHERMAL SCHEDULING (Fitness Function)
%% Guided by : Dr. P.K. Rout, SOA University
%% Coded by  : Krishnanand K.R., Santanu Kumar Nayak
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fitness Function to be called from the optimization algorithm
%% Evaluates a population of row vectors
function [y Count] = MY_FUNCTION13_3(input_array)
siz = size(input_array,1);
Count = siz ;
y = zeros(siz,1);
for i =1:siz
    y(i,1) =  Fn_Eval(input_array(i,:));
end
end
%% Evaluates a single row vector
function y = Fn_Eval(x)
x=round(x*10000)/10000; %% For fixing the 4 digit precision
y = fn_HT_ELD_Case_3(x); %% Change Here to Call a Different System
end
%%