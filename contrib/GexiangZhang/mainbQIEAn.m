% bQIEAn for knapsack problems
% main program

clear all
clc
format long;

initialTime=cputime;
runs=30;
MAXgen=1000;
item=400;
bestsolution=zeros(1,runs);

for i=1:runs
    rrun=i
    [result]=bQIEAn(item,MAXgen)
    bestsolution(1,i)=result;
end
elapsedtime=(cputime-time1)/runs;

save result bestsolution elapsedtime;
