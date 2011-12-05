% function for bQIEAcm

function [result]=bQIEAcm(item,MAXgen)

Ps=20;   % population size
PsL=Ps*item; % the number of Q-bits
q=zeros(2,item); % Q-bits for one individual
P=zeros(1,PsL);  % Q-bits for the population
eachprofit=zeros(1,Ps); % fitness
bestprofit=zeros(1,MAXgen); % best fitness
gen=1;   % initial generation

% loading data of knapsack problems
load knapsack;

% initialization Q(gen)
for j=1:item
    q(1,j)=sign(-1+2*rand(1))*(1/sqrt(2));
    q(2,j)=sign(-1+2*rand(1))*(1/sqrt(2));
end
Q=q;
for kki=1:Ps-1
    for j=1:item
        q(1,j)=sign(-1+2*rand(1))*1/sqrt(2);
        q(2,j)=sign(-1+2*rand(1))*1/sqrt(2);
    end
    Q=[Q q];
end

% construct P(gen)
for i=1:PsL
    rr=rand(1);
    if rr<Q(2,i)^2
        bit=1;
    else
        bit=0;
    end
    P(1,i)=bit;
end

% evaluation
maxfit=-9999;
maxbit=zeros(1,item);
for i=1:Ps
    % repair
    if sum(weight.*P(1,(i-1)*item+1:i*item))>capacity
        knapscakoverfilled=0;
        if sum(weight.*P(1,(i-1)*item+1:i*item))>capacity
            knapscakoverfilled=1;
        end
        while (knapscakoverfilled>0.5)
            position=1+floor(rand(1)*item);
            P(1,(i-1)*item+position)=0;
            if sum(weight.*P(1,(i-1)*item+1:i*item))<=capacity
                knapscakoverfilled=0;
            end
        end
        while (knapscakoverfilled<0.5)
            position=1+floor(rand(1)*item);
            P(1,(i-1)*item+position)=1;
            if sum(weight.*P(1,(i-1)*item+1:i*item))>capacity
                knapscakoverfilled=1;
            end               
        end          
        P(1,(i-1)*item+position)=0;
    end
    % calculating fitness
    eachprofit(1,i)=sum(profit.*P(1,(i-1)*item+1:i*item));
    if eachprofit(1,i)>maxfit
        maxfit=eachprofit(1,i);
        maxbit=P(1,(i-1)*item+1:i*item);
    end
end

bestprofit(1,gen)=maxfit;
currentmaxbit=maxbit;

while (gen<MAXgen)
    gen=gen+1;
    %construct P(gen)
    for i=1:PsL
        rr=rand(1);
        if rr<Q(2,i)^2
            bit=1;
        else
            bit=0;
        end
        P(1,i)=bit;
    end
    
    maxfit=-9999;
    maxbit=zeros(1,item);
    for i=1:Ps
        % repair
        if sum(weight.*P(1,(i-1)*item+1:i*item))>capacity
            knapscakoverfilled=0;
            if sum(weight.*P(1,(i-1)*item+1:i*item))>capacity
                knapscakoverfilled=1;
            end
            while (knapscakoverfilled>0.5)
                position=1+floor(rand(1)*item);
                P(1,(i-1)*item+position)=0;
                if sum(weight.*P(1,(i-1)*item+1:i*item))<=capacity
                    knapscakoverfilled=0;
                end
            end
            while (knapscakoverfilled<0.5)
                position=1+floor(rand(1)*item);
                P(1,(i-1)*item+position)=1;
                if sum(weight.*P(1,(i-1)*item+1:i*item))>capacity
                    knapscakoverfilled=1;
                end               
            end          
            P(1,(i-1)*item+position)=0;
        end
        % calculating fitness
        eachprofit(1,i)=sum(profit.*P(1,(i-1)*item+1:i*item));
        if eachprofit(1,i)>maxfit
            maxfit=eachprofit(1,i);
            maxbit=P(1,(i-1)*item+1:i*item);
        end
    end
    
    if (maxfit>bestprofit(1,gen-1))
        bestprofit(1,gen)=maxfit;
        currentmaxbit=maxbit;
    else
        bestprofit(1,gen)=bestprofit(1,gen-1);
    end

    %update Q(gen)
    for k=1:Ps
        midbit=P(1,(i-1)*item+1:item*i);
        fx=eachprofit(1,k);
        for j=1:item
            x=midbit(1,j);
            b=currentmaxbit(1,j);
            aa=Q(1,(k-1)*item+j);
            bb=Q(2,(k-1)*item+j);
            fb=bestprofit(1,gen);
            therta=QgatebQIEAcm(x,b,aa,bb,fx,fb);
            u=[cos(therta) -sin(therta)
                sin(therta) cos(therta)];
            Q(1:2,(k-1)*item+j)=u*Q(1:2,(k-1)*item+j);
        end
    end
    
    % crossover
    pc=0.8;
    for i=1:Ps/2
        pcross=rand(1);
        if pcross<pc
            position=1+floor(item*rand(1));
            temp=Q(:,(i-1)*item+position);
            Q(:,(i-1)*item+position)=Q(:,(Ps-i)*item+position);
            Q(:,(Ps-i)*item+position)=temp;
        end
    end
    
    % mutation
    pm=0.05;
    for i=1:Ps
        pmutation=rand(1);
        if pmutation<pm
            position=1+floor(item*rand(1));
            temp=Q(1,(i-1)*item+position);
            Q(1,(i-1)*item+position)=Q(2,(i-1)*item+position);
            Q(2,(i-1)*item+position)=temp;
        end
    end   
    
end
result=bestprofit(1,gen);