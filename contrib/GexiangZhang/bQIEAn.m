% knapsack problem

function [result]=bQIEAn(item,MAXgen)

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

maxfit=-9999;
maxab=zeros(2,item);
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
    % evaluation
    eachprofit(1,i)=sum(profit.*P(1,(i-1)*item+1:i*item));
    if eachprofit(1,i)>maxfit
        maxfit=eachprofit(1,i);      
        maxab=Q(:,(i-1)*item+1:i*item);
        maxbit=P(1,(i-1)*item+1:i*item);
    end
end

bestprofit(1,gen)=maxfit;
currentmaxab=maxab;
currentmaxbit=maxbit;
Bprofit=eachprofit;
Bbit=P;

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
    maxab=zeros(2,item);
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
        % evaluation
        eachprofit(1,i)=sum(profit.*P(1,(i-1)*item+1:i*item));
        if eachprofit(1,i)>maxfit
            maxfit=eachprofit(1,i);
            maxab=Q(:,(i-1)*item+1:i*item);
            maxbit=P(1,(i-1)*item+1:i*item);
        end
    end
    
    if (maxfit>bestprofit(1,gen-1))
        bestprofit(1,gen)=maxfit;
        currentmaxab=maxab;
        currentmaxbit=maxbit;
    else
        bestprofit(1,gen)=bestprofit(1,gen-1);
    end
    
    % update B
    maxbp=[Bprofit eachprofit];
    maxBbit=[Bbit P];
    for i=1:length(Bprofit)
        [maxvalue maxvaluepos]=max(maxbp);
        Bprofit(1,i)=maxvalue;
        Bbit(1,(i-1)*item+1:i*item)=maxBbit(1,(maxvaluepos-1)*item+1:maxvaluepos*item);
        maxbp(1,maxvaluepos)=-9.9e+100;
    end
    
    %update Q(gen)
    for k=1:Ps
        fx=eachprofit(1,k);
        for j=1:item
            aa=Q(1,(i-1)*item+j);
            bb=Q(2,(i-1)*item+j);
            angleaa=currentmaxab(1,j);
            anglebb=currentmaxab(2,j);          
            fb=bestprofit(1,gen);
            therta=QgatebQIEAn(fx,fb,aa,bb,angleaa,anglebb);
            kvalue=0.5*pi*exp(-5*gen/MAXgen);
            therta=therta*kvalue;
            u=[cos(therta) -sin(therta)
                sin(therta) cos(therta)];
            Q(1:2,(k-1)*item+j)=u*Q(1:2,(k-1)*item+j);
        end
    end
    
    
    % local migration
    for i=1:Ps/2
        if Bprofit(1,2*i-1)>Bprofit(1,2*i)
            Bprofit(1,2*i)=Bprofit(1,2*i-1);
            Bbit(1,(2*i-1)*item+1:(2*i)*item)=Bbit(1,(2*i-2)*item+1:(2*i-1)*item);
        else
            Bprofit(1,2*i-1)=Bprofit(1,2*i);
            Bbit(1,(2*i-2)*item+1:(2*i-1)*item)=Bbit(1,(2*i-1)*item+1:(2*i)*item);
        end
    end
        
    % global migration 
    if (mod(gen,100)==0)
        for i=1:Ps
            Bprofit(1,i)=bestprofit(1,gen);
            Bbit(1,(i-1)*item+1:i*item)=currentmaxbit;
        end
    end
    
    % catastrophe
    catgen=45;
    if gen>catgen
        if var(bestprofit(1,gen-catgen+1:gen))<1e-6
           % a new random Q
           newq=zeros(2,item); 
           newP=zeros(1,PsL); 
           neweachprofit=zeros(1,Ps);
           for j=1:item
               newq(1,j)=sign(-1+2*rand(1))*rand(1);
               newq(2,j)=sign(-1+2*rand(1))*sqrt(1-q(1,j)^2);
           end
           newQ=newq;
           for kki=1:Ps-1
               for j=1:item
                   newq(1,j)=sign(-1+2*rand(1))*rand(1);
                   newq(2,j)=sign(-1+2*rand(1))*sqrt(1-newq(1,j)^2);
               end
               newQ=[newQ newq];
           end
           
           % construct newP
           for i=1:PsL
               rr=rand(1);
               if rr<newQ(2,i)^2
                   bit=1;
               else
                   bit=0;
               end
               newP(1,i)=bit;
           end
           
           newmaxfit=-9999;
           newmaxab=zeros(2,item);
           for i=1:Ps
               % repair
               if sum(weight.*newP(1,(i-1)*item+1:i*item))>capacity
                   knapscakoverfilled=0;
                   if sum(weight.*newP(1,(i-1)*item+1:i*item))>capacity
                       knapscakoverfilled=1;
                   end
                   while (knapscakoverfilled>0.5)
                       position=1+floor(rand(1)*item);
                       newP(1,(i-1)*item+position)=0;
                       if sum(weight.*newP(1,(i-1)*item+1:i*item))<=capacity
                           knapscakoverfilled=0;
                       end
                   end
                   while (knapscakoverfilled<0.5)
                       position=1+floor(rand(1)*item);
                       newP(1,(i-1)*item+position)=1;
                       if sum(weight.*newP(1,(i-1)*item+1:i*item))>capacity
                           knapscakoverfilled=1;
                       end
                   end
                   newP(1,(i-1)*item+position)=0;
               end
               neweachprofit(1,i)=sum(profit.*newP(1,(i-1)*item+1:i*item));
               if neweachprofit(1,i)>newmaxfit
                   newmaxfit=neweachprofit(1,i);
                   newmaxab=newQ(:,(i-1)*item+1:i*item);
               end
           end
           bestprofit(1,gen)=newmaxfit;
           currentmaxab=newmaxab;
        end
    end            
    
end
result=max(bestprofit);