% Fsph

function [final_value]=rQIEA(maxNoE)

global initial_flag;

% initialization
Ps=20; % the size of population
L=30;   % the num of parameters
PsL=Ps*L;
minpara=-100*ones(1,L);
MINPARA=-100;
maxpara=100*ones(1,L);
MAXPARA=100;
range=(MAXPARA-MINPARA)*ones(1,L);
gen=0;
flag=1;
Q=zeros(2,PsL);
P=zeros(1,PsL);
MAXGEN=maxNoE/Ps;

% initialize Q(gen)
Q1sign=-1+2*rand(1,PsL);
Q1sign=sign(Q1sign);
Q2sign=-1+2*rand(1,PsL);
Q2sign=sign(Q2sign);
Q1rand=rand(1,PsL);
for qi=1:PsL
    Q(1,qi)=Q1sign(1,qi)*Q1rand(1,qi);
    Q(2,qi)=Q2sign(1,qi)*sqrt(1-Q(1,qi)^2);
end
clear Q1sign Q2sign Q1rand qi;

%  ***********************************
minfitness=999e100;
while flag>0
    %construct P(gen)
    rr=rand(1,PsL);
    for i=1:Ps
        for j=1:L
            if rr(1,(i-1)*L+j)<0.5
                para=Q(1,i)^2;
            else
                para=Q(2,i)^2;
            end
            P(1,(i-1)*L+j)=minpara(1,j)+(maxpara(1,j)-minpara(1,j))*para;
        end
    end

    gen=gen+1;
    midminfitness=999e100;
    sumfit=0;
    for i=1:Ps
        x=[];
        for xnum=1:L
            x=[x P(1,(i-1)*L+xnum)];
        end
        fitness=0;
        for fitnum=1:L
            fitness=fitness+x(1,fitnum)*x(1,fitnum); 
        end
        if fitness<midminfitness
            midminfitness=fitness;
            angle=Q(:,(i-1)*L+1:i*L);
            midx=x;
        end
        clear rr x;
    end

    if midminfitness<minfitness
        minfitness=midminfitness;
        anglemin=angle;
        minx=midx;
    end
    
    % change the range of parameters
    for ii=1:L
        range(1,ii)=range(1,ii)*0.983;
        minpara(1,ii)=minx(1,ii)-range(1,ii)/2;
        if minpara(1,ii)<MINPARA
            minpara(1,ii)=MINPARA;
        end
        maxpara(1,ii)=minx(1,ii)+range(1,ii)/2;
        if maxpara(1,ii)>MAXPARA
            maxpara(1,ii)=MAXPARA;
        end
    end
        
    % termination condition
    NoE=gen*Ps;
    if (NoE>maxNoE-1)|(minfitness==0)
        flag=-1;        
    end

   % Quantum gate operation
    for i=1:Ps
        for j=1:L
            aa=Q(1,(i-1)*L+j);
            bb=Q(2,(i-1)*L+j);
            angleaa=anglemin(1,j);
            anglebb=anglemin(2,j);
            therta=rotation(aa,bb,angleaa,anglebb);
            kvalue=pi/(100+mod(gen,100)); % 500 is good
            therta=therta*kvalue;
            u=[cos(therta) -sin(therta)
                sin(therta) cos(therta)];
            Q(1:2,(i-1)*L+j)=u*Q(1:2,(i-1)*L+j);
        end
    end

    % local migration
    %pc=0.05;
    pc=0.05; %0.05
    for i=1:Ps
        if rand(1)<pc
            posP1=1+floor(Ps*rand(1));
            posL1=1+floor(L*rand(1));
            posP2=1+floor(Ps*rand(1));
            posL2=1+floor(L*rand(1));
            posL3=min(posL1,posL2);
            posL4=max(posL1,posL2);
            temp1=Q(1,(posP1-1)*L+posL3:(posP1-1)*L+posL4);
            temp2=Q(2,(posP1-1)*L+posL3:(posP1-1)*L+posL4);
            Q(1,(posP1-1)*L+posL3:(posP1-1)*L+posL4)=Q(1,(posP2-1)*L+posL3:(posP2-1)*L+posL4);
            Q(2,(posP1-1)*L+posL3:(posP1-1)*L+posL4)=Q(2,(posP2-1)*L+posL3:(posP2-1)*L+posL4);
            Q(1,(posP2-1)*L+posL3:(posP2-1)*L+posL4)=temp1;
            Q(2,(posP2-1)*L+posL3:(posP2-1)*L+posL4)=temp2;
        end
    end    
end
final_value=minfitness;