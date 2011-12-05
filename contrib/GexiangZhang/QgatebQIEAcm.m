function  therta=rotationbQIEAm(x,b,aa,bb,fx,fb)
if fx>=fb
    ther_chan=0;
    s=0;
else
   if ((x==0)&(b==1))
       ther_chan=0.01*pi;
       if aa*bb>0
            s=1;
        elseif aa*bb<0
            s=-1;
        else
            ss=rand(1);
            if ss>0.5
                s=1;
            else
                s=0;
            end
        end        
    elseif ((x==1)&(b==0))
        ther_chan=-0.01*pi;        
        if aa*bb>0
            s=1;
        elseif aa*bb<0
            s=-1;
        else
            ss=rand(1);
            if ss>0.5
                s=1;
            else
                s=0;
            end
        end
    else
        ther_chan=0;
        s=0;
    end
end
therta=ther_chan*s;
   
        
        