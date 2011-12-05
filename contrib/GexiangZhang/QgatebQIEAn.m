function  therta=QgatebQIEAn(fx,fb,aa,bb,angleaa,anglebb)
d1=angleaa*anglebb;
d2=aa*bb;
if (d1*d2==0)
    r=rand(1);
    if r<=0.5
        s=-1;
    else
        s=1;
    end
else
    angle_best=atan(anglebb/angleaa);
    angle_f=atan(bb/aa);    
    if ((d1>0)&(d2>0))                              
        if angle_best>=angle_f
            s=+1;
        else
            s=-1;
        end
    elseif ((d1<0)&(d2<0))
        if angle_best>=angle_f
            s=sign(aa*angleaa);
        else
            s=-sign(aa*angleaa);
        end
    elseif (d1>0)&(d2<0)
        s=sign(aa*angleaa);
    elseif (d1<0)&(d2>0)
        s=-sign(aa*angleaa);
    else
        r=rand(1);
        if r<=0.5
            s=-1;
        else
            s=1;
        end
    end
end
therta=s;

