function [ f ] = rotation( aa,bb,angleaa,anglebb )
    xi_b = atan(anglebb / angleaa);
    xi_ij = atan(bb / aa);
    
    if xi_b > 0 && xi_ij > 0
        if xi_b >= xi_ij
            f = 1;
        else
            f = -1;
        end
    elseif xi_b > 0 && xi_ij <= 0
        f = sign(angleaa * aa);
    elseif xi_b <= 0 && xi_ij > 0
        f = -sign(angleaa * aa);
    elseif xi_b <= 0 && xi_ij <= 0
        if xi_b >= xi_ij
            f = 1;
        else
            f = -1;
        end
    elseif xi_b == 0 || xi_ij == 0 || abs(xi_b - pi/2) < 0.001 || abs(xi_b - pi/2) < 0.001
        f = sign(rand() - .5);
    else
        disp('err')
    end
end

