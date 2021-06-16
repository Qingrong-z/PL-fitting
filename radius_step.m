function [radius_init,step]=radius_step(E_min,E_max)
x = E_max - E_min;
y = 0.02;
step = fix(x/y) + 5;
if rem(x,y)<0.01
    radius_init = 0.01;
else
    radius_init = 0.015;
end

