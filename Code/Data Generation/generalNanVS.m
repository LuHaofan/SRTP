function [K11, K22, K33, V, S] = generalNanVS(Kp,Km,ak, f, a, cos2_theta)
V =  4/3*pi*a(1)*a(2)*a(3);
p = a(3)/a(1);
L = zeros(3,1);
if p > 1
    L(1) = p*p/(2*(p*p-1))-p*acosh(p)/(2*(p*p-1)^(1.5));
    gamma = (2+1/p)*ak/a(1);
    e2 = 1-a(1)^2/a(3)^2;
    S = 2*pi*a(1)^2*(1+(a(3)*asin(e2^0.5))/(a(1)*e2^0.5));
elseif p < 1
    L(1) = p*p/(2*(p*p-1))+p*acos(p)/(2*(1-p*p)^(1.5));
    gamma = (1+2*p)*ak/a(3);
    e2 = 1-a(3)^2/a(1)^2;
    S = 2*pi*a(1)^2*(1+(a(3)^2*atanh(e2^0.5))/(e2^0.5*a(1)^2));
else
    L(1) = 1/3;
    gamma =(1+2*p)*ak/a(3);
    S = 4*pi*a(1)^2;
end
L(2) = L(1);
L(3) = 1-2*L(1);
Kc = Kp./(1+gamma.*L.*Kp./Km);
beta = (Kc-Km)./(Km+L.*(Kc-Km));
%K = Km*(3+f*(2*beta(1)*(1-L(1))+beta(3)*(1-L(3))))/(3-f*(2*beta(1)*L(1)+beta(3)*L(3)));
K11_num = Km*(2+f*beta(1)*(1-L(1))*(1+cos2_theta)+beta(3)*(1-L(3))*(1-cos2_theta));
K11_den = 2-f*(beta(1)*L(1)*(1+cos2_theta)+beta(3)*L(3)*(1-cos2_theta));
K11 = K11_num/K11_den;
K22 = K11;
K33_num = Km*(1+f*(beta(1)*(1-L(1))*(1-cos2_theta)+beta(3)*(1-L(3))*cos2_theta));
K33_den = 1-f*(beta(1)*L(1)*(1-cos2_theta)+beta(3)*L(3)*cos2_theta);
K33 = K33_num/K33_den;
end

