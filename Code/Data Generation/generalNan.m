function [K11, K22, K33] = generalNan(Kp,Km,ak, f, a, cos2_theta)
p = a(3)/a(1);
L = zeros(3,1);
if p > 1
    L(1) = p*p/(2*(p*p-1))-p*acosh(p)/(2*(p*p-1)^(1.5));
    gamma = (2+1/p)*ak/a(1);
elseif p < 1
    L(1) = p*p/(2*(p*p-1))+p*acos(p)/(2*(1-p*p)^(1.5));
    gamma = (1+2*p)*ak/a(3);
else
    L(1) = 1/3;
    gamma =(1+2*p)*ak/a(3);
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

