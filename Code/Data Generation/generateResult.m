%% Initialization
clear ; close all; clc

Kp = [0.2, 0.5, 1, 2, 5, 10, 20];
Km = [0.2, 0.5, 1, 2, 5, 10, 20];
ak = 0;
f = 0.184725648;
a = [0.49 0.3 0.3];
filename = 'ellipsoid_nan.csv';
fid = fopen(filename,'w');
fprintf(fid,['Kp,', 'Km,','K11_Nan,', 'K33_Nan','\n']);
for j = (1:1:length(Km))
    for i = (1:1:length(Kp))
        [K11, K22, K33] = generalNan(Kp(i), Km(j), 0, f, a, 1);
        fprintf(fid,['%.4f,','%.4f,','%.4f,','%.4f,','\n'],Kp(i),Km(j), real(K11), real(K33));
    end
end
fclose(fid);