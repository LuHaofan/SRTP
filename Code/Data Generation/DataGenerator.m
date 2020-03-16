%% Initialization
clear ; close all; clc

n_samples = 100;
Kp = 100*rand(n_samples,1);
Km = 100*rand(n_samples,1);
ak = 0;
f = rand(n_samples,1);
a1 = ceil(10*rand(n_samples,1));
a = [a1 a1 ceil(10*rand(n_samples,1))];

filename = 'test_file.csv';
fid = fopen(filename,'w');
fprintf(fid,['Kp,', 'Km,', 'f,','Volume,', 'Surface Area,','AvgR,','p,','K', '\n']);

% for i = (1:1:n_samples)
%     sphericity = EllipsoidSphericity(a(i,:));
%     result = misEllipsoid(Kp(i), Km(i), ak, f(i), a(i,:));
%     fprintf(fid,['%.4f,','%.4f,','%.4f,','%.4f,','%.4f,','%.4f','\n'],Kp(i),Km(i),f(i),real(sphericity), a(i,3)/a(i,1), real(result));
% end
% for j = (1:1:length(Km))
%     for i = (1:1:length(Kp))
%         %sphericity = EllipsoidSphericity([1 1 1]);
%         [K11, K22, K33] = generalNan(Kp(i), Km(j), 0,f, [0.3 0.3 0.4], 1);
%         fprintf(fid,['%.4f,','%.4f,','%.4f,','%.4f,','%.4f','\n'],Kp(i),Km(j), f, real(K11), real(K33));
%     end
% end
for i = (1:1:n_samples)
    [K, V, S] = misEllipsoid(Kp(i), Km(i), 0, f(i), a(i,:));
    p = a(i,3)/a(i,1);
    AvgR = (a(i,1)*a(i,2)*a(i,3))^(1/3);
    fprintf(fid,['%.4f,','%.4f,','%.4f,','%.4f,','%.4f,','%.4f,','%.4f,','%.4f','\n'],Kp(i),Km(i), f(i), V, S, AvgR, p, real(K));
end
fclose(fid);
