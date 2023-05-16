close; clear; clc;

M = 1000;
N = 10000;
% Something is wrong for M ~= N  
c = N/M;
X = randn(M,N);

X1 = X;

X1(:,3) = X(:,3)*1.8;
X1(:,5) = X(:,5)*1.8;
X1(:,7) = X(:,7)*1.8;
X1(:,9) = X(:,7)*1.8;


H = (1/N)*(X*X');
H1 = (1/N)*(X1*X1');


a = (1 - (1/sqrt(c)))^2;
b = (1 + (1/sqrt(c)))^2;
xhi = linspace(a,b);

p = 1 ./(2 * pi * xhi) .* sqrt((b-xhi).* (xhi-a));
figure()


eigPlot = sort(eig(H),'descend');
eigPlot1 = sort(eig(H1),'descend');

title("Marcenko-Pastur Distribution")
figure()
plot(xhi,p)
hold on
histogram(eigPlot, 100)
hold off
figure()
plot(xhi,p)
hold on

histogram(eigPlot1,100)
legend({'N(0,1)','w/ Signal'},'Location','southwest')
hold off