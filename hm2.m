function hm2(nstep, dt)

w = 1;
V = 1;
C1 = 1;
nlevel = 100;
x0 = 1;

% Harmonic oscillator
H0 = diag(((0:nlevel-1) + 0.5) * w);

% X operator
X = zeros(nlevel);
for j = 1:nlevel - 1
    X(j,j+1) = sqrt(j)./sqrt(2*w);
    X(j+1,j) = sqrt(j)./sqrt(2*w);
end

H = [H0+C1*X, eye(nlevel)*V; eye(nlevel)*V, H0-C1*X];

% set a coherent state
alpha = x0 * sqrt(w/2);

c = zeros(nlevel*2,1);
%c(1) = 1;
for j = 1:nlevel
    c(j) = exp(-0.5*alpha^2)*alpha^j/sqrt(factorial(j));
end
c = c / sqrt(sum(c.^2));


U = expm(-1i*H*dt);

P1 = zeros(nstep,1);
P2 = zeros(nstep,1);

for j = 1:nstep
    
    c = U*c;
    P1(j) = sum(abs(c(1:nlevel)).^2);
    P2(j) = sum(abs(c(nlevel+1:nlevel*2)).^2);
end

t = (1:nstep)*dt;

plot(t, P1 - P2, 'b-');

end