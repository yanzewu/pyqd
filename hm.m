function hm(nstep, dt)

function dY = evolve(t, Y)

x = Y(1);
v = Y(2);

dx = v;
dv = -100*x;

dY = [dx;dv];

end

[t, y] = ode113(@evolve, 0:dt:dt*nstep, [0; 10], odeset('RelTol', 1e-14, 'AbsTol', 1e-14));


plot(t, y(:,1).^2*0.5*100 + y(:,2).^2*0.5);


end

