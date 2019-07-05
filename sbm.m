function sbm(step, dt)

% Default variables

    if ~exist('dt', 'var')
        dt = 1e-2;
    end
    

    eta = 0.5;
    cutoff = 1;
    T = 0.1;
    V = 1;
    
    [w, C1] = sample(eta, cutoff, 100);
    w = 1;
    C1 = 1;
    
    N = numel(w);
    
    x0 = randn([1,N])./sqrt(2*w.*tanh(w/2/T));
    k0 = randn([1,N]).*sqrt(w./2./tanh(w/2/T));
    x0 = 1;
    k0 = 0;

    m_evolve = @(~, Y) evolve(Y, V, C1.', w.'.^2);
    
    options = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);
    [t, y] = ode45(m_evolve, 0:dt:dt*step, [1.0,0.0,0.0,x0,k0].', options);
   
    plot(t, real(y(:,1)));
    xlabel('t')
    ylabel('\rho_{11}-\rho_{22}')
    
    return
    
    % Energy
    E = zeros(numel(t),1);
    for j = 1:numel(t)
        rho11m22 = y(j,1);
        rho12p21 = y(j,3);
        x = y(j, 4:N+3);
        v = y(j, N+4:N*2+3);
        E(j) = rho11m22*dot(C1,x) + rho12p21*V + 0.5*dot(w.^2,x.^2) + 0.5*dot(v,v);
    end
    
    figure(2);
    plot(t, E);
    
    % Position
    figure(3);
    hold on
    for j = 1:N
        plot(t, y(:, 3+j));
    end
    
    % Momentum
    figure(4);
    hold on
    for j = 1:N
        plot(t, y(:, 3+N+j));
    end
end

function dY = evolve(Y, V, C, w2)

    rho11m22 = Y(1);
    rho12m21 = Y(2);
    rho12p21 = Y(3);
    
    N = numel(C);
    
    x = Y(4:N+3);
    v = Y(N+4:N*2+3);

    y = dot(C, x);
    
    dY = zeros(size(Y));
    dY(1) = 2i*V*rho12m21;
    dY(2) = 2i*(V*rho11m22 - y*rho12p21);
    dY(3) = -2i*(rho12m21*y);
    dY(4:N+3) = v;
    dY(N+4:N*2+3) = -rho11m22.*C - w2.*x;
    
end