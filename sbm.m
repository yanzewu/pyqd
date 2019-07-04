function sbm(file_prefix, step, dt)

% Default variables

    if ~exist('dt', 'var')
        dt = 1e-2;
    end
    
    file_model = fopen(strcat(file_prefix, '-model.txt'), 'r');
    fgetl(file_model);  % H=
    lineH = fgetl(file_model);  fgetl(file_model); fgetl(file_model);  % C1=
    lineC1 = fgetl(file_model); fgetl(file_model);
    lineC2 = fgetl(file_model);
    
    file_x0 = fopen(strcat(file_prefix, '-x.txt'), 'r');
    linex0 = fgetl(file_x0);
    
    file_k0 = fopen(strcat(file_prefix, '-p.txt'), 'r');
    linek0 = fgetl(file_k0);
    
    fclose('all');
    
    lineH = strrep(lineH, ',', ' ');
    lineH = strrep(lineH, 'Q', '');
    
    A = str2num(lineH);
    dE = A(1);
    cnorm = A(2);
    V = A(3);
    
    %C1 = str2num(lineC1) * cnorm;
    %C2 = str2num(lineC2);
    
    eta = 0.5;
    cutoff = 0.25;
    T = 2;
    
    [w, C1] = sample(eta, cutoff, 500);
    
    %w = rand([1,1000])*(cutoff*8-0.01)+0.01;
    %J = eta*cutoff*w./(w.^2+cutoff.^2);
    
    N = numel(w);
    %g = N/(cutoff*8-0.01);
    %C1 = sqrt(2*w.*J./pi/g);
    C2 = 0.5*w.^2;
    
    %x0 = str2num(strrep(linex0, ',', ' '));
    %k0 = str2num(strrep(linek0, ',', ' '));
    
    %x0 = randn([1,N])*sqrt(T)./w;
    %k0 = randn([1,N])*sqrt(T);
    
    x0 = randn([1,N])./sqrt(2*w.*tanh(w/2/T));
    k0 = randn([1,N]).*sqrt(w./2./tanh(w/2/T));

    m_evolve = @(~, Y) evolve(Y, V, C1.', C2.'*2);
    
    [t, y] = ode113(m_evolve, 0:dt:dt*step, [1.0,0.0,0.0,x0,k0].');
   
    plot(t, y(:,1));
    
end

function dY = evolve(Y, V, C, w2)

    rho11m22 = Y(1);
    rho12m21 = Y(2);
    rho12p21 = Y(3);
    
    N = numel(C);
    
    x = Y(4:N+3);
    v = Y(N+4:N*2+3);

    R = dot(C, x);
    
    dY = zeros(size(Y));
    dY(1) = -2i*V*rho12m21;
    dY(2) = -2i*(V*rho11m22 - R*rho12p21);
    dY(3) = 2i*(rho12m21*R);
    dY(4:N+3) = v;
    dY(N+4:N*2+3) = -rho11m22.*C - w2.*x;
    
end