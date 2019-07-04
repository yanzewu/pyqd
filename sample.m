function [omegaj, cj] = sample(eta, omegac, N)

omegamax=100*omegac;

%plot
omega=0:.1:omegamax;
%figure
%plot(omega,eta*omega*omegac./(omega.^2+omegac^2),'b')

j=1:N;
omegaj=(j/N).^2*omegamax;

JD=eta*omegaj*omegac./(omegaj.^2+omegac^2);
W=N/2./sqrt(omegaj*omegamax);

cj=sqrt(omegaj*2/pi.*JD./W);
%hold on
%plot(omegaj,cj,'.r')
%hold off

end