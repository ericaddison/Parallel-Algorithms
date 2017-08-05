n = 1:100;

L = 25;

nf = 50;
x = zeros(nf,length(n));

for k = 1:2:2*nf
   x((k+1)/2,:) = k+(4/(pi*k))*sin(k*pi*n/L);
end

close all
subplot(121)
plot(n,x,'LineWidth',2)
axis off

for k = 1:nf
   x(k,:) = x(k,:)-(2*k-1);
end


subplot(122)
plot(n,sum(x),'LineWidth',2)
axis off
%axis([0, 100, -1.25 1.25])


set(gcf, 'PaperUnits', 'points');
set(gcf, 'PaperPosition', [100, 100, 500, 250]);
saveas(gcf,['./signalBuilder'],'epsc');