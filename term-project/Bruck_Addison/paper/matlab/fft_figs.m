close all

n = 1:1000;
f0 = 0.25;

y = sin(2*pi*0.001*n) + 0.5*cos(2*pi*0.2*n) + 0.2*sin(2*pi*0.4*n + pi/4);

Y = abs(fft(y))*2/length(n);

figure('position',[100,100,750,250])
subplot(121)
plot(y)
title('x_1[n]')
xlabel('n')

subplot(122)
plot(linspace(0,1,length(Y)),Y)
axis([-0.05 1.05 0 1.2])
title('FFT of x_1[n]')
xlabel('f')

set(gcf, 'PaperUnits', 'points');
set(gcf, 'PaperPosition', [100, 100, 750, 250]);
saveas(gcf,['./sins'],'epsc');

x = rand(1,length(n)) - 0.5;


x = x + 0.2*sin(2*pi*f0*n);

X = abs(fft(x))*2/length(n);

figure('position',[100,100,750,250])
subplot(121)
plot(x)
title('x_2[n]')
xlabel('n')

subplot(122)
plot(linspace(0,1,length(X)),X)
title('FFT of x_2[n]')
xlabel('f')
set(gcf, 'PaperUnits', 'points');
set(gcf, 'PaperPosition', [100, 100, 750, 250]);
saveas(gcf,['./random'],'epsc');


