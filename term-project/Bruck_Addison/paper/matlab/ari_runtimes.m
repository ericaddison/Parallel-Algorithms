data = [1	0.696	0	0.015	0	0.91
2	0.463	0.001	0.03	0.002	1.069
4	0.247	0.004	0.044	0.005	7.066
8	0.323	0.014	0.028	0.008	0.541
16	0.495	0.033	0.043	0.014	0.277
32	0.594	0.076	0.079	0.027	0.325
64	0.548	0.139	0.098	0.055	0.408
128	1.693	0.486	0.176	0.099	0.345
256	1.039	0.889	0.385	0.256	0.404
512	3.12	1.212	0.585	0.405	0.638
1024	2.531	2.232	0.547	0.667	0.826
2048	2.849	3.324	0.709	1.005	1.353
4096	4.159	4.954	1.028	1.724	2.488
8192	7.77	9.796	2.176	3.633	5.055
16384	16.245	20.478	4.606	7.598	10.522
32768	34.587	43.362	9.535	16.2	21.615
65536	73.226	90.266	20.574	34.415	45.223
131072	154.714	188.272	42.458	72.778	94.405
262144	325.201	392.8	91.548	159.872	195.173
524288	691.775	820.841	200.258	342.848	410.232
1048576	1425.78	1709.27	428.686	724.379	842.62
2097152	3018.62	3657.53	877.969	1686.86	1761.6
4194304	6359.05	7523.82	2019.84	4250.59	3715.88
8388608	14412.1	16665	4314.14	7186.1	8933.46];


n = data(:,1);
scale = 1000;
cilk_rec_basecase_1 = data(:,2)/scale;
noncilk_rec = data(:,3)/scale;
cilk_iter = data(:,4)/scale;
noncilk_iter = data(:,5)/scale;
cilk_rec_basecase_128 = data(:,6)/scale;
close all

%% figure 1
plot(n,(noncilk_rec),'-s','MarkerSize',2,'LineWidth',2)
hold on
plot(n,(cilk_rec_basecase_1),'-sr','MarkerSize',2,'LineWidth',2)
plot(n,(cilk_rec_basecase_128),'-sg','MarkerSize',2,'LineWidth',2)
plot(n,(noncilk_iter),'-sk','MarkerSize',2,'LineWidth',2)
plot(n,(cilk_iter),'-sm','MarkerSize',2,'LineWidth',2)
l = legend('non-cilk recursive','cilk recursive basecase=1', 'cilk recursive basecase=128','non-cilk iterative', 'cilk iterative')
set(l,...
    'Position',[0.16064453125 0.660807291666667 0.4228515625 0.21484375]);

title('FFT: cilk vs CPP')
ylabel('Time (seconds)')
xlabel('Array Size n')
set(gca,'ygrid','on','GridLineStyle','-')
set(gcf, 'PaperUnits', 'points');
set(gcf, 'PaperPosition', [100, 100, 500, 250]);
saveas(gcf,['./fft_cilk_cpp'],'epsc');

%% figure 2
figure
plot(n,(cilk_rec_basecase_1),'-s','MarkerSize',2,'LineWidth',2)
hold on
plot(n,(cilk_rec_basecase_128),'-sr','MarkerSize',2,'LineWidth',2)
l = legend('cilk recursive basecase=1', 'cilk recursive basecase=128')
set(l,...
    'Position',[0.16064453125 0.660807291666667 0.4228515625 0.21484375]);

title('Recursive FFT in cilk: cilk\_max\_recombine 1 vs 128')
ylabel('Time (seconds)')
xlabel('Array Size n')
set(gca,'ygrid','on','GridLineStyle','-')
set(gcf, 'PaperUnits', 'points');
set(gcf, 'PaperPosition', [100, 100, 500, 250]);
saveas(gcf,['./cilk_max_recombine'],'epsc');

%% figure 3
figure
plot(n,(noncilk_rec),'-s','MarkerSize',2,'LineWidth',2)
hold on
plot(n,(noncilk_iter),'-sr','MarkerSize',2,'LineWidth',2)
l = legend('non-cilk recursive','non-cilk iterative')
set(l,...
    'Position',[0.16064453125 0.660807291666667 0.4228515625 0.21484375]);

title('Iterative vs Recursive non-cilk FFT')
ylabel('Time (seconds)')
xlabel('Array Size n')
set(gca,'ygrid','on','GridLineStyle','-')
set(gcf, 'PaperUnits', 'points');
set(gcf, 'PaperPosition', [100, 100, 500, 250]);
saveas(gcf,['./iterative_recursive_non_cilk'],'epsc');

%% figure 4
figure
plot(n,(noncilk_iter),'-s','MarkerSize',2,'LineWidth',2)
hold on
plot(n,(cilk_iter),'-sr','MarkerSize',2,'LineWidth',2)
l = legend('non-cilk iterative', 'cilk iterative')
set(l,...
    'Position',[0.16064453125 0.660807291666667 0.4228515625 0.21484375]);

title('Iterative cilk vs Iterative non-cilk FFT')
ylabel('Time (seconds)')
xlabel('Array Size n')
set(gca,'ygrid','on','GridLineStyle','-')
set(gcf, 'PaperUnits', 'points');
set(gcf, 'PaperPosition', [100, 100, 500, 250]);
saveas(gcf,['./iterative_fft'],'epsc');
