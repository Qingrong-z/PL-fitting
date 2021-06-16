clearvars
close all

global E;

h=6.626e-34; %Planck's constant
h_ = h/(2*pi); %Reduced Planck's constant
c=2.998e8; %Speed of light
q=1.602e-19; % Elementary charge
k = 1.381e-23; %Boltzmann's constant
E_LO = 36.1e-3; %Energy of the LO phonon in GaAs (eV)
T_L = 296; % Room temperature

m_e=0.067; %Effective mass of electrons, relative to the free electron mass
m_h=0.47; %Effective mass of holes, relative to the free electron mass
D=3; %Dimensionality of bulk


%% Choose the sample
%200nm
% day='2019_07_05';
% sample='1873_1_4';
% detector='vis';
% laser=532;

%100nm
% day='2019_12_10';
% sample='1927_2_1';
% detector='vis';
% laser=532;

%50nm
% day='2019_12_10';
% sample='1926_2_5';
% detector='vis';
% laser=532;

%20nm
day='2019_07_05';
sample='1872_1_4';
detector='vis';
laser=532;

[data, power, spot_surface, Int, E]=load_data_PL(day,sample,detector,laser);

%% Define parameters

switch sample
    case '1872_1_4' %GaAs 20nm Au mirror
        thickness=20e-7;
        barrier_thickness=85;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1872_wav_300_1000_habs_20_h_both_AlGaAs_84_Au_mirror.mat');
        A_GaAs_E_min1=A_GaAs_E;
        A_total_E_min1=A_total_E;
        Abs_layer_E_min1=Abs_layer_E;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1872_wav_300_1000_habs_20_h_both_AlGaAs_86_Au_mirror.mat');
        A_GaAs_E_plus1=A_GaAs_E;
        A_total_E_plus1=A_total_E;
        Abs_layer_E_plus1=Abs_layer_E;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1872_wav_300_1000_habs_20_h_both_AlGaAs_85_Au_mirror.mat');
        
    case '1873_1_4' %GaAs 200nm Au mirror
        thickness=200e-7;
        barrier_thickness=93;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1873_wav_300_1000_habs_200_h_both_AlGaAs_92_Au_mirror.mat');
        A_GaAs_E_min1=A_GaAs_E;
        A_total_E_min1=A_total_E;
        Abs_layer_E_min1=Abs_layer_E;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1873_wav_300_1000_habs_200_h_both_AlGaAs_94_Au_mirror.mat');
        A_GaAs_E_plus1=A_GaAs_E;
        A_total_E_plus1=A_total_E;
        Abs_layer_E_plus1=Abs_layer_E;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1873_wav_300_1000_habs_200_h_both_AlGaAs_93_Au_mirror.mat');
        %find the optimal, uncertainty of the absorpiton
     
    case '1926_2_5' %GaAs 50nm Au mirror
        thickness=50e-7;
        barrier_thickness=82;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1926_wav_300_1000_habs_50_h_both_AlGaAs_81_Au_mirror')
        A_GaAs_E_min1=A_GaAs_E;
        A_total_E_min1=A_total_E;
        Abs_layer_E_min1=Abs_layer_E;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1926_wav_300_1000_habs_50_h_both_AlGaAs_83_Au_mirror')
        A_GaAs_E_plus1=A_GaAs_E;
        A_total_E_plus1=A_total_E;
        Abs_layer_E_plus1=Abs_layer_E;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1926_wav_300_1000_habs_50_h_both_AlGaAs_82_Au_mirror')
    
    case '1927_2_1' %GaAs 100nm Au mirror
        thickness=100e-7;
        barrier_thickness=80;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1927_wav_300_1000_habs_100_h_both_AlGaAs_79_Au_mirror')
        A_GaAs_E_min1=A_GaAs_E;
        A_total_E_min1=A_total_E;
        Abs_layer_E_min1=Abs_layer_E;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1927_wav_300_1000_habs_100_h_both_AlGaAs_81_Au_mirror')
        A_GaAs_E_plus1=A_GaAs_E;
        A_total_E_plus1=A_total_E;
        Abs_layer_E_plus1=Abs_layer_E;
        load('absorption\Abs_TMM_AlGaAs_GaAs_HCSC_ELO_1927_wav_300_1000_habs_100_h_both_AlGaAs_80_Au_mirror')  
end

%
switch day
    case '2019_07_05'
        switch sample
            
            case '1872_1_4' %GaAs 20nm Au mirror
                
                %removing points on which the model does not apply (lattice heating...)
                points_removed=[1 2 3 4];
                Int(points_removed)=[];
                power(points_removed)=[]; %power is in W.cm^-2
                
                E_min = 1.47;
                E_max = 1.52;
                E_ratio = 1.51;
                E_plot=[1.35 1.6];
                Eg = 1.424; %Band gap (eV)
                
            case '1873_1_4' %GaAs 200nm Au mirror
                
                %removing points on which the model does not apply (lattice heating...)
                points_removed=[1 2 12];
                Int(points_removed)=[];
                power(points_removed)=[]; %power is in W.cm^-2
                
                E_min = 1.45;
                E_max = 1.56;
                E_ratio = 1.51;
                E_plot=[1.35 1.6];
                Eg = 1.424; %Band gap (eV)             
        end
       
    case '2019_12_10'
        switch sample
            case '1927_2_1'
                
                %removing points on which the model does not apply (lattice heating...)
                points_removed=[7 8 9];
                Int(points_removed)=[];
                power(points_removed)=[]; %power is in W.cm^-2
                
                E_min = 1.48;
                E_max = 1.55;
                E_plot=[1.35 1.6];
                Eg = 1.424; %Band gap (eV)
                
            case '1926_2_5'
                
                points_removed=[6 7];
                Int(points_removed)=[];
                power(points_removed)=[]; %power is in W.cm^-2
                
                E_min = 1.47;
                E_max = 1.57;
                E_plot=[1.35 1.6];
                Eg = 1.424; %Band gap (eV)                      
        end
        
end

colors=lines(length(Int));
P_inc=power/(1000*spot_surface); % incident power (W.cm^-2)
E_laser=h*c/(q*laser*1e-9); %Energy of the laser (eV)

%% Consider different interval
E_min_init = E_min;
E_max_init = E_max;
Inv_diff = 0.01; % Fit the interval from -0.01 to +0.01
E_min = E_min_init-0.03;
E_max = E_max_init-0.03;
for i = [1:7]
    E_min = E_min + Inv_diff;
    E_max = E_max + Inv_diff;
%% Calculate the ratios with band filling
x_init=[T_L, 1.2, 1.2];
x_min=[T_L-10, 0.8, 0.8];
x_max=[500, Eg, Eg];

%% Initialize to find the range of mu_1 values

ratio_fit_init=@(x) mu_T_ratio(x(1), x(2), x(3),E(Einv(E_min):Einv(E_max)),T_L,m_e,m_h,Eg,D);

for i1 = 1:length(Int)
    ratio{i1}= real(Int{i1}./Int{1});
    ratio_opt_init{i1}=@(x) abs(1-ratio_fit_init(x)./ratio{i1}(Einv(E_min):Einv(E_max)));
    [x_sol_init{i1},res_sol_init(i1)]=lsqnonlin(ratio_opt_init{i1},x_init,x_min,x_max);
    T_init(i1)=x_sol_init{i1}(1);
    mu_init(i1)=x_sol_init{i1}(2);
    mu_ref_init(i1)=x_sol_init{i1}(3);
end

T_init(1)=T_L;
mu_init(1)=NaN;
mu_ref_init(1)=NaN;
%% Find the optimal mu_1

mu_ref_vec=linspace(min(mu_ref_init),max(mu_ref_init));

for i2=1:length(mu_ref_vec)
    for i1=2:length(Int)
        ratio_fit{i2}=@(x) mu_T_ratio(x(1), x(2), mu_ref_vec(i2),E(Einv(E_min):Einv(E_max)),T_L,m_e,m_h,Eg,D);
        ratio_opt{i1,i2}=@(x) 1-abs(ratio_fit{i2}(x)./ratio{i1}(Einv(E_min):Einv(E_max)));
        [x_sol{i1,i2},res_sol(i1,i2)]=lsqnonlin(ratio_opt{i1,i2},x_init,x_min,x_max);
    end
end

res_sol_sum=(sum(res_sol,1));
[min_res,idx_res]=min(res_sol_sum);

mu_ref(i)=mu_ref_vec(idx_res);

for i1=2:length(Int)
    T(i,i1)=x_sol{i1,idx_res}(1);
    mu(i,i1)=x_sol{i1,idx_res}(2);
end
T(i,1)=T_L;
mu(i,1)=mu_ref(i);


%% Log ratio fit
for i3 = 2:length(Int)
    y = log(ratio{1,i3}(Einv(E_min):Einv(E_max)));
    x = E(Einv(E_min):Einv(E_max));
    modelfun = @(a,x) log(dmu_T_ratio(a(1), a(2), a(3),x,T_L,m_e,m_h,Eg,D));
    options = statset('FunValCheck','off','DerivStep',10^-13,'robust','on');
    beta0_log_init{i3} = [T(i,i3);mu(i,i3)-mu_ref(i);mu_ref(i)];
%     beta0_dmu_init{i3} = [T(i);mu(i)-mu_ref_init(i);mu_ref_init(i)];
    [beta_log_init{i3},r,J,cov] = nlinfit(x,y,modelfun,beta0_log_init{i3},options);
    T_log_init(i3) = beta_log_init{i3}(1,1);
    delta_mu_log_init(i3) = beta_log_init{i3}(2,1);
    mu_log_init(i3) = beta_log_init{i3}(2,1)+beta_log_init{i3}(3,1);
    mu_ref_log_init(i3) = beta_log_init{i3}(3,1);
    ci_log_init{i3} = nlparci(beta_log_init{i3},r,'covar',cov);
    cj_log_init{i3} = nlparci(beta_log_init{i3},r,'Jacobian',J);
    mu_ref_log_init_fiterror(i3) = (ci_log_init{i3}(3,2)-ci_log_init{i3}(3,1))/2;
end
mu_ref_log_init(1) = NaN;
T_log_init(1)=T_L;
%  delta_mu_init(1)=NaN;
%  mu_nfit_dmu_init(1)=NaN;
%  mu_ref_nfit_dmu_init(1)=NaN;

%% mu_ref interval error
Inv_diff = 0.01; % Fit the interval from -0.01 to +0.01
E_min_muref = E_min-0.02;
E_max_muref = E_max-0.02;
for j = 1:3
    E_min_muref = E_min_muref + Inv_diff;
    E_max_muref = E_max_muref + Inv_diff;
    for i4 = 2:length(Int)
    y = log(ratio{1,i4}(Einv(E_min_muref):Einv(E_max_muref)));
    x = E(Einv(E_min_muref):Einv(E_max_muref));
    modelfun = @(a,x) log(dmu_T_ratio(a(1), a(2), a(3),x,T_L,m_e,m_h,Eg,D));
    options = statset('FunValCheck','off','DerivStep',10^-12,'robust','on');
    beta0_log_inv{i4} = [T_log_init(i4);delta_mu_log_init(i4);mu_ref_log_init(i4)]; % Use the obtained logarithm results
    [beta_log_inv{i4},r,J,cov] = nlinfit(x,y,modelfun,beta0_log_inv{i4},options);
    mu_ref_log_inv(j,i4) = beta_log_inv{i4}(3,1);
    ci_log_inv{j,i4} = nlparci(beta_log_inv{i4},r,'covar',cov);
    cj_log_inv{j,i4} = nlparci(beta_log_inv{i4},r,'Jacobian',J);
    end
    mu_ref_log_inv(j,1) = NaN;
end
    

for i5=2:length(Int)
    mu_ref_log_raw_init(i5) = mu_ref_log_inv(2,i5);
    mu_ref_log_plus_init(i5) = mu_ref_log_inv(3,i5);
    mu_ref_log_min_init(i5) = mu_ref_log_inv(1,i5);
    mu_ref_interval_sigma(i5) = (max(abs(mu_ref_log_plus_init(i5)-mu_ref_log_raw_init(i5)),abs(mu_ref_log_raw_init(i5)-mu_ref_log_min_init(i5))))/2;
    mu_ref_log_raw_sigma(i5) = mu_ref_log_init_fiterror(i5)/2;
    mu_ref_log_sigma(i5) = mu_ref_interval_sigma(i5) + mu_ref_log_raw_sigma(i5);
    mu_ref_top(i5) = mu_ref_log_init(i5)./(mu_ref_log_sigma(i5).^2);
    sigma_inverse(i5) = 1/mu_ref_log_sigma(i5);
end
 
 mu_ref_log(i) = sum(mu_ref_top)./sum((sigma_inverse.^2));%Reverse-variance weighting method
 var_mu_ref_log(i) = 2.*sqrt(1./sum((sigma_inverse.^2)));% 95% confidence interval

 
T_log_init(1)=T_L;


%% Refit with obtained optimal mu_ref
for i6 = 1:length(Int)
    y = log(ratio{1,i6}(Einv(E_min):Einv(E_max)));
    x = E(Einv(E_min):Einv(E_max));
    modelfun = @(a,x) log(dmu_T_ratio(a(1), a(2), mu_ref_log(i),x,T_L,m_e,m_h,Eg,D));
    options = statset('FunValCheck','off','DerivStep',10^-12,'robust','on');
    beta0_log{i6} = [T_log_init(i6);delta_mu_log_init(i6)];
    [beta_log{i6},r,J,cov] = nlinfit(x,y,modelfun,beta0_log{i6},options);
    T_log(i,i6) = beta_log{i6}(1,1);
    delta_mu_log(i,i6) = beta_log{i6}(2,1);
    mu_nfit_log(i,i6) = beta_log{i6}(2,1)+mu_ref_log(i);
    ci_log{i,i6} = nlparci(beta_log{i6},r,'covar',cov);
    cj_log{i,i6} = nlparci(beta_log{i6},r,'Jacobian',J);
    T_error_log(i,i6) = (ci_log{i,i6}(1,2)-ci_log{i,i6}(1,1))/2;
    delta_mu_error_log(i,i6) = (ci_log{i,i6}(2,2)-ci_log{i,i6}(2,1))/2;
end


end

for i=1:length(Int)
    T_log_raw(i) = T_log(2,i);
    T_plus_log(i) = T_log(3,i);
    T_min_log(i) = T_log(1,i);
    delta_T_interval_log(i) = max(abs(T_plus_log(i)-T_log_raw(i)),abs(T_log_raw(i)-T_min_log(i)));
    delta_T_init(i) = T_error_log(2,i);
    delta_T_log(i) = delta_T_interval_log(i)+delta_T_init(i);
    
    dmu_log(i) = delta_mu_log(2,i);
    dmu_plus_log(i) = delta_mu_log(3,i);
    dmu_min_log(i) = delta_mu_log(1,i);
    delta_dmu_interval_log(i) = max(abs(dmu_plus_log(i)-dmu_log(i)),abs(dmu_log(i)-dmu_min_log(i)));
    delta_mu_log_init(i) = delta_mu_error_log(2,i);
    delta_dmu_log(i) = delta_dmu_interval_log(i)+delta_mu_log_init(i);
end


%% Absorbed power

A_barrier_E=Abs_layer_E(1,:)+Abs_layer_E(3,:); % absorption in the barriers
A_laser_GaAs=A_GaAs_E(round(interp1(E_A, (1:length(E_A)),E_laser))); % absorption in GaAs at laser wavelength
A_laser_GaAs_min1=A_GaAs_E_min1(round(interp1(E_A, (1:length(E_A)),E_laser))); % absorption in GaAs at laser wavelength
A_laser_GaAs_plus1=A_GaAs_E_plus1(round(interp1(E_A, (1:length(E_A)),E_laser))); % absorption in GaAs at laser wavelength

A_laser_barriers=A_barrier_E(round(interp1(E_A, (1:length(E_A)),E_laser))); % Absorption in barriers at laser wavelength
Barriers_to_GaAs=0.1; % Estimated fraction of carriers generated in the barriers that recombine in the absorber
A_laser=A_laser_GaAs+Barriers_to_GaAs*A_laser_barriers; % Equivalent absorptivity in the absorber
P_abs=A_laser*P_inc; %Equivalent absorbed power in the absorber
P_gen=(P_abs/thickness)*(1-Eg/E_laser);
%% Plots


% figure
% title('log ratio')
% semilogx(P_abs, mu_ref_log_init(2,:),'LineWidth',3)
% hold on
% errorbar(P_abs,mu_ref_log_init(2,:),mu_ref_log_raw_init_fiterror,mu_ref_log_raw_init_fiterror,[],[], '+', 'markerSize',10,'color', colors(2,:),'LineWidth', 3);
% ylabel('$\mu_{ref}$ (eV)','Interpreter','Latex')
% xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
% box on
% xlim([0 P_abs(end)])
% % ylim([0 inf])
% set(gca,'Fontsize',16)
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gcf,'color','w')
% box on

% Linear fit
% p_dmu1 = polyfit(T_log_raw-T_L,P_abs,1); %Temperature fit
% Q_dmu_nfit = p_dmu1(1);
% deltaT0_dmu_nift = p_dmu1(2);
% deltaT_dmu_nfitdata = deltaT0_dmu_nift + Q_dmu_nfit*(T_log_raw-T_L);
% 
% p_dmu2 = polyfit(log(P_abs),dmu_log,1); %mu fit
% mu_slope_dmu_nfit = p_dmu2(1);
% mu_ref_dmu_nfit0 = p_dmu2(2);
% mu_dmu_nfitdata = mu_ref_dmu_nfit0+mu_slope_dmu_nfit*log(P_abs);




% Plots
% figure
% % title('f(\Delta\mu)')
% hold on
% % scatter(P_abs, T_dmu-T_L,100,'x','MarkerEdgeColor', colors(1,:),'LineWidth',3);
% errorbar(P_abs,T_log_raw-T_L,delta_T_log,delta_T_log,[],[], '+', 'markerSize',10,'color', colors(1,:),'LineWidth', 3);
% plot(deltaT_dmu_nfitdata,T_log_raw-T_L,'--','color', colors(2,:),'LineWidth', 2);
% ylabel('$T-T_L$ (K)','Interpreter','Latex')
% xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
% box on
% xlim([0 P_abs(end)])
% ylim([0 inf])
% set(gca,'Fontsize',16)
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gcf,'color','w')
% box on

% figure
% % title('f(\Delta\mu)')
% % scatter(P_abs, dmu_dmu,200,'x','MarkerEdgeColor', colors(1,:),'LineWidth',3);
% errorbar(P_abs,dmu_log,delta_dmu_log,delta_dmu_log,[],[], '+', 'markerSize',10,'color', colors(1,:),'LineWidth', 3);
% hold on
% plot(P_abs, mu_dmu_nfitdata,'--','color', colors(2,:),'LineWidth',2)
% axis
% ylabel('$\Delta\mu$ (eV)','Interpreter','Latex')
% xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
% box on
% xlim([P_abs(1) P_abs(end)])
% ylim([0 inf])
% set(gca,'Fontsize',16)
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gca,'XScale','log')
% set(gcf,'color','w')

% mu_ref_dmu_init = mu_ref_log(2)-var_mu_ref_log(2);
% for i=[1:3]
%     mu_r(i) = mu_ref_dmu_init;
%     mu_ref_dmu_init = mu_ref_dmu_init+var_mu_ref_log(2);
% figure
% for i1=1:length(Int)
%     plot(E, ratio{i1},'--','color', colors(i1,:),'linewidth',2)
%     hold on
%     plot(E,dmu_T_ratio(T_log_raw(i1), dmu_log(i1), mu_r(i),E,T_L,m_e,m_h,Eg,D),'color', colors(i1,:))
%     hold on
% end
% xlim([1.45 1.6])
% xlabel('$E \: \mathrm{(eV)}$','Interpreter','Latex')
% ylabel('$\phi_n/\phi_1$','Interpreter','Latex')
% box on
% set(gca,'Fontsize',16)
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gcf,'color','w')
% box on
% end


E_min = E_min_init-0.03;
E_max = E_max_init-0.03;
figure
for i = 1:7
E_min = E_min + Inv_diff;
E_max = E_max + Inv_diff;
plot(P_abs, delta_mu_error_log(i,:),'LineWidth',2);
leg{i} = [num2str(E_min) '-' num2str(E_max)];
hold on
end
hold on
ylabel('$\Delta\mu$ error (eV)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([0 P_abs(end)])
ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gcf,'color','w')
legend(leg,'location','southeast')
box on


E_min = E_min_init-0.03;
E_max = E_max_init-0.03;
figure
for i = 1:7
E_min = E_min + Inv_diff;
E_max = E_max + Inv_diff;
plot(P_abs, T_error_log(i,:),'LineWidth',2);
leg{i} = [num2str(E_min) '-' num2str(E_max)];
hold on
end
hold on
ylabel('$T$ error (K)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([P_abs(1) P_abs(end)])
ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gcf,'color','w')
legend(leg,'location','southeast')
box on

E_min = E_min_init-0.03;
E_max = E_max_init-0.03;
figure
for i = 1:7
E_min = E_min + Inv_diff;
E_max = E_max + Inv_diff;
plot(P_abs, T_log(i,:)-T_L,'LineWidth',2);
leg{i} = [num2str(E_min) '-' num2str(E_max)];
hold on
end
hold on
ylabel('$T$ (K)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([0 P_abs(end)])
% ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gcf,'color','w')
legend(leg)
box on

E_min = E_min_init-0.03;
E_max = E_max_init-0.03;
figure
for i = 1:7
E_min = E_min + Inv_diff;
E_max = E_max + Inv_diff;
semilogx(P_abs, delta_mu_log(i,:),'LineWidth',2);
leg{i} = [num2str(E_min) '-' num2str(E_max)];
hold on
end
hold on
ylabel('$\Delta\mu $ (eV)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([0 P_abs(end)])
ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gcf,'color','w')
legend(leg)
box on
% 
% 
% figure
% for i1=1:length(Int)
%     semilogy(E, ratio{i1},'--','color', colors(i1,:),'linewidth',2)
%     hold on
%     semilogy(E,dmu_T_ratio(T_log_raw(i1), dmu_log(i1), mu_ref_log(2),E,T_L,m_e,m_h,Eg,D),'color', colors(i1,:))
%     hold on
% end
% xlim(E_plot)
% xlabel('$E \: \mathrm{(eV)}$','Interpreter','Latex')
% ylabel('$\phi_n/\phi_1$','Interpreter','Latex')
% box on
% set(gca,'Fontsize',16)
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gcf,'color','w')
% box on
% 
% figure
% for i1=1:length(Int)
%     plot(E, ratio{i1},'--','color', colors(i1,:),'linewidth',2)
%     hold on
%     plot(E,dmu_T_ratio(T_log_raw(i1), dmu_log(i1), mu_ref_log(2),E,T_L,m_e,m_h,Eg,D),'color', colors(i1,:))
%     hold on
% end
% xlim([1.45 1.56])
% xlabel('$E \: \mathrm{(eV)}$','Interpreter','Latex')
% ylabel('$\phi_n/\phi_1$','Interpreter','Latex')
% box on
% set(gca,'Fontsize',16)
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gcf,'color','w')
% box on
% 

