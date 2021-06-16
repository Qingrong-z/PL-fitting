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


% % Get the value of the gap with the low intensity experience (no heat)
% Eg0 = 1.52; %eV at 0K
% alpha = 8.871*10^(-4); % eV/kelvin
% beta = 572; %Kelvin

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
                
                E_min = 1.45;
                E_max = 1.56;
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
                
                E_min = 1.43;
                E_max = 1.58;
                E_plot=[1.35 1.6];
                Eg = 1.424; %Band gap (eV)
                
            case '1926_2_5'
                
                points_removed=[6 7];
                Int(points_removed)=[];
                power(points_removed)=[]; %power is in W.cm^-2
                
                E_min = 1.45;
                E_max = 1.59;
                E_plot=[1.35 1.6];
                Eg = 1.424; %Band gap (eV)                      
        end
        
end

colors=lines(length(Int));
P_inc=power/(1000*spot_surface); % incident power (W.cm^-2)
E_laser=h*c/(q*laser*1e-9); %Energy of the laser (eV)
%% Calculate the ratios with band filling
%Parameter intialization and limits
% T=zeros(size(Int));
% mu=zeros(size(Int));
% mu_ref=zeros(size(Int));

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

mu_ref=mu_ref_vec(idx_res);

for i1=2:length(Int)
    T(i1)=x_sol{i1,idx_res}(1);
    mu(i1)=x_sol{i1,idx_res}(2);
end
T(1)=T_L;
mu(1)=mu_ref;

%% Log ratio fit
for i3 = 2:length(Int)
    y = log(ratio{1,i3}(Einv(E_min):Einv(E_max)));
    x = E(Einv(E_min):Einv(E_max));
    modelfun = @(a,x) log(dmu_T_ratio(a(1), a(2), a(3),x,T_L,m_e,m_h,Eg,D));
    options = statset('FunValCheck','off','DerivStep',10^-13,'robust','on');
    beta0_log_init{i3} = [T(i3);mu(i3)-mu_ref;mu_ref]; %Use the previous data as initial guess
    [beta_log_init{i3},r,J,cov] = nlinfit(x,y,modelfun,beta0_log_init{i3},options);
    T_log_init(i3) = beta_log_init{i3}(1,1);% Initial Temperature
    delta_mu_log_init(i3) = beta_log_init{i3}(2,1);% Initial delta_mu
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
E_min_init = E_min;
E_max_init = E_max;
E_min = E_min_init-0.02;
E_max = E_max_init-0.02;
for i = 1:3
    E_min = E_min + Inv_diff;
    E_max = E_max + Inv_diff;
    for i4 = 2:length(Int)
    y = log(ratio{1,i4}(Einv(E_min):Einv(E_max)));
    x = E(Einv(E_min):Einv(E_max));
    modelfun = @(a,x) log(dmu_T_ratio(a(1), a(2), a(3),x,T_L,m_e,m_h,Eg,D));
    options = statset('FunValCheck','off','DerivStep',10^-12,'robust','on');
    beta0_log_inv{i4} = [T_log_init(i4);delta_mu_log_init(i4);mu_ref_log_init(i4)]; % Use the obtained logarithm results
    [beta_log_inv{i4},r,J,cov] = nlinfit(x,y,modelfun,beta0_log_inv{i4},options);
%     T_log_inv(i,i4) = beta_log_inv{i4}(1,1);
%     delta_mu_log_inv(i,i4) = beta_log_inv{i4}(2,1);
    mu_ref_log_inv(i,i4) = beta_log_inv{i4}(3,1);
%     mu_nfit_log_inv(i,i4) = beta_log_inv{i4}(2,1)+beta_log_inv{i4}(3,1);
    ci_log_inv{i,i4} = nlparci(beta_log_inv{i4},r,'covar',cov);
    cj_log_inv{i,i4} = nlparci(beta_log_inv{i4},r,'Jacobian',J);
    end
    mu_ref_log_inv(i,1) = NaN;
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
 
 mu_ref_log = sum(mu_ref_top)./sum((sigma_inverse.^2));%Reverse-variance weighting method
 var_mu_ref_log = 2.*sqrt(1./sum((sigma_inverse.^2)));% 95% confidence interval


E_min = E_min_init;
E_max = E_max_init;
%% Refit with obtained optimal mu_ref
for i6 = 1:length(Int)
    y = log(ratio{1,i6}(Einv(E_min):Einv(E_max)));
    x = E(Einv(E_min):Einv(E_max));
    modelfun = @(a,x) log(dmu_T_ratio(a(1), a(2), mu_ref_log,x,T_L,m_e,m_h,Eg,D));
    options = statset('FunValCheck','off','DerivStep',10^-12,'robust','on');
    beta0_log{i6} = [T_log_init(i6);delta_mu_log_init(i6)];% Take the results of last step as initial guess, and fix mu_ref
    [beta_log{i6},r,J,cov] = nlinfit(x,y,modelfun,beta0_log{i6},options);
    T_log(i6) = beta_log{i6}(1,1); % Temperature
    delta_mu_log(i6) = beta_log{i6}(2,1); % Delta_mu
    mu_log(i6) = beta_log{i6}(2,1)+mu_ref_log;
    ci_log{i6} = nlparci(beta_log{i6},r,'covar',cov);
    cj_log{i6} = nlparci(beta_log{i6},r,'Jacobian',J);
    T_error_log(i6) = (ci_log{i6}(1,2)-ci_log{i6}(1,1))/2; % fitting error of temperature
    delta_mu_error_log(i6) = (ci_log{i6}(2,2)-ci_log{i6}(2,1))/2; % fitting error of Delta_mu
end


%% Plus Minus interval fitting
Inv_diff = 0.01; % Fit the interval from -0.01 to +0.01
E_min = E_min_init-0.02;
E_max = E_max_init-0.02;
for i = 1:3
    E_min = E_min + Inv_diff;
    E_max = E_max + Inv_diff;
    mu_ref_log_err = [mu_ref_log+var_mu_ref_log;mu_ref_log;mu_ref_log-var_mu_ref_log];
    for i7 = 1:length(Int)
    y = log(ratio{1,i7}(Einv(E_min):Einv(E_max)));
    x = E(Einv(E_min):Einv(E_max));
    modelfun = @(a,x) log(dmu_T_ratio(a(1), a(2), mu_ref_log_err(i),x,T_L,m_e,m_h,Eg,D));
    options = statset('FunValCheck','off','DerivStep',10^-12,'robust','on');
    beta0_log_inv{i7} = [T_log_init(i7);delta_mu_log_init(i7)]; % Use the obtained logarithm results
    [beta_log_inv{i7},r,J,cov] = nlinfit(x,y,modelfun,beta0_log_inv{i7},options);
    T_log_inv(i,i7) = beta_log_inv{i7}(1,1);
    delta_mu_log_inv(i,i7) = beta_log_inv{i7}(2,1);
%     mu_nfit_log_inv(i,i7) = beta_log_inv{i7}(2,1)+beta_log_inv{i7}(3,1);
    ci_log_inv{i,i7} = nlparci(beta_log_inv{i7},r,'covar',cov);
    cj_log_inv{i,i7} = nlparci(beta_log_inv{i7},r,'Jacobian',J);
    end
end

%% Error calculation
for i=1:length(Int)
    T_log_raw(i) = T_log_inv(2,i);
    T_plus_log(i) = T_log_inv(3,i);
    T_min_log(i) = T_log_inv(1,i);
    delta_T_interval_log(i) = max(abs(T_plus_log(i)-T_log_raw(i)),abs(T_log_raw(i)-T_min_log(i)));% interval error
    delta_T_init(i) = T_error_log(i); % fitting error
    delta_T_log(i) = delta_T_interval_log(i)+delta_T_init(i); % total error
    
    dmu_log(i) = delta_mu_log_inv(2,i);
    dmu_plus_log(i) = delta_mu_log_inv(3,i);
    dmu_min_log(i) = delta_mu_log_inv(1,i);
    delta_dmu_interval_log(i) = max(abs(dmu_plus_log(i)-dmu_log(i)),abs(dmu_log(i)-dmu_min_log(i))); % interval error
    delta_mu_log_init(i) = delta_mu_error_log(i);% fitting error
    delta_dmu_log(i) = delta_mu_log_init(i)+delta_dmu_interval_log(i);% total error
    
    mu_log_error(i) = sqrt(var_mu_ref_log^2+delta_dmu_log(i)^2);
end
mu_log_error(1) = var_mu_ref_log;
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

% Errors
delta_power_rel=0.1; %relative error on power
delta_spot_surface_rel=0.2; %relative error on spot_surface
delta_A_GaAs=max(abs(A_laser_GaAs_plus1-A_laser_GaAs),abs(A_laser_GaAs-A_laser_GaAs_min1));
delta_A_laser_rel=(delta_A_GaAs+0.1*A_laser_barriers)/A_laser; %relative error on absorptivity
delta_P_abs_rel=sqrt(delta_power_rel^2+delta_spot_surface_rel^2+delta_A_laser_rel^2); %relative error on the absorbed power: systematic error
delta_P_gen_rel=delta_P_abs_rel;


%% Linear fit

switch sample
    case '1873_1_4'
        T_interv=[3:length(T)];
        mu_interv=[1:5];
    case '1927_2_1'
        T_interv=[3:length(T)];
        mu_interv=[1:3];
    case '1926_2_5'
        T_interv=[2:length(T)];
        mu_interv=[1:4];
    case '1872_1_4'
        T_interv=[2:length(T)];
        mu_interv=[1:4];
    otherwise
        T_interv=[1:length(T)];
        mu_interv=[1:length(mu)];
end

p_dmu1 = polyfit(T_log(T_interv)-T_L,P_abs(T_interv),1); %Temperature fit
Q_dmu_nfit = p_dmu1(1);
deltaT0_dmu_nift = p_dmu1(2);
deltaT_dmu_nfitdata = deltaT0_dmu_nift + Q_dmu_nfit*(T_log-T_L);

p_dmu2 = polyfit(log(P_abs(mu_interv)),delta_mu_log(mu_interv),1); %mu fit
mu_slope_dmu_nfit = p_dmu2(1);
mu_ref_dmu_nfit0 = p_dmu2(2);
mu_dmu_nfitdata = mu_ref_dmu_nfit0+mu_slope_dmu_nfit*log(P_abs);

p_dmu3 = polyfit(log(P_abs(mu_interv)),mu_log(mu_interv),1); %mu fit
mu_slope_dmu = p_dmu3(1);
mu_ref_dmu = p_dmu3(2);
mu_nfitdata = mu_ref_dmu+mu_slope_dmu*log(P_abs);

delta_Q_rel=delta_P_abs_rel;
delta_Q=Q_dmu_nfit*delta_Q_rel;
%% Plots

figure
semilogx(P_abs, mu_ref_log_init,'LineWidth',3)
hold on
errorbar(P_abs,mu_ref_log_init,2*mu_ref_log_sigma,2*mu_ref_log_sigma,[],[], '+', 'markerSize',10,'color', colors(2,:),'LineWidth', 3);
ylabel('$\mu_{ref}$ (eV)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([0 P_abs(end)])
% ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gcf,'color','w')
box on

figure
errorbar(P_abs,delta_mu_log,delta_dmu_log,delta_dmu_log,[],[], '+', 'markerSize',10,'color', colors(1,:),'LineWidth', 3);
hold on
plot(P_abs, mu_dmu_nfitdata,'--','color', colors(2,:),'LineWidth',2)
axis
ylabel('$\Delta\mu$ (eV)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([P_abs(1) P_abs(end)])
% ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'XScale','log')
set(gcf,'color','w')

figure
hold on
errorbar(P_abs,T_log-T_L,delta_T_log,delta_T_log,[],[], '+', 'markerSize',10,'color', colors(1,:),'LineWidth', 3);
plot(deltaT_dmu_nfitdata,T_log_raw-T_L,'--','color', colors(2,:),'LineWidth', 2);
ylabel('$T-T_L$ (K)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([0 P_abs(end)])
ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gcf,'color','w')
box on

figure
errorbar(P_abs,mu_log,mu_log_error,mu_log_error,[],[], '+', 'markerSize',10,'color', colors(1,:),'LineWidth', 3);
hold on
plot(P_abs, mu_nfitdata,'--','color', colors(2,:),'LineWidth',2)
axis
ylabel('$\mu$ (eV)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([P_abs(1) P_abs(end)])
% ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'XScale','log')
set(gcf,'color','w')

x = P_abs;
y1 = (mu_ref_log+var_mu_ref_log).*ones(length(x));
y2 = (mu_ref_log-var_mu_ref_log).*ones(length(x));
figure
hold on
hold all
errorbar(P_abs,mu_ref_log_init,2*mu_ref_log_sigma,2*mu_ref_log_sigma,[],[], '+', 'markerSize',10,'color', colors(2,:),'LineWidth', 3);
hold on
plot(x, y1,'HandleVisibility','off');
plot(x, y2,'HandleVisibility','off');
yline(mu_ref_log,'--','color','m')
X=[x,fliplr(x)];               
Y=[y1,fliplr(y2)];
fill(X,Y,'b','FaceAlpha',0.1,'EdgeAlpha',0);
ylabel('$\mu_{ref}$ (eV)','Interpreter','Latex')
xlabel('$P_{abs} \: (\mathrm{W.cm^{-2})}$','Interpreter','Latex')
box on
xlim([0 P_abs(end)])
% ylim([0 inf])
set(gca,'Fontsize',16)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gcf,'color','w')
box on
legend('initial \mu_{ref} error','optimal \mu_{ref}','\mu_{ref} error','location','SouthEast')
