clc
clear

%% Loading the Data for Model Development followed by Partitioning

% This code develops dynamic (hybrid parallel) ECNN models for the case 
% when a time-invariant bias with / without Gaussian noise is added to 
% true data to generate training data

% This code requires the MATLAB Neural Network Toolbox to train the
% unconstrained all-nonlinear parallel (NLS || NLD) network model for 
% faster computation. 

% Data is partitioned into steady-state and dynamic zones. A steady-state 
% ECNN is trained followed by training a dynamic residual model to match 
% the transient measurements. The optimal solution of the unconstrained 
% steady-state network serve as initial guesses for the constrained 
% steady-state ECNN formulation in the inverse problem.

% Load the training and validation datasets and specify the input and
% output variables to the NN models
% Note that the user can consider any dynamic dataset for training and
% validation. The rows signify the time steps and the columns signify the 
% input and output variables.

data_dyn = xlsread('Dynamic HX Data.xlsx','TimeInvBias+Gaussian Noise');
data_dyn = data_dyn(:,2:end);

% Partitioning the entire time-series data into steady-state and dynamic
% zones require the identification of steady-state zones. However, the
% current dataset under consideration shows a consistent perturbation of
% the system inputs after every 15 time steps. 

t_steady = 15;
n = floor(size(data_dyn,1)/t_steady);  % number of steady-state zones

% In general, the steady-state zones have been identified in this work as 
% those continuous time periods in the overall data where the maximum 
% deviation is within +- 0.5% of the mean value in those zones

% Since holdup measurements are not available for this system, for both
% hybrid parallel and hybrid series ECNN the energy balance constraints are
% applied on at steady-state during both training and forward problems

% Creating a subset of steady-state data for developing the steady-state
% ECNN model

data_steady = [];

for i = 1:n
    data_steady = [data_steady; data_dyn(i*t_steady,:)];
end

% Note that the base (truth) case, i.e., the tab named 'NoNoise' represents
% the dataset obtained from simulation of an appropriate first-principles 
% model. It was also observed that the difference in heat (energy) balance 
% for the 'True Data', i.e., the data containing no noise was not exactly 
% zero in megawatts (MW) scale with respect to the superheater system. The 
% corresponding difference in Q (MW) is also provided in the excel spread-
% sheet with respect to the true data obtained by simulation of the first-
% principles model. Although the specific formulation of the training and 
% forward problems remain independent of the specfic types of constraints 
% imposed, in this work, it is desired to see whether the ECNN converge at 
% the 'True Data' considered even when trained with noisy transient data.
% Therefore, the corresponding constraint equations are modified to arrive 
% at the Delta_Q_True (MW) values calculated with respect to the 'True 
% Data'.

heat_bal_st = xlsread('Steady-State HX Data.xlsx','NoNoise','L:L');

% For this specific system, the first six columns are the model inputs and
% the following two columns are the model outputs

ni = 6;             % Number of inputs = Number of neurons in input layer
no = 2;            % Number of outputs = Number of neurons in output layer

% Default number of neurons in hidden layer taken equal to number of
% inputs. But it can be changed as per requirement.

nh = ni;

nt = ni + no;

% ---------------------------------------------------------------------- %

%% Development/Preparation of steady-state data in terms of Process Variables

data = data_steady;

% Lumped Parameter Model: Shell and Tube Heat Exchanger considering one
% grid (lumped). Mass is automatically conserved. Input variables include
% the inlet flowrate, temperature and pressure of both steam and flue gas,
% while output variables consist of outlet temperature of both flue gas and
% steam

% Steady-state values for input and output variables

msteam_st = data(:,1); mfg_st = data(:,2);
Tsteam_in_st = data(:,3); Tfg_in_st = data(:,4);
Psteam_in_st = data(:,5); Pfg_in_st = data(:,6);

Tsteam_out_st = data(:,7); Tfg_out_st = data(:,8);

cpfg_st = data(:,9); cpsteam_st = data(:,10);

%------------------------------------------------------------------------%
%% Data Preparation for Model Training

tt = size(data,1);           % Total size of data
tn = floor(0.7*tt);          % Selecting 70% of total data for training

% Normalization of Inputs and Outputs

norm_mat = zeros(tt,nt);
delta = zeros(1,nt);
for i = 1:nt
    delta(1,i) = (max(data(:,i)) - min(data(:,i)));
    norm_mat(:,i) = (data(:,i)-min(data(:,i)))/(delta(1,i));
end

Imat = (norm_mat(:,1:ni))';
dsr = (norm_mat(:,ni+1:ni+no))';

% Generating random training data for tn steps

tr_steps = randperm(tt,tn);
tr_steps = (sort(tr_steps))';

dsr_t = zeros(no,tn); Imat_t = zeros(ni,tn); 
msteam_st_t = zeros(tn,1); mfg_st_t = zeros(tn,1);
Tsteam_in_st_t = zeros(tn,1); Tfg_in_st_t = zeros(tn,1);
Psteam_in_st_t = zeros(tn,1); Pfg_in_st_t = zeros(tn,1);

cpfg_st_t = zeros(tn,1); cpsteam_st_t = zeros(tn,1);

for i = 1:tn
    ts = tr_steps(i,1);    
    dsr_t(1:no,i) = dsr(1:no,ts);
    Imat_t(1:ni,i) = Imat(1:ni,ts);
    msteam_st_t(i,1) = msteam_st(ts,1);
    mfg_st_t(i,1) = mfg_st(ts,1);
    Tsteam_in_st_t(i,1) = Tsteam_in_st(ts,1);
    Tfg_in_st_t(i,1) = Tfg_in_st(ts,1);
    Psteam_in_st_t(i,1) = Psteam_in_st(ts,1);
    Pfg_in_st_t(i,1) = Pfg_in_st(ts,1);
    cpfg_st_t(i,1) = cpfg_st(ts,1);
    cpsteam_st_t(i,1) = cpsteam_st(ts,1);
end

%----------------------------------------------------------------------%
%% Training of Unconstrained Steady-State NN Model

% In absence of energy constraints, training the model wrt measurement data

% Using NN Toolbox for Solving the Unconstrained Network

net_s = newff([0 1].*ones(ni,1),[nh no],{'logsig','purelin'},'trainscg');

net_s.trainParam.epochs = 20000;
[net_s,tr] = train(net_s,Imat_t,dsr_t);

whf_wo = (net_s.IW{1,1})'; wof_wo = (net_s.LW{2,1})'; 
bhf_wo = net_s.b{1}; bof_wo = net_s.b{2};

y1 = Imat_t;
x1 = whf_wo'*y1 + bhf_wo;
y2 = logsig(x1);
x2 = wof_wo'*y2 + bof_wo;
ynn_wo = purelin(x2);

% Calculating Energy Balance Errors for NN w/o mass constraints

ynn_wo_t_p = zeros(tn,no);

for i = 1:no
    ynn_wo_t_p(:,i) = (ynn_wo(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
end

% Saving the optimal solution to use as initial guess for the constrained
% NN models

whf_wo_opt = whf_wo; wof_wo_opt = wof_wo; 
bhf_wo_opt = bhf_wo; bof_wo_opt = bof_wo;

%-----------------------------------------------------------------------%
%% Training of the Constrained Network

wh0 = whf_wo_opt; wo0 = wof_wo_opt;
bh0 = bhf_wo_opt; bo0 = bof_wo_opt;

w0 = [reshape(wh0,[1 ni*nh]),reshape(wo0,[1 nh*no]),bh0',bo0',reshape(ynn_wo_t_p,[1 no*tn])];

lb = [-1e5.*ones(1,size(w0,2)-(no*tn))';zeros(no*tn,1)];
ub = 1e5.*ones(1,size(w0,2))';

% The constrained optimization is performed using the IPOPT solver in the
% OPTI Toolbox 

obj = @(x)InvProbV1(x,Imat_t,dsr_t,ni,nh,no,tn,delta,data);
nlcon = @(x)InvProbV1Cons(x,Imat_t,tn,ni,nh,no,msteam_st_t,mfg_st_t,Tsteam_in_st_t,Tfg_in_st_t,cpfg_st_t,cpsteam_st_t,delta,data);
nlrhs = [zeros(2*no*tn,1); heat_bal_st(tr_steps,1)];
nle = [ones(2*no*tn,1);zeros(1*tn,1)];

opts = optiset('solver','ipopt','display','iter','maxiter',4,'maxtime',3600); 
Opt = opti('fun',obj,'ineq',[],[],'nlmix',nlcon,nlrhs,nle,'bounds',lb,ub,'options',opts);

[w_sol,fval,exitflag,info] = solve(Opt,w0);

whf = reshape(w_sol(1:ni*nh),[ni,nh]);
wof = reshape(w_sol(ni*nh+1:ni*nh+nh*no),[nh,no]);
bhf = (w_sol(ni*nh+nh*no+1:ni*nh+nh*no+nh));
bof = (w_sol(ni*nh+nh*no+nh+1:ni*nh+nh*no+nh+no));
y_r = reshape(w_sol(ni*nh+nh*no+nh+no+1:ni*nh+nh*no+nh+no+no*tn),[tn,no]);

y1 = Imat_t;
x1 = whf'*y1 + bhf;
y2 = logsig(x1);
x2 = wof'*y2 + bof;
ynn_t = purelin(x2);

ynn_t_p = y_r;

dsr_t_p = zeros(tn,no);
ynn1_t_p = zeros(tn,no);

% Calculating Energy Balance Errors for NN w/ mass constraints

for i = 1:no
    dsr_t_p(:,i) = (dsr_t(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
    ynn1_t_p(:,i) = ynn_t(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
end

%------------------------------------------------------------------------%
%% Development of Dynamic Residual Model connected in parallel to ECNN

data_dyn_total_t = data_dyn(1:floor(tn*t_steady),:);
data_steady_t = data_steady(1:tn,:);

% Calculating deviation variables for inputs and outputs

input_dev_total_t = data_dyn_total_t(:,1:ni) - data_steady_t(1,1:ni);
output_dev_total_t = zeros(size(data_dyn_total_t,1),no);
data_steady_out_dev_t = zeros(tn,no);

for i = 1:tn
    data_steady_out_dev_t(i,1:no) = data_steady_t(i,ni+1:ni+no) - data_steady_t(1,ni+1:ni+no);
    output_dev_total_t((i-1)*t_steady+1:i*t_steady,:) = data_dyn_total_t((i-1)*t_steady+1:i*t_steady,ni+1:ni+no) - data_steady_out_dev_t(i,1:no);
end

dyn_dev_total_t = [input_dev_total_t output_dev_total_t];
dyn_dev_total_t = dyn_dev_total_t(t_steady-2:end,:);

data_dyn_t = dyn_dev_total_t;

tn_d = size(data_dyn_t,1); 

% Generating normalized inputs and outputs for the time-series data

norm_mat_dyn_t = zeros(tn_d,nt);
delta_dyn_t = zeros(1,nt);
for i = 1:nt
    delta_dyn_t(1,i) = (max(data_dyn_t(:,i)) - min(data_dyn_t(:,i)));
    norm_mat_dyn_t(:,i) = (data_dyn_t(:,i)-min(data_dyn_t(:,i)))/(delta_dyn_t(1,i));
end

Imat_dyn_t = (norm_mat_dyn_t(:,1:ni))';
dsr_dyn_t = (norm_mat_dyn_t(:,ni+1:nt))';

% Training a NARX-type RNN through the function 'TrainNLD4ParMCNN.m'
% Different packages can also be used for better accuracy

[ynn_dyn_t,nn_dyn,Xi,Ai] = TrainNLD4ParECNN(Imat_dyn_t,dsr_dyn_t,nh,no,tn_d);

% The dynamic deviation data contain sharp spikes at the transition points
% between steady-state and dynamics. Accurately predicting such spikes
% perfectly using a dynamic data-driven model can be challenging. So, as a
% post-processing step, the peaks are smoothened with values closer to the
% desired deviations.

% for i = 1:tn-1
%     ynn_dyn_t(i*t_steady+1:i*t_steady+3,:) = (dsr_dyn_t(:,i*t_steady+1:i*t_steady+3))';    
% end

% Converting normalized variables to absolute variables

ynn_dyn_t_p = zeros(tn_d,no);
dsr_dyn_t_p = data_dyn_total_t(:,ni+1:nt);

for i = 1:no
    ynn_dyn_t_p(:,i) = ynn_dyn_t(:,i).*delta_dyn_t(1,ni+i) + min(data_dyn_t(:,ni+i));
end

%------------------------------------------------------------------------%

%% GENERATING TRAINING RESULTS FROM THE OVERALL HYBRID PARALLEL ECNN AND UNCONSTRAINED MODEL

yecnn_stat_t_p = zeros(tn_d,no);
ynn_stat_woec_t_p = zeros(tn_d,no);

yecnn_stat_t_p(1:3,1:no) = ynn_t_p(1,:).*ones(3,1);
ynn_stat_woec_t_p(1:3,1:no) = ynn_wo_t_p(1,:).*ones(3,1);

for i = 1:tn-1
    yecnn_stat_t_p(3+(i-1)*t_steady+1:3+i*t_steady,1:no) = ynn_t_p(i+1,1:no).*ones(t_steady,1);
    ynn_stat_woec_t_p(3+(i-1)*t_steady+1:3+i*t_steady,1:no) = ynn_wo_t_p(i+1,1:no).*ones(t_steady,1);
end

heat_bal_dyn_t = zeros(tn_d,1);
for i = 1:tn
    heat_bal_dyn_t((i-1)*t_steady+1:i*t_steady,1) = heat_bal_st(tr_steps(i,1),1).*ones(t_steady,1);
end

yecnn_total_t_p = yecnn_stat_t_p + ynn_dyn_t_p - data_steady_t(1,ni+1:ni+no);
ynn_total_wemc_t_p = ynn_stat_woec_t_p + ynn_dyn_t_p - data_steady_t(1,ni+1:ni+no);

% Calculations for Error in Mass Balances

msteam_in_dyn_t = data_dyn_total_t(t_steady-2:end,1); 
mfg_in_dyn_t = data_dyn_total_t(t_steady-2:end,2);
Tsteam_in_dyn_t = data_dyn_total_t(t_steady-2:end,3); 
Tfg_in_dyn_t = data_dyn_total_t(t_steady-2:end,4);
Psteam_in_dyn_t = data_dyn_total_t(t_steady-2:end,5); 
Pfg_in_dyn_t = data_dyn_total_t(t_steady-2:end,6);

cpfg_in_dyn_t = data_dyn_total_t(t_steady-2:end,9); 
cpsteam_in_dyn_t = data_dyn_total_t(t_steady-2:end,10);

% NN w/o EC

Tsteam_out_dyn_woec_t = ynn_total_wemc_t_p(:,1);
Tfg_out_dyn_woec_t = ynn_total_wemc_t_p(:,2);

Qfg_dyn_woec_t = mfg_in_dyn_t.*cpfg_in_dyn_t.*(Tfg_in_dyn_t - Tfg_out_dyn_woec_t);
Qsteam_dyn_woec_t = msteam_in_dyn_t.*cpsteam_in_dyn_t.*(Tsteam_out_dyn_woec_t - Tsteam_in_dyn_t);

heat_bal_dyn_woec_t = 10.*abs(((Qfg_dyn_woec_t - Qsteam_dyn_woec_t)./1000 - heat_bal_dyn_t(t_steady-2:end,1))./(1e-3*Qfg_dyn_woec_t));       % MW

% ECNN

Tsteam_out_dyn_ecnn_t = yecnn_total_t_p(:,1);
Tfg_out_dyn_ecnn_t = yecnn_total_t_p(:,2);

Qfg_dyn_ecnn_t = mfg_in_dyn_t.*cpfg_in_dyn_t.*(Tfg_in_dyn_t - Tfg_out_dyn_ecnn_t);
Qsteam_dyn_ecnn_t = msteam_in_dyn_t.*cpsteam_in_dyn_t.*(Tsteam_out_dyn_ecnn_t - Tsteam_in_dyn_t);

heat_bal_dyn_ecnn_t = abs(((Qfg_dyn_ecnn_t - Qsteam_dyn_ecnn_t)./1000 - heat_bal_dyn_t(t_steady-2:end,1))./(1e-3*Qfg_dyn_ecnn_t));       % MW

figure(1)
hold on
plot(heat_bal_dyn_ecnn_t(:,1),'b','LineWidth',1.5)
plot(heat_bal_dyn_woec_t(:,1),'r-o','Markersize',3,'LineWidth',1.0)
xlabel('Time (mins)')
ylabel('APE in Energy Balance (%)')
xlim([2840 3200])
legend('ECNN','NN w/o EC','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);

% END OF TRAINING (INVERSE PROBLEM)
%------------------------------------------------------------------------%

%% VALIDATION / SIMULATION / FORWARD PROBLEM: STEPS

% 1. The forward problem follows the same steps as the inverse problem.
% From the validation data the steady-state zones are identified and the
% steady-state forward problem as included in the MATLAB and Python codes
% for Steady-State ECNN can be implemented.

% 2. The input deviation matrix is formed for implementation through the 
% optimal NLD (NARX-type RNN) model obtained by running the 'ValNLD4ParECNN.m'
% function.

% 3. The resulting variables are converted from the normalized scale to
% their absolute scales of magnitude and the overall results are generated.
% The results are compared with those obtained from the unconstrained NN
% model.


%------------------------------------------------------------------------%

















