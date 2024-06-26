clc
clear

%% Loading the Data for Model Development 

% This code develops dynamic (hybrid series NLS - NLD) ECNN models for the 
% case when a time-invariant bias with / without Gaussian noise is added to 
% true data to generate training data. The other type of hybrid series ECNN
% i.e., NLD - NLS model can also be implemented. The sequential training
% algorithm will just be the reverse in that case. More details about the
% sequential training algorithms for the NLS - NLD and NLD - NLS models can
% be found in:

% Mukherjee, A. & Bhattacharyya, D. Hybrid Series/Parallel All-Nonlinear 
% Dynamic-Static Neural Networks: Development, Training, and Application to
% Chemical Processes. Ind. Eng. Chem. Res. 62, 3221–3237 (2023). 
% Available online at: pubs.acs.org/doi/full/10.1021/acs.iecr.2c03339

% This code requires the MATLAB Neural Network Toolbox to train the
% unconstrained all-nonlinear series (NLS - NLD) network model for 
% faster computation. 

% Load the training and validation datasets and specify the input and
% output variables to the NN models
% Note that the user can consider any dynamic dataset for training and
% validation. The rows signify the time steps and the columns signify the 
% input and output variables.

data_dyn = xlsread('Dynamic HX Data.xlsx','TimeInvBias+Gaussian Noise');
data_dyn = data_dyn(:,2:end);

% Though this approach (Approach 2) does not require partitioning the 
% entire time-series data into steady-state and dynamic zones, it still
% requires the identification of steady-state zones to apply the mass 
% balance constraints. However, the current dataset under consideration 
% shows a consistent perturbation of the system inputs after every 15 time 
% steps. 

t_steady = 15;

% In general, the steady-state zones have been identified in this work as 
% those continuous time periods in the overall data where the maximum 
% deviation is within +- 0.5% of the mean value in those zones.

% Since holdup measurements are not available for this system, for both
% hybrid parallel and hybrid series ECNN the energy balance constraints are
% applied on at steady-state during both training and forward problems

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

%% Defining the System Model in terms of Process Variables

data = data_dyn;

% Lumped Parameter Model: Shell and Tube Heat Exchanger considering one
% grid (lumped). Mass is automatically conserved. Input variables include
% the inlet flowrate, temperature and pressure of both steam and flue gas,
% while output variables consist of outlet temperature of both flue gas and
% steam

msteam_in = data(:,1); mfg_in = data(:,2);
Tsteam_in = data(:,3); Tfg_in = data(:,4);
Psteam_in = data(:,5); Pfg_in = data(:,6);

Tsteam_out = data(:,7); Tfg_out = data(:,8);

cpfg = data(:,9); cpsteam = data(:,10);

%------------------------------------------------------------------------%

%% Data Preparation for Model Training

tt = size(data,1);           % Total size of data
tn = floor(0.5*tt);          % Selecting 50% of total data for training

% Normalization of Inputs and Outputs

norm_mat = zeros(tt,nt);
delta = zeros(1,nt);
for i = 1:nt
    delta(1,i) = (max(data(:,i)) - min(data(:,i)));
    norm_mat(:,i) = (data(:,i)-min(data(:,i)))/(delta(1,i));
end

Imat = (norm_mat(:,1:ni))';
dsr = (norm_mat(:,ni+1:ni+no))';

% Selecting first tn steady-states from the overall data

tr_steps = 1:tn;
tr_steps = (sort(tr_steps))';

dsr_t = zeros(no,tn); Imat_t = zeros(ni,tn); 
msteam_t = zeros(tn,1); mfg_t = zeros(tn,1);
Tsteam_in_t = zeros(tn,1); Tfg_in_t = zeros(tn,1);
Psteam_in_t = zeros(tn,1); Pfg_in_t = zeros(tn,1);

cpfg_t = zeros(tn,1); cpsteam_t = zeros(tn,1);

for i = 1:tn
    ts = tr_steps(i,1);    
    dsr_t(1:no,i) = dsr(1:no,ts);
    Imat_t(1:ni,i) = Imat(1:ni,ts);
    msteam_t(i,1) = msteam_in(ts,1);
    mfg_t(i,1) = mfg_in(ts,1);
    Tsteam_in_t(i,1) = Tsteam_in(ts,1);
    Tfg_in_t(i,1) = Tfg_in(ts,1);
    Psteam_in_t(i,1) = Psteam_in(ts,1);
    Pfg_in_t(i,1) = Pfg_in(ts,1);
    cpfg_t(i,1) = cpfg(ts,1);
    cpsteam_t(i,1) = cpsteam(ts,1);
end

%----------------------------------------------------------------------%

%% Training of Unconstrained all-nonlinear series (NLS -NLD) Model

% In absence of energy constraints, training the model wrt measurement data
% Using NN Toolbox for Training the Unconstrained Network

[ynn_womc_t,nn_stat,nn_dyn,Xi,Ai] = TrainNLSNLD4SerECNN(Imat_t,dsr_t,nh,no,tn);

dsr_t_p = zeros(tn,no);
ynn_woec_t_p = zeros(tn,no);

for i = 1:no
    dsr_t_p(:,i) = dsr_t(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
    ynn_woec_t_p(:,i) = ynn_womc_t(:,i).*delta(1,ni+i) + min(data(:,ni+i));
end

% Saving the optimal solution to use as initial guess for the constrained
% NN models

whs = (nn_stat.IW{1})'; wos = (nn_stat.LW{2,1})'; 
bhs = nn_stat.b{1}; bos = nn_stat.b{2};
whd = (nn_dyn.IW{1})'; wfd = (nn_dyn.LW{1,2})'; wod = (nn_dyn.LW{2,1})'; 
bhd = nn_dyn.b{1}; bod = nn_dyn.b{2};

%-----------------------------------------------------------------------%

%% Training of the Constrained Hybrid Series NLS - NLD MCNN

n_st = floor(tn/t_steady);     % number of steady states

maxiter = 2;                   
yecnn_t = zeros(tn,no);
int_mat_1 = zeros(nh,tn);
x_in = Imat_t;                 % Initializing the intermediate variables
check1 = Inf;

% Initialize the NLS model

net_s = newff([0 1].*ones(ni,1),[nh, nh],{'logsig','logsig'},'trainlm');
net_s.IW{1} = whs'; net_s.LW{2,1} = wos';
net_s.b{1} = bhs; net_s.b{2} = bos;

for iter = 1:maxiter

    % Initialize the NLD model

    wd0 = [reshape(whd,[1 nh*nh]),reshape(wod,[1 nh*no]),reshape(wfd,[1 no*nh]),bhd',bod',reshape(ynn_woec_t_p,[1 tn*no])];

    lb = [-1e5.*ones(1,size(wd0,2)-(no*tn))';zeros(no*tn,1)];
    ub = 1e5.*ones(1,size(wd0,2))';

    obj = @(x)DynECNNInvProbV1(x,Imat_t,dsr_t,ni,nh,no,tn,delta,data);
    nlcon = @(x)DynECNNInvProbV1Cons(x,Imat_t,dsr_t,tn,ni,nh,no,msteam_t,mfg_t,Tsteam_in_t,Tfg_in_t,cpfg_t,cpsteam_t,delta,data,t_steady);
    nlrhs = [zeros(2*no*tn,1); heat_bal_st(1:n_st,1)];
    nle = [ones(2*no*tn,1);zeros(1*n_st,1)];

    opts = optiset('solver','ipopt','display','iter','maxiter',2,'maxtime',20000); 
    Opt = opti('fun',obj,'ineq',[],[],'nlmix',nlcon,nlrhs,nle,'bounds',lb,ub,'options',opts);

    [w_sol,fval,exitflag,info] = solve(Opt,wd0);

    whd = reshape(w_sol(1:ni*nh),[ni,nh]);
    wod = reshape(w_sol(ni*nh+1:ni*nh+nh*no),[nh,no]);
    wfd = reshape(w_sol(ni*nh+nh*no+1:ni*nh+nh*no+no*nh),[no,nh]);
    bhd = (w_sol(ni*nh+nh*no+no*nh+1:ni*nh+nh*no+no*nh+nh));
    bod = (w_sol(ni*nh+nh*no+no*nh+nh+1:ni*nh+nh*no+no*nh+nh+no));
    y_r = reshape(w_sol(ni*nh+nh*no+no*nh+nh+no+1:ni*nh+nh*no+no*nh+nh+no+tn*no),[tn,no]);

    yecnn_t(1,:) = dsr_t(:,1);

    for i = 2:tn
        yecnn_t(i,:) = purelin(wod'*tansig(whd'*Imat_t(:,i) + wfd'*(yecnn_t(i-1,:))' + bhd) + bod);
    end

    int_mat_1(i,:) = x_in(i,:);                  % Setting intermediate values for NLS - NLD architecture

    % Training of NLS Model

    net_s.trainParam.epochs = 5000;
    [net_s,tr] = train(net_s,Imat_t,int_mat_1);
    
    y_1 = sim(net_s,Imat_t);
    
    x_in = y_1;                                  % Updating the intermediate variables by direct substitution
    
    sse = sum((dsr_t' - yecnn_t).^2);
    mse = (1/(no*tn))*sum(sse);
    
    if mse <= check1       
        nn_stat = net_s;
        whdf = whd; wodf = wod; wfdf = wfd; bhdf = bhd; bodf = bod;
        ynn_final = y_r;
        check1 = mse;
    end  
    
end

yecnn_unnorm_t = ynn_final;       % outputs of constrained model, but yet to be post-processed

%------------------------------------------------------------------------%

%% POST-PROCESSING AFTER TRAINING TO GENERATE DESIRED OUTPUTS

data_steady = data_dyn((1:n_st)*t_steady,:);

output_dev_total = zeros(tn,no);

for i = 1:n_st
    output_dev_total((i-1)*t_steady+1:i*t_steady,:) = data_dyn((i-1)*t_steady+1:i*t_steady,ni+1:ni+no) - data_steady(i,ni+1:ni+no);
end

data_dev_t = output_dev_total;

yecnn_steady_t = zeros(tn,no);

for i = 1:n_st
    yecnn_steady_t((i-1)*t_steady+1:i*t_steady,1:no) = yecnn_unnorm_t(i*t_steady,1:no).*ones(t_steady,1);
end

ymcnn_total_t_p = data_dev_t + yecnn_steady_t;
ynn_total_woec_t_p = ynn_woec_t_p;
dsr_dyn_t_p = dsr_t_p;

%------------------------------------------------------------------------%
%% GENERATING TRAINING RESULTS

% Plotting Training Results for T_St_out (for the HX case study)

figure(1)
hold on
plot(yecnn_total_t_p(:,1),'b-x','MarkerSize',1,'LineWidth',1.5)
plot(ynn_total_woec_t_p(:,1),'g--','MarkerSize',1,'LineWidth',1.5)
plot(dsr_dyn_t_p(:,1),'r--o','MarkerSize',1,'LineWidth',1.2)
xlabel('Time (mins)')
ylabel('T_{St, out} (^oC)')
xlim([0 tn])
title('Training of Energy Constrained Neural Network (Hybrid Series)')
legend('ECNN','NN w/o EC','Measurements','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);

% Calculations for Error in Mass Balances

% NN w/o EC

Tsteam_out_dyn_woec_t = ynn_total_woec_t_p(:,1);
Tfg_out_dyn_woec_t = ynn_total_woec_t_p(:,2);

Qfg_dyn_woec_t = mfg_t.*cpfg_t.*(Tfg_in_t - Tfg_out_dyn_woec_t);
Qsteam_dyn_woec_t = msteam_t.*cpsteam_t.*(Tsteam_out_dyn_woec_t - Tsteam_in_t);

heat_bal_dyn_woec_t = 10.*abs(((Qfg_dyn_woec_t - Qsteam_dyn_woec_t)./1000)./(1e-3*Qfg_dyn_woec_t));       % MW

% ECNN

Tsteam_out_dyn_ecnn_t = yecnn_total_t_p(:,1);
Tfg_out_dyn_ecnn_t = yecnn_total_t_p(:,2);

Qfg_dyn_ecnn_t = mfg_t.*cpfg_t.*(Tfg_in_t - Tfg_out_dyn_ecnn_t);
Qsteam_dyn_ecnn_t = msteam_t.*cpsteam_t.*(Tsteam_out_dyn_ecnn_t - Tsteam_in_t);

heat_bal_dyn_ecnn_t = abs(((Qfg_dyn_ecnn_t - Qsteam_dyn_ecnn_t)./1000)./(1e-3*Qfg_dyn_ecnn_t));       % MW

figure(2)
hold on
plot(heat_bal_dyn_ecnn_t(:,1),'b','LineWidth',1.5)
plot(heat_bal_dyn_woec_t(:,1),'r-o','Markersize',3,'LineWidth',1.0)
xlabel('Time (mins)')
ylabel('APE in Energy Balance (%)')
% xlim([2840 3200])
legend('ECNN','NN w/o EC','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);

% END OF TRAINING (INVERSE PROBLEM)
%------------------------------------------------------------------------%

%% VALIDATION / SIMULATION / FORWARD PROBLEM: STEPS

% 1. The forward problem follows the same steps as the inverse problem.
% The entire time-series validation data are subjected to the optimal 
% NLS - NLD model followed by the Dynamic Data Reconciliation (DDR) block
% to impose constraints on the the identified steady-state zones.

% 2. The output deviation post-processing is performed to generate the
% overall dynamic outputs from the optimal hybrid series ECNN model. The
% validation results from the unconstrained NLS - NLD model can be
% generated by running the 'ValNLSNLD4SerECNN.m' function file.

% 3. The results obtained from the ECNN are compared with those obtained 
% from the unconstrained NN model.


%------------------------------------------------------------------------%



