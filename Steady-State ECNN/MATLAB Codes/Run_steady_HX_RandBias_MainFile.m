clc
clear

%% Loading the Data for Model Development

% This code develops steady-state MCNN models for the case when a random
% bias with / without Gaussian noise is added to true data to generate 
% training data. A linear noise model is considered.

% This code requires the MATLAB Neural Network Toolbox to train the
% unconstrained neural network model for faster computation. The optimal
% solution of the unconstrained network serve as initial guesses for the
% constrained formulation of the training problem.

% Load the training and validation datasets and specify the input and
% output variables to the NN models.
% Note that the user can consider any steady-state dataset for training and
% validation. The rows signify the time steps and the columns signify the 
% input and output variables.

data = xlsread('Steady-State HX Data.xlsx','RandomBias+Gaussian Noise');
data = data(:,2:end);

% For this specific system, the first six columns are the model inputs and
% the following two columns are the model outputs. The next two columns
% denote the specific heats of flue gas and steam respectively based on
% appropriate calculations / functions.

input_data = data(:,1:6); output_data = data(:,7:8);
cp_data = data(:,9:10);

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

heat_bal_st = data(:,11);

ni = size(input_data,2);             % Number of inputs = Number of neurons in input layer
no = size(output_data,2);            % Number of outputs = Number of neurons in output layer

% Default number of neurons in hidden layer taken equal to number of
% inputs. But it can be changed as per requirement.

nh = ni;

nt = ni + no;

% Total number of datasets available for model development

n = size(data,1);

%-----------------------------------------------------------------------%

%% Defining the System Model in terms of Process Variables

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

%% Training of Unconstrained NN Model

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

Tsteam_out_st_wo = ynn_wo_t_p(:,1);
Tfg_out_st_wo = ynn_wo_t_p(:,2);

Qfg_st_wo = mfg_st_t.*cpfg_st_t.*(Tfg_in_st_t - Tfg_out_st_wo);
Qsteam_st_wo = msteam_st_t.*cpsteam_st_t.*(Tsteam_out_st_wo - Tsteam_in_st_t);

heat_bal_st_wo = 100.*abs(((Qfg_st_wo - Qsteam_st_wo)./1000 - heat_bal_st(tr_steps,:))./(1e-3*Qfg_st_wo));       % MW

% Saving the optimal solution to use as initial guess for the constrained
% NN models

whf_wo_opt = whf_wo; wof_wo_opt = wof_wo; 
bhf_wo_opt = bhf_wo; bof_wo_opt = bof_wo;

%-----------------------------------------------------------------------%

%% Training of the Constrained Network

wh0 = whf_wo_opt; wo0 = wof_wo_opt;
bh0 = bhf_wo_opt; bo0 = bof_wo_opt;

w0 = [reshape(wh0,[1 ni*nh]),reshape(wo0,[1 nh*no]),bh0',bo0',reshape(ynn_wo_t_p,[1 no*tn]),zeros(1,no)];

lb = [-1e5.*ones(1,size(w0,2)-(no*tn)-no)';zeros(no*tn,1);-1e5.*ones(no,1)];
ub = 1e5.*ones(1,size(w0,2))';

% The constrained optimization is performed using the IPOPT solver in the
% OPTI Toolbox 

obj = @(x)InvProbV2(x,Imat_t,dsr_t,ni,nh,no,tn,delta,data);
nlcon = @(x)InvProbV2Cons(x,Imat_t,tn,ni,nh,no,msteam_st_t,mfg_st_t,Tsteam_in_st_t,Tfg_in_st_t,cpfg_st_t,cpsteam_st_t,delta,data);
nlrhs = [zeros(2*no*tn,1); heat_bal_st(tr_steps,1)];
nle = [ones(2*no*tn,1);zeros(1*tn,1)];

opts = optiset('solver','ipopt','display','iter','maxiter',1e1,'maxtime',3600); 
Opt = opti('fun',obj,'ineq',[],[],'nlmix',nlcon,nlrhs,nle,'bounds',lb,ub,'options',opts);

[w_sol,fval,exitflag,info] = solve(Opt,w0);

whf = reshape(w_sol(1:ni*nh),[ni,nh]);
wof = reshape(w_sol(ni*nh+1:ni*nh+nh*no),[nh,no]);
bhf = (w_sol(ni*nh+nh*no+1:ni*nh+nh*no+nh));
bof = (w_sol(ni*nh+nh*no+nh+1:ni*nh+nh*no+nh+no));
y_r = reshape(w_sol(ni*nh+nh*no+nh+no+1:ni*nh+nh*no+nh+no+no*tn),[tn,no]);

af = w_sol(ni*nh+nh*no+nh+no+no*tn+1:ni*nh+nh*no+nh+no+no*tn+no);

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

Tsteam_out_st_t = ynn_t_p(:,1);
Tfg_out_st_t = ynn_t_p(:,2);

Qfg_st_t = mfg_st_t.*cpfg_st_t.*(Tfg_in_st_t - Tfg_out_st_t);
Qsteam_st_t = msteam_st_t.*cpsteam_st_t.*(Tsteam_out_st_t - Tsteam_in_st_t);

heat_bal_st_t = 100.*abs(((Qfg_st_t - Qsteam_st_t)./1000 - heat_bal_st(tr_steps,1))./(1e-3*Qfg_st_t));

% END OF TRAINING
%------------------------------------------------------------------------%

%% Validation / Simulation of Unconstrained Network

flag = 1;
tv = tt - tn;
val_steps = zeros(tv,1);

for i = 1:tt
    check = ismember(i,tr_steps);    
    if check == 0
        val_steps(flag,1) = i;
        flag = flag+1;
    end
end

val_steps = sort(val_steps);

dsr_v = zeros(no,tv); Imat_v = zeros(ni,tv); 
msteam_st_v = zeros(tv,1); mfg_st_v = zeros(tv,1);
Tsteam_in_st_v = zeros(tv,1); Tfg_in_st_v = zeros(tv,1);
Psteam_in_st_v = zeros(tv,1); Pfg_in_st_v = zeros(tv,1);

cpfg_st_v = zeros(tv,1); cpsteam_st_v = zeros(tv,1);

for i = 1:tv
    ts = val_steps(i,1);    
    dsr_v(1:no,i) = dsr(1:no,ts);
    Imat_v(1:ni,i) = Imat(1:ni,ts);
    msteam_st_v(i,1) = msteam_st(ts,1);
    mfg_st_v(i,1) = mfg_st(ts,1);
    Tsteam_in_st_v(i,1) = Tsteam_in_st(ts,1);
    Tfg_in_st_v(i,1) = Tfg_in_st(ts,1);
    Psteam_in_st_v(i,1) = Psteam_in_st(ts,1);
    Pfg_in_st_v(i,1) = Pfg_in_st(ts,1);
    cpfg_st_v(i,1) = cpfg_st(ts,1);
    cpsteam_st_v(i,1) = cpsteam_st(ts,1);
end

y1 = Imat_v;
x1 = whf_wo_opt'*y1 + bhf_wo_opt;
y2 = logsig(x1);
x2 = wof_wo_opt'*y2 + bof_wo_opt;
ynn_wo_v = purelin(x2);

ynn_wo_v_p = zeros(tv,no);

for i = 1:no
    ynn_wo_v_p(:,i) = ynn_wo_v(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
end

Tsteam_out_st_NN_v_wo = ynn_wo_v_p(:,1);
Tfg_out_st_NN_v_wo = ynn_wo_v_p(:,2);

Qfg_st_v_wo = mfg_st_v.*cpfg_st_v.*(Tfg_in_st_v - Tfg_out_st_NN_v_wo);
Qsteam_st_v_wo = msteam_st_v.*cpsteam_st_v.*(Tsteam_out_st_NN_v_wo - Tsteam_in_st_v);

heat_bal_st_v_wo = 100.*abs(((Qfg_st_v_wo - Qsteam_st_v_wo)./1000 - heat_bal_st(val_steps,1))./(1e-3*Qfg_st_v_wo));

%------------------------------------------------------------------------%

%% Validation / Simulation of ECNN (including Dynamic Data Reconciliation Post-Processing Step)

y1 = Imat_v;
x1 = whf'*y1 + bhf;
y2 = logsig(x1);
x2 = wof'*y2 + bof;
ynn_v = purelin(x2);

ynn_v_p = zeros(tv,no);

for i = 1:no
    ynn_v_p(:,i) = ynn_v(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
end
    
ynn_v = ynn_v_p';

% Dynamic Data Reconciliation Post-Processing

disp('START: Data Reconciliation Step during Validation')

ynn_v0 = reshape(1*ynn_v,[1 no*tv]);

lb = -1e5.*ones(1,size(ynn_v0,2))';
ub = 1e5.*ones(1,size(ynn_v0,2))';

obj_v = @(x)ForwProbV1(x,ynn_v,tv,no);
nlcon_v = @(x)ForwProbV1Cons(x,ynn_v,tv,no,msteam_st_v,mfg_st_v,Tsteam_in_st_v,Tfg_in_st_v,cpfg_st_v,cpsteam_st_v);
nlrhs_v = [zeros(no*tv,1); heat_bal_st(val_steps,1)];
nle_v = [ones(no*tv,1);zeros(1*tv,1)];

opts_v = optiset('solver','ipopt','display','iter','maxiter',1e2,'maxtime',3600); 
Opt_v = opti('fun',obj_v,'ineq',[],[],'nlmix',nlcon_v,nlrhs_v,nle_v,'bounds',lb,ub,'options',opts_v);

[ynn_v_r,fval_r,exitflag_r,info_r] = solve(Opt_v,ynn_v0);

disp('END: Data Reconciliation Step during Validation')

ynn_v_r = reshape(ynn_v_r,size(ynn_v));

ynn_v_p = ynn_v_r';
dsr_v_p = zeros(tv,no);

for i = 1:no
    dsr_v_p(:,i) = (dsr_v(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
end

Tsteam_out_st_NN_v = ynn_v_p(:,1);
Tfg_out_st_NN_v = ynn_v_p(:,2);

Qfg_st_v = mfg_st_v.*cpfg_st_v.*(Tfg_in_st_v - Tfg_out_st_NN_v);
Qsteam_st_v = msteam_st_v.*cpsteam_st_v.*(Tsteam_out_st_NN_v - Tsteam_in_st_v);

heat_bal_st_v = 100.*abs(((Qfg_st_v - Qsteam_st_v)./1000 - heat_bal_st(val_steps,1))./(1e-3*Qfg_st_v));

%-----------------------------------------------------------------------%

%% Plotting Error in Energy Balance Plots for Training

figure(1)
hold on
plot(tr_steps,heat_bal_st_wo(:,1),'r');
plot(tr_steps,heat_bal_st_t(:,1),'b');
title('Lumped HX System');
xlabel('Indices for Training Data');
ylabel('APE in Energy Balance (%)');
legend('NN w/o EC','ECNN','Location','northeast')
grid on
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alllines=findall(a,'Type','line'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14); set(alllines,'Linewidth',1.2);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);


%% Plotting Error in Energy Balance Plots for Validation / Simulation

figure(2)
hold on
plot(val_steps,heat_bal_st_v_wo(:,1),'r');
plot(val_steps,heat_bal_st_v(:,1),'b');
title('Lumped HX System');
xlabel('Indices for Validation Data');
ylabel('APE in Energy Balance (%)');
legend('NN w/o EC','ECNN','Location','northeast')
grid on
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alllines=findall(a,'Type','line'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14); set(alllines,'Linewidth',1.2);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);





