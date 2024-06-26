function c = DynECNNInvProbV1Cons(x,Imat_t,dsr_t,tn,ni,nh,no,msteam_t,mfg_t,Tsteam_in_t,Tfg_in_t,cpfg_t,cpsteam_t,delta,data,t_steady)

whd = reshape(x(1:ni*nh),[ni,nh]);
wod = reshape(x(ni*nh+1:ni*nh+nh*no),[nh,no]);
wfd = reshape(x(ni*nh+nh*no+1:ni*nh+nh*no+no*nh),[no,nh]);
bhd = (x(ni*nh+nh*no+no*nh+1:ni*nh+nh*no+no*nh+nh))';
bod = (x(ni*nh+nh*no+no*nh+nh+1:ni*nh+nh*no+no*nh+nh+no))';

n_st = floor(size(dsr_t,2)/t_steady);
y_r = reshape(x(ni*nh+nh*no+no*nh+nh+no+1:ni*nh+nh*no+no*nh+nh+no+tn*no),[tn,no]);

bhdmat = zeros(nh,tn);
for i = 1:size(bhdmat,2)
    bhdmat(:,i) = bhd;
end

bodmat = zeros(no,tn);
for i = 1:size(bodmat,2)
    bodmat(:,i) = bod;
end

ynn_t = zeros(tn,no);

ynn_t(1,:) = dsr_t(:,1);

for i = 2:tn
    ynn_t(i,:) = purelin(wod'*tansig(whd'*Imat_t(:,i) + wfd'*(ynn_t(i-1,:))' + bhdmat(:,i)) + bodmat(:,i));
end

ynn2_t_p = zeros(tn,no);

for i = 1:no
    ynn2_t_p(:,i) = ynn_t(:,i).*delta(1,ni+i) + min(data(:,ni+i));
end

ynn1_t_p = y_r;

Tsteam_out_NN = ynn1_t_p(:,1);
Tfg_out_NN = ynn1_t_p(:,2);

c1 = [ynn1_t_p(:,1);ynn1_t_p(:,2);ynn2_t_p(:,1);ynn2_t_p(:,2)];

Qfg_NN = mfg_t.*cpfg_t.*(Tfg_in_t - Tfg_out_NN);
Qsteam_NN = msteam_t.*cpsteam_t.*(Tsteam_out_NN - Tsteam_in_t);

c2 = (Qfg_NN(t_steady.*(1:n_st),:) - Qsteam_NN(t_steady.*(1:n_st),:))./1000;

c = [c1;c2];

end