function c = InvProbV1Cons(x,Imat_t,tn,ni,nh,no,msteam_st_t,mfg_st_t,Tsteam_in_st_t,Tfg_in_st_t,cpfg_st_t,cpsteam_st_t,delta,data)

wh = reshape(x(1:ni*nh),[ni,nh]);
wo = reshape(x(ni*nh+1:ni*nh+nh*no),[nh,no]);
bh = (x(ni*nh+nh*no+1:ni*nh+nh*no+nh))';
bo = (x(ni*nh+nh*no+nh+1:ni*nh+nh*no+nh+no))';
y_r = reshape(x(ni*nh+nh*no+nh+no+1:ni*nh+nh*no+nh+no+no*tn),[tn,no]);

bhmat = zeros(nh,tn);
for i = 1:size(bhmat,2)
    bhmat(:,i) = bh;
end

bomat = zeros(no,tn);
for i = 1:size(bomat,2)
    bomat(:,i) = bo;
end

y1 = Imat_t;
x1 = wh'*y1 + bhmat;
y2 = logsig(x1);
x2 = wo'*y2 + bomat;
yNN = purelin(x2);

ynn2_t_p = zeros(tn,no);

for i = 1:no
    ynn2_t_p(:,i) = yNN(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
end

ynn1_t_p = y_r;

Tsteam_out_st_NN = ynn1_t_p(:,1);
Tfg_out_st_NN = ynn1_t_p(:,2);

c1 = [ynn1_t_p(:,1);ynn1_t_p(:,2);ynn2_t_p(:,1);ynn2_t_p(:,2)];

Qfg_st_NN = mfg_st_t.*cpfg_st_t.*(Tfg_in_st_t - Tfg_out_st_NN);
Qsteam_st_NN = msteam_st_t.*cpsteam_st_t.*(Tsteam_out_st_NN - Tsteam_in_st_t);

c2 = (Qfg_st_NN - Qsteam_st_NN)./1000;

c = [c1;c2];

end