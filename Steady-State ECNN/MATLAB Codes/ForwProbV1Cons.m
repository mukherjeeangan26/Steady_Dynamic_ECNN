function c = ForwProbV1Cons(x,ynn_v,tv,no,msteam_st_v,mfg_st_v,Tsteam_in_st_v,Tfg_in_st_v,cpfg_st_v,cpsteam_st_v)

ynn_v_r = x(1:tv*no);
ynn_v_r = reshape(ynn_v_r,size(ynn_v));

ynn1_v_p = ynn_v_r';

Tsteam_out_st_NN_r = ynn1_v_p(:,1);
Tfg_out_st_NN_r = ynn1_v_p(:,2);

c1 = [ynn1_v_p(:,1);ynn1_v_p(:,2)];

Qfg_st_v = mfg_st_v.*cpfg_st_v.*(Tfg_in_st_v - Tfg_out_st_NN_r);
Qsteam_st_v = msteam_st_v.*cpsteam_st_v.*(Tsteam_out_st_NN_r - Tsteam_in_st_v);

c2 = (Qfg_st_v - Qsteam_st_v)./1000;

c = [c1;c2];

end