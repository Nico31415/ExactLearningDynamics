function qtqtT=q_tq_t_notsquare(t_max,tau, X, Y, X_tilde, Y_tilde,FolderName,W1,W2,plot_variable)
  % Compute the analitical solution of the q_q_t equation with non square
  % input outputs from the equation number (Lukas derivation)
        
 
    Q0 = [W1'; W2];
    samples_n=size(X_tilde);
    sigma_xy_tilde =(Y_tilde*X_tilde');
     
     [U, S, V] = svd(Y*X');
    [U_tilde, S_tilde, V_tilde] = svd(sigma_xy_tilde);
    V_tilde= V_tilde;
    [mo,mi]=size(sigma_xy_tilde);
    Ne=abs(mo-mi);
    small_zero=zeros(mi,mi);
    large_zero=zeros(mo,mo);
    
 
    U_hat=U_tilde(:,mi+1:mo);
    OE= (1 ./ sqrt(2)) .*[zeros(mi,Ne);U_hat]; 

    if mi < mo
            U_hat=U_tilde(:,mi+1:mo);
            OE= (1 ./ sqrt(2)) .*[zeros(mi,Ne);U_hat]; 
            U_tilde = U_tilde(:, 1:mi);
            S_tilde=S_tilde(1:mi,1:mi);
            LD=[S_tilde,small_zero; small_zero,-S_tilde];
    end
    
    if mi > mo
            V_hat = V_tilde(:, mo+1:end);
            V_tilde = V_tilde(:, 1:mo);
            OE= (1 ./ sqrt(2)) .*[V_hat; zeros(mo,Ne)]; 
            S_tilde=S_tilde(1:mo,1:mo);
            LD=[S_tilde,large_zero; large_zero,-S_tilde];
    end
    
    
    OD= (1 ./ sqrt(2)) .*[V_tilde,V_tilde;U_tilde,-U_tilde];
   
    % Run
    F=[small_zero,sigma_xy_tilde';sigma_xy_tilde,large_zero];
    LE=eye(Ne,Ne);
    Q0_size=size(Q0);
    i=0;
    t_vector=linspace(0, t_max , t_max);
    for t=t_vector
        i=i+1;
        tt = (t/tau);
        O_ = OD * expm(LD .* tt) * OD';
        iv_LD=inv(LD);
        L_ = OD * iv_LD * OD';
        MMT = 2 .* OE * OE';

        left = (O_ + MMT)  * Q0;
        center_center = O_ * L_ * O_ - L_ + 2 .* tt .* MMT;
        center_1= eye(Q0_size(2));
        center =inv(center_1 + 0.5 .* Q0' * center_center * Q0);
        right = Q0' * (O_ + MMT);

        qtqtT(i,:,:)= left * center * right;
       
    end
    

  