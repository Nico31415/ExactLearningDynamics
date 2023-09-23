function qtqtT=q_tq_t_square(t_max,tau, X, Y, X_tilde, Y_tilde,FolderName,plot_variable,tau_squares)
 % Compute the analitical solution of the q_q_t equation with square
 % input outputs from equation number 1
        
    %Inputs 
    si_x= size(X);
    sigma_xy =  (Y*X') ;
    sigma_xy_tilde = (Y_tilde*X_tilde');
    [U, S, V] = svd(sigma_xy);
    A_zero = S;  
    [U_tilde, S_tilde, V_tilde] = svd(sigma_xy_tilde);
    
    B = U'* U_tilde + V'*V_tilde;
    det(B)
    C = U'*U_tilde - V'*V_tilde;
    size_y=size(Y); 
    identity = eye(length(S));
    binv =inv(B);
    ainv = inv(A_zero);
    sinv = inv(S_tilde);
    %sinv(1,1) = 1
   
    %Saving
    qtqtT = zeros(t_max, 2.*size_y(1), 2.*size_y(1));
    %pre_save= zeros(t_max, 2.*size_y(1), size_y(1));
    %post_save= zeros(t_max, size_y(1), 2*size_y(1));
    inv_tot_save = zeros(t_max, size_y(1), size_y(1));
    
    %Running
    i=0;
    t_vector=linspace(0, t_max , t_max);
    for t=t_vector
           i=i+1;
            tt =((t *tau_squares / tau));
            exp_s = diag(exp(-diag(S_tilde) .* tt));
            exp_2_s = diag(exp(-diag(S_tilde) .* 2 .* tt));
            exp_s_binv = exp_s *binv;
            binvT_exp_s = binv' * exp_s;
            pre_binvT_exp_s(i,:,:)=[binvT_exp_s];

            %Pre
            pre_0 = exp_s * C' * binvT_exp_s;
            pre_0_save(i,:,:)=[pre_0];
            pre_1 = V_tilde * (identity- pre_0);
            pre_2 = U_tilde * (identity +pre_0);
            pre = [pre_1; pre_2];
            pre_save(i,:,:)=[pre_1; pre_2];

            %Inv
            inv_1 = 4 .* exp_s_binv * ainv * binvT_exp_s;
            inv1_save(i,:,:)=[inv_1];
            inv_2 = (identity-exp_2_s) * sinv; % 
            inv2_save(i,:,:)=[inv_2];
            inv_3 = exp_s_binv * C *(exp_2_s - identity) * sinv * C' * binvT_exp_s;
            inv3_save(i,:,:)=[inv_3];

            inv_tot_save(i,:,:) = (inv_1 + inv_2 - inv_3);
            inv_tot= inv(inv_1 + inv_2 - inv_3);
            
            %Post
            post_save(i,:,:)= pre';
            post= pre';
            qtqtT(i,:,:) = (1/tau_squares) *(pre * inv_tot * post);

            %Version number 2 based the equation from number 2
            % tt = t / tau;
            % exp_2_sm = diag(exp(-diag(S_tilde) .* 2 .* tt));
            % exp_2_s = diag(exp(diag(S_tilde) .* 2 .* tt));
            %exp_sm = diag(exp(-diag(S_tilde) .* tt));
            % exp_s = diag(exp(diag(S_tilde) .* tt));
            % exp_s_b = exp_s *B';
            % exp_s_C = exp_sm *C';
            %Pre
            % pre_1 = V_tilde * (exp_s_b -exp_s_C )*A_zero_sqr*R';
            % pre_2 = U_tilde * (exp_s_b+ exp_s_C)*A_zero_sqr*R' ;
            % pre = [pre_1; pre_2];
            % pre_save(i,:,:)=[pre_1; pre_2];
            % Inv
            % inv_1 = eye(8,8);
            %  inv1_save(i,:,:)=[inv_1];
            % inv_2 =(1/4)*R*A_zero_sqr*(B*(exp_2_s -identity)*sinv*B')*A_zero_sqr*R';
            %inv2_save(i,:,:)=[inv_2];
            %inv_3 =(1/4)*R*A_zero_sqr*(C*(exp_2_sm -identity)*sinv*C')*A_zero_sqr*R';
            % inv3_save(i,:,:)=[inv_3];
            %inv_tot_save(i,:,:) = inv(sqrt(inv_1 + inv_2 - inv_3));
            %inv_tot= inv(sqrt(inv_1 + inv_2 - inv_3));
            %Post
            %post_save(i,:,:)= pre';
            %post= pre';
            %qtqtT(i,:,:) = (pre * inv_tot * post);
    end
    line=linspace(3,t_max,100) ;
    line=round(line)-1;
    
% Plotting
    if plot_variable==1
       
        % Plot Q_TQ_T solution
        fig_num=807;
        name_vid=horzcat('./data/',FolderName,'/Video/QtQt_square_Q_Q_anlalitic.avi');
        qtqtT_plot_1(:,:,:)= qtqtT(1:t_max,1:2*size_y(1),1:2*size_y(1));
        name_plot='Q(t)Q(t)T Analytical QtQt_square';
        x_label='Input                    Output';
        y_label='Output                    Input';
        name_fig=horzcat('./data/',FolderName,'/Figures/QtQt_square_Q_Q_analytic.png');
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,qtqtT_plot_1,line,x_label,y_label)
        
     
    end 
end