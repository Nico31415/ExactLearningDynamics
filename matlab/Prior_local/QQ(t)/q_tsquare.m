function qtT=q_tsquare(t_max,tau, X, Y, X_tilde, Y_tilde,FolderName,R,plot_variable)
% Partial analitical solution of the Q_t matrix over time (Up to an
% othogonal transformation D)

    % Inputs
     
    samples_n=[10,2];
    sigma_xy =  (Y*X') ;
    sigma_xy_tilde = (Y_tilde*X_tilde')/(samples_n(2)-1);
    [U, S, V] = svd(sigma_xy);
    A_zero_sqr = sqrt(S);  
    [U_tilde, S_tilde, V_tilde] = svd(sigma_xy_tilde);
    B = U'* U_tilde + V'*V_tilde;
    C = U'*U_tilde - V'*V_tilde;
    size_y=size(Y);
    identity = eye(length(S));
    sinv = inv(S_tilde);

    % Saving
    QtQtTs = zeros(t_max, 2.*size_y(1), 2.*size_y(1));
    pre_save= zeros(t_max, 2.*size_y(1), size_y(1));
    post_save= zeros(t_max, size_y(1), 2*size_y(1));
    inv_tot_save = zeros(t_max, size_y(1), size_y(1));


    % Running
    i=0;
    t_vector=linspace(0, t_max , t_max);
    for t=t_vector
           i=i+1;
            tt = (t / tau);
            exp_2_sm = diag(exp(-diag(S_tilde) .* 2 .* tt));
            exp_2_s = diag(exp(diag(S_tilde) .* 2 .* tt));
            exp_sm = diag(exp(-diag(S_tilde) .* tt));
            exp_s = diag(exp(diag(S_tilde) .* tt));
            exp_s_b = exp_s *B';
            exp_s_C = exp_sm *C';

            %Pre
            pre_1 = V_tilde * (exp_s_b -exp_s_C )*A_zero_sqr*R';
            pre_2 = U_tilde * (exp_s_b+ exp_s_C)*A_zero_sqr*R' ;
            pre = [pre_1; pre_2];
            pre_save(i,:,:)=[pre_1; pre_2];

            % Inv
            inv_1 = eye(size_y(1),size_y(1));
            inv1_save(i,:,:)=[inv_1];
            inv_2 =(1/4)*R*A_zero_sqr*(B*(exp_2_s -identity)*sinv*B')*A_zero_sqr*R';
            inv2_save(i,:,:)=[inv_2];
            inv_3 =(1/4)*R*A_zero_sqr*(C*(exp_2_sm -identity)*sinv*C')*A_zero_sqr*R';
            inv3_save(i,:,:)=[inv_3];

            inv_tot_save(i,:,:) = (inv_1 + inv_2 - inv_3)^(-0.5);
            inv_tot= (inv_1 + inv_2 - inv_3)^(-0.5);

            %Post
            post_save(i,:,:)= pre';
            post= pre';
            qtT(i,:,:) =- 1/2*(pre * inv_tot );
            Q(:,:)=-qtT(i,:,:);
            QtQtTs(i,:,:)= Q*Q';

    end
    line=linspace(3,t_max,100) ;
    line=round(line)-1;

    % Plotting
    if plot_variable ==1
           % Plot the full q_t term
            fig_num=1001;
            name_vid=horzcat('./data/',FolderName,'/Video/qt_analytic_img.avi');
            qt_plot_1(:,:,:)= qtT(1:t_max,1:2*size_y(1),1:size_y(1));
            name_plot='Q(t)T Analytical';
            x_label='Hidden';
            y_label='Output                    Input';
            name_fig=horzcat('./data/',FolderName,'/Figures/qt_analytic_img.png');
            plot_rates_maps(fig_num,name_plot,name_vid,name_fig,qt_plot_1,line,x_label,y_label)
            
    end 
end


