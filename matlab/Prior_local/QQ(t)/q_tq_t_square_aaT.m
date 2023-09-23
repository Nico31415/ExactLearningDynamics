function qtqtT=q_tq_t_square(t_max,tau, X, Y, X_tilde, Y_tilde,FolderName,plot_variable,tau_squares,A,R)
 % Compute the analitical solution of the q_q_t equation with square
 % input outputs from equation number 1
        
%Inputs 
sigma_xy =  (Y*X') ;
sigma_xy_tilde = (Y_tilde*X_tilde');
[U, S, V] = svd(sigma_xy);
[U_tilde, S_tilde, V_tilde] = svd(sigma_xy_tilde);
size_y=size(Y); 
identity = eye(length(S));    
%spetial 2*2  
AAT= A'*A;
a_1=AAT(1,1);
a_2= AAT(1,2);
a_3=AAT(2,2);

s1=S_tilde(1,1);
s2=S_tilde(2,2);
fact_1= (1/((a_1*a_3)-a_2^2));

 %V2
ainvainvT=inv(A'*A);

%V3
Ainv=fact_1*[a_3,-a_2; -a_2,a_1];

%Saving
qtqtT = zeros(t_max, 2.*size_y(1), 2.*size_y(1));

%Running
i=0;
t_vector=linspace(0, t_max , t_max);
    for t=t_vector
           i=i+1;
              
            tt =((t *tau_squares / tau));
            exp_s1_m = exp((-s1) .* tt);
            exp_s2_m = exp((-s2) .* tt);
            exp_s1_m_2 = exp((-2*s1) .* tt);
            exp_s2_m_2 = exp((-2*s2) .* tt);
       
            %Pre
            pre_1 = V_tilde;
            pre_2 = U_tilde  ;
            pre = [pre_1; pre_2];
            pre_save(i,:,:)=[pre_1; pre_2];
        
            %Center
            center_1 = (fact_1*(exp_s2_m* a_1*exp_s2_m))+(1/s2)-(exp_s2_m_2*(1/s2));
            center_2 = fact_1*(exp_s2_m* a_2* exp_s1_m);
            center_2_save(i,:,:)=[center_2];
            center_3 =  center_2 ;
            center_4= ( fact_1*(exp_s1_m* a_3*exp_s1_m))+((1/s1)*(1-exp_s1_m_2)) ;
            center_4_save(i,:,:)=[center_4];
            center=[center_1, center_2; center_3,center_4];
    
            fact_2= (center_4*center_1 )- (center_3*center_2);
            %fact_2_2=(fact_1*(exp_s2_m_2*exp_s1_m_2*(1-(a_1/s1)-(a_3/s2))+exp_s2_m_2*(a_1/s1)+exp_s1_m_2 *(a_3/s2)))+(1/(s1*s2))  ;    
            
            inv_tot=(1/fact_2)*center;
            [U_a,S_a,V_a]=svd(inv_tot);
            identity=eye(2,2);
            U_time(i,:,:)=norm((U_tilde-U_tilde*(U_a).^2),'fro')^2 ;
            V_time(i,:,:)=norm((V_tilde-V_tilde*(V_a).^2),'fro')^2;
            S_time(i,:,:)= U_time(i,:,:)+ V_time(i,:,:);
            post_save(i,:,:)= pre';
            post= pre';
            qtqtT(i,:,:) = (1/tau_squares)* (pre * inv_tot * post);

%V3  
           %tt =((t *tau_squares / tau));
         %  exp_s1_m_2 = exp((-s1).* 2.* tt);
           % exp_s2_m_2 = exp((-2*s2) .* tt);
          %  exp_s_m = diag(exp(diag(-S_tilde) .* tt));
          %  exp_2_s_m = diag(exp(diag(-S_tilde) .* 2 .* tt));
        
          

            %Pre
          %  pre_1 = V_tilde;
          %  pre_2 = U_tilde  ;
          %  pre = [pre_1; pre_2];
            
          %  pre_save(i,:,:)=[pre_1; pre_2];
        
       
          %  inv_1 =  exp_s_m*Ainv* exp_s_m ;
            %inv_1 =  exp_s_m*Ainv* exp_s_m ;
          %  inv1_save(i,:,:)=[inv_1];
          %  inv_2 = (+identity-exp_2_s_m) * sinv; % 
          %  inv2_save(i,:,:)=[inv_2];
          

            %inv_tot_= inv(inv_1 + inv_2);
            %post_save(i,:,:)= pre';
           % post= pre';
           % qtqtT(i,:,:) = (1/tau_squares) *(pre * inv_tot * post);

      %V2
            
           %tt =((t *tau_squares / tau));
            %exp_s_m = diag(exp(diag(-S_tilde) .* tt));
            %exp_2_s_m = diag(exp(diag(-S_tilde) .* 2 .* tt));
        
          

            %Pre
           % pre_1 = V_tilde;
            %pre_2 = U_tilde  ;
            %pre = [pre_1; pre_2];
            
           % pre_save(i,:,:)=[pre_1; pre_2];
        
            % Inv
        
            %inv_1 =  exp_s_m*ainvainvT* exp_s_m ;
            %inv_1 =  exp_s_m*ainvainvT* exp_s_m ;
            %inv1_save(i,:,:)=[inv_1];
            %inv_2 = (+identity-exp_2_s_m) * sinv; % 
            %inv2_save(i,:,:)=[inv_2];
          

            %inv_tot= inv(inv_1 + inv_2);
            %post_save(i,:,:)= pre';
            %post= pre';
            %qtqtT(i,:,:) = (1/tau_squares) *(pre * inv_tot * post);
  %V1
            %tt =((t *tau_squares / tau));
            %exp_s = diag(exp(diag(S_tilde) .* tt));
            %exp_2_s = diag(exp(diag(S_tilde) .* 2 .* tt));
        
        

            %Pre
            %pre_1 = V_tilde * exp_s *A' ;
            %pre_2 = U_tilde *exp_s *A' ;
            %pre = [pre_1; pre_2];
            
            %pre_save(i,:,:)=[pre_1; pre_2];

            % Inv
            %inv_1 = identity ;
            %inv1_save(i,:,:)=[inv_1];
            %inv_2 = A*(-identity+exp_2_s) * sinv*A'; % 
            %inv2_save(i,:,:)=[inv_2];
          

            %inv_tot= inv(inv_1 + inv_2);
            %post_save(i,:,:)= pre';
            %post= pre';
           % qtqtT(i,:,:) = (1/tau_squares) *(pre * inv_tot * post);
    end

    
    
end 


    