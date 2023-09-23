%% House Keeping
close all
clear all 

% Choose settings
% generate training examples and make sure to change the saving name
% accordingly
FolderName=['AA',datestr(now,'mm-dd-yyyy HH-MM')]; % add _dense _diag _nondiag %add _small _inter _large
% diagonal equal
Y = [[5, 0];[0, 5]];
% diagonal unequal
% Y = [[5, 0];[0, 1]];
% dense
%Y = [[4, 1];[3, 5]];

% small
%l=0.0000001;
% inter
l=0.1
% large
%l=5

addpath('../PlottingRoutine/cbrewer','../PlottingRoutine','../ntk','../QQ(t)','../Fig5-Alignement')  
cd('..');
cd('./data/Fig5_Alignement/');
mkdir(FolderName, 'Video'); 
mkdir(FolderName, 'Figures'); 
cd('..');
cd('..');


learnrate = 0.001;
eta = learnrate;
tau=(1/eta);
N_x_t = 2;    %teacher input dimension
N_y_t = 2;  %teacher input dimension
RSA_timepoints = [140];
rsa_idx = 1;
target_precision_1 = 4.08e-5;
X= eye(N_x_t,N_x_t);
Ni = size(X,1);
Nh = Ni;
No = size(Y,1);
sigma_xy =  Y*X';
[U,S,V]=svd(sigma_xy);
r(:,:)= normrnd(0,0.000001,Nh,Nh);
[R,R_2 ,R_3]=svd(r);
A(:,:)= normrnd(0,l,Nh,Nh);
W1 =  R *A * V';
W2 =  U * A' * R';

Y_a=W2*W1*X;
inputs =  X;
targets = Y;
training_steps_max = 255000000;
x=0;

% Saving
cd(horzcat('./data/Fig5_Alignement/',FolderName,'/'));
name=horzcat('Inputs.txt');
fileID = fopen(name,'w');
formatSpec = ' learnrate %4.10f\n';
fprintf(fileID ,formatSpec, learnrate);
formatSpec = ' N_x_t %4.10f\n';
fprintf(fileID ,formatSpec, N_x_t);
formatSpec = ' N_y_t %4.10f\n';
fprintf(fileID ,formatSpec, N_y_t);
formatSpec = ' target_precision_1 %4.20f\n';
fprintf(fileID ,formatSpec, target_precision_1);
formatSpec = ' A %4.10f\n';
fprintf(fileID ,formatSpec, l);


path_y_t=horzcat('A.csv');
csvwrite(path_y_t,A);
path_y_t_new=horzcat('y_t_new.csv');
csvwrite(path_y_t_new,Y);
cd('..');


%% Run
for training_step=1:1:training_steps_max
    % Compute Q_TQ_T
    Qt = [W1'; W2];
    Q(training_step,:,:)= Qt;
    QtQtTs(training_step,:,:)=(Qt*Qt');
    QQT(:,:)=QtQtTs(training_step,:,:);
    [Uqq,Sqq,Vqq]= svd(QQT);
    Uq(training_step,:,:)=Uqq;
    Vq(training_step,:,:)=Vqq;
    Sq(training_step,:)= diag(Sqq);
    h1 = W1 *inputs;
    Y_hat = W2 * h1;
    x_t_input_e = eye(N_x_t); 
    h_1 = W1* x_t_input_e;
    
    if any(training_step==RSA_timepoints)
       RSA_mat{rsa_idx} = h_1'*h_1; 
       rsa_idx = rsa_idx + 1;
    end
    
    %Compute loss task 1
    cError = Y-Y_hat; %train error       
    Cost = sum(sum(cError.^2))/Ni;
    Cost_vector(training_step) = Cost;
    losses(training_step)=  mean(mean(0.5 * (Y- Y_hat)*(Y- Y_hat)'));
      
   
    % Gradient step
    delta = (Y_hat - targets);
    delta_W2 = -eta .* delta * h1';
    delta = W2' * delta ;
    delta_W1 = -eta .* delta * inputs';
    delta_W_1_overtime(training_step,:,:)=delta_W1;
    delta_W_2_overtime(training_step,:,:)= delta_W2;
    
    % Update weights
    W1 = W1 + delta_W1;
    W2 =  W2+ delta_W2;
    W_1_overtime(training_step,:,:)=W1;
    W_2_overtime(training_step,:,:)=W2;
    
    % SVD of task 1
    [U_w1, S_w1, V_w1] = svd(W2*W1); 
    Uu_w1(training_step,:,:)=U_w1;
    Vv_w1(training_step,:,:)=V_w1;
    Ss_w1(training_step,:)= diag(S_w1);
    
    % SVD of task 2
    [U_w2, S_w2, V_w2] = svd(W2);  
    Uu_w2(training_step,:,:)=U_w2;
    Vv_w2(training_step,:,:)=V_w2;
    Ss_w2(training_step,:)= diag(S_w2);
  
    
   % Switch from task 1 to task 2
    if losses(training_step) < target_precision_1
        x=x+1;
        if x==1
            Prioritize_variable=1; %Prioritize during task 2
            t1 = training_step;
    
       
            %Save SVD representation of W1 and W2 at the end of the first
            %task for Projection
            [U_1, S_1, V_1] = svd(W1);  
            U_task_1_w1=U_1;
            Vv_task_1_w1=V_1;
            Ss_task_1_w1= diag(S_1);
            [U_2, S_2, V_2] = svd(W2);  
            U_task_1_w2=U_2;
            Vv_task_1_w2=V_2;
            Ss_task_1_w2=diag(S_2);
             break 
        end
    end
    % End of task 2
   
end
tau_squares=1;
plot_variable_qqt=1;
Qqtaa=q_tq_t_square_aaT(t1,tau, X, Y_a, X, Y,FolderName,plot_variable_qqt,tau_squares,A,R);
QtQtTst1=QtQtTs(1:t1,:,:);

% Plot the Analitical and Simulation solution as well as their
% difference.¨¨
figure(2001); 
subplot(3,1,1) 
v=1:1:t1;
 for i=1:1:N_x_t*2
     for j=1:1:N_x_t*2
                hold on
                plot(v,QtQtTst1(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('QQT Sim'); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Arial')
subplot(3,1,2) 
v=1:1:t1;
 for i=1:1:N_x_t*2
     for j=1:1:N_x_t*2
                hold on
                plot(v,Qqtaa(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('QQT Ana'); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Arial')
hold on
subplot(3,1,3) 
v=1:1:t1;
 for i=1:1:N_x_t*2
     for j=1:1:N_x_t*2
                hold on
                plot(v,QtQtTst1(:,i,j)-Qqtaa(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Difference'); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Arial')
name=horzcat(FolderName,'/Figures/QQt_Ana_aa.png');
filename_2001  = sprintf(name);
saveas(2001,filename_2001);

sigma_xy_tilde = (Y*X');
[U, S, V] = svd(sigma_xy);
[U_tilde, S_tilde, V_tilde] = svd(sigma_xy_tilde);
%spetial 2*2  
AAT= A'*A;
a_1=AAT(1,1);
a_2= AAT(1,2);
a_3=AAT(2,2);
s1=S_tilde(1,1);
s2=S_tilde(2,2);
fact_1= (1/((a_1*a_3)-a_2^2));

        
t_2=(-tau/(2*s1))*(log(((2*a_1^2)-a_1*s1)/((s1-2*a_1)*(s1-a_1))));
t_3=(tau/(2*s1))*(log(((s1^2)-a_1*s1)/((a_1)*(a_1))));
t_4=(tau/(2*s1))*(log(((s1))/((2*a_1))));
t_7=(tau/(2*s1))*(log(((s1))/((a_1))));
t_5=(tau/(2*s1))*(log(((s1/a_1)-2)));
t_6=(tau/(2*s1))*(log(((s1/a_1)-2)));
t_8=(tau/(2*s1))*(log(-(a_1+a_3-s1)/(a_3-a_1)));
t_9=(tau/(2*s1))*(log(((s1/(2*a_1))-1)));

t_ntk=1/s2;
t_ntk_fsmax_clem=real((tau/(2*s1))*log((-2 *a_1 + s1)/(a_1* (-2 + a_1^2))));
t_ntk_fsmax=(-tau/(2*s1))*(log(((-s1/(a_1^2))+1)));
t_ntk_fsmax= real((tau/(s2*2))*log(s2/(a_1)^2));
t_bumps=(-tau/(4*s1))*(log(((a_1*a_3)-a_2^2)/(s1*(s1-a_1-a_3))));
t_andrew=(tau/s2)*log(s2/(a_1));


QQ_W2W1_overtime=Qqtaa(:,3:4,1:2);
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     QQ_W2W1(:,:)=QQ_W2W1_overtime(t,:,:);
     Projection_QQw2w1 =U'*QQ_W2W1*V;
     Projection_W2W1= U'*W_2_t*W_1_t*V;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_W2W1_ij(t,i,j)= Projection_W2W1(i,j);
                 Projection_QQW2W1_ij(t,i,j)=Projection_QQw2w1(i,j);
          end
     end 
end 


figure(2003); 
QQ_W1W1_overtime=Qqtaa(:,1:2,1:2);
QQ_W2W2_overtime=Qqtaa(:,3:4,3:4);
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     QQ_W1W1(:,:) =QQ_W1W1_overtime(t,:,:);
     QQ_W2W2(:,:)= QQ_W2W2_overtime(t,:,:);
     Q_NTK=QQ_W2W2+QQ_W1W1;
     W_NTK= W_1_t'*W_1_t+ W_2_t*W_2_t';
     for i=1:1:Ni
          for j=1:1:Ni
                NTK_ij(t,i,j)=W_NTK(i,j);
                QNTK_ij(t,i,j)=Q_NTK(i,j);
          end
     end 
end 



figure(2004); 
QQ_W1W1_overtime=Qqtaa(:,1:2,1:2);
QQ_W2W2_overtime=Qqtaa(:,3:4,3:4);
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     QQ_W1W1_proj(:,:) =QQ_W1W1_overtime(t,:,:);
     QQ_W2W2_proj(:,:)=QQ_W2W2_overtime(t,:,:);
     Projection_NTK_proj=  U_task_1_w2'*QQ_W2W2_proj*U_task_1_w2 +  Vv_task_1_w1'*QQ_W1W1_proj*Vv_task_1_w1;
     Projection_W_NTK_proj=U_task_1_w2'*W_1_t'*W_1_t*U_task_1_w2 +Vv_task_1_w1'*W_2_t*W_2_t'*Vv_task_1_w1;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_WNTKP_ij(t,i,j)=Projection_W_NTK_proj(i,j);
                 Projection_NTKP_ij(t,i,j)=Projection_NTK_proj(i,j);
          end
     end 
end 
QQ_W2W1_overtime=Qqtaa(:,3:4,1:2);



for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     QQ_W2W1(:,:)=QQ_W2W1_overtime(t,:,:);
     Projection_QQW2W1=U'*QQ_W2W1*V;
     Projection_W2W1= U'*W_2_t*W_1_t*V;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_W_1_ij(t,i,j)=Projection_W2W1(i,j);
                 Projection_QQ_ij(t,i,j)=Projection_QQW2W1(i,j);
          end
     end 
end 


tau_squares=1;
plot_variable_qqt=0;
Qqtsquare=q_tq_t_square(t1,tau, X, Y_a, X, Y,FolderName,plot_variable_qqt,tau_squares);

% Plot the Analitical and Simulation solution as well as their difference.
figure(2000); 
subplot(3,1,1) 
v=1:1:t1;
 for i=1:1:N_x_t*2
     for j=1:1:N_x_t*2
                hold on
                plot(v,QtQtTst1(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('QQT Sim'); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Arial')
subplot(3,1,2) 
v=1:1:t1;
 for i=1:1:N_x_t*2
     for j=1:1:N_x_t*2
                hold on
                plot(v,Qqtsquare(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('QQT Ana'); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); 
set(x_label, 'FontSize', 20,'FontName' , 'Arial')
hold on
subplot(3,1,3) 
v=1:1:t1;
 for i=1:1:N_x_t*2
     for j=1:1:N_x_t*2
                hold on
                plot(v,QtQtTst1(:,i,j)-Qqtsquare(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Difference'); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Arial')
name=horzcat(FolderName,'/Figures/QQt_Ana.png');
filename_2000  = sprintf(name);
saveas(2000,filename_2000);

 

% Plot Projection of W1 onto SVD basis of W1 at the end of task 1
% During task 2


for t=1:1:t1
     W_2_t(:,:)=W_2_overtime(t,:,:);
     Projection_W_2=U_task_1_w2'*W_2_t*W_2_t'*Vv_task_1_w2;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_W_2_ij(t,i,j)=Projection_W_2(i,j);

          end
     end 
end 

% Plot Projection of W1 onto SVD basis of W1 at the end of task 1


for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     Projection_W_1=U_task_1_w1'*W_1_t'*W_1_t*Vv_task_1_w1;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_W_1_ij(t,i,j)=Projection_W_1(i,j);

          end
     end 
end 


QtQtTst1=QtQtTs(1:t1,:,:);

    
v=1:1:t1;
l=linspace(100,t1-1,40);
l=round(l);
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     W2W1= W_2_t*W_1_t;
     for i=1:1:Ni
          for j=1:1:Ni
                 W2W1_ij(t,i,j)=W2W1(i,j);
          end
     end 
end 

           

     
v=1:1:t1;
l=linspace(100,t1-1,40);
l=round(l);
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     Projection_W2W1= U'*W_2_t*W_1_t*V;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_W2W1_ij(t,i,j)=Projection_W2W1(i,j);
          end
     end 
end 
           
           

          
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     W_NTK= (N_y_t *(X'*W_1_t'*W_1_t*X))+ ((norm(W_2_t,"fro")^2).*X'*X);
     for i=1:1:Ni
          for j=1:1:Ni
                 NTK_ij(t,i,j)=W_NTK(i,j);
          end
     end 
end 
            


      
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     Projection_W_NTK_proj=U_task_1_w2'*W_1_t'*W_1_t*U_task_1_w2 +Vv_task_1_w1'*W_2_t*W_2_t'*Vv_task_1_w1;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_WNTKP_ij(t,i,j)=Projection_W_NTK_proj(i,j);
          end
     end 
end 

            
 
tau_squares=1;
plot_variable_qqt=0;
Qqt_square=q_tq_t_square(t1,tau, X, Y_a, X, Y,FolderName,plot_variable_qqt,tau_squares);

Qqt_square_W2W1=Qqt_square(:,3:4,1:2);


QtQtTst1=QtQtTs(1:t1,:,:);

           
v=1:1:t1;
l=linspace(100,t1-1,40);
l=round(l);
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     W_2_t(:,:)=W_2_overtime(t,:,:);
     W2W1= W_2_t*W_1_t;
     for i=1:1:Ni
          for j=1:1:Ni
                 W2W1_ij(t,i,j)=W2W1(i,j);
                 
          end
     end 
end 
                     
cd(horzcat(FolderName,'/'));
save('run')
cd('..');
cd('..');



