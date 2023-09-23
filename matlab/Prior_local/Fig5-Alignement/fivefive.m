%% House Keeping
close all
clear all 

% Choose settings
% generate training examples and make sure to change the saving name
% accordingly
FolderName=['fivefive',datestr(now,'mm-dd-yyyy HH-MM')]; % add _dense _diag _nondiag %add _small _inter _large
% diagonal equal
%Y = [[5,0,0,0,0];
    %[0,5,0,0,0];
   % [0,0,5,0,0];
   % [0,0,0,5,0];
   % [0,0,0,0,5]];

% diagonal unequal
%Y = [[5,0,0,0,0];
    %[0,1,0,0,0];
   % [0,0,2,0,0];
   % [0,0,0,3,0];
   % [0,0,0,0,4]];
% dense
Y = [[5,6,3,0,1];
    [4,1,0,1,2];
    [3,0,2,4,0];
    [3,4,0,3,2];
    [2,0,1,3,4]];
% small
l=0.001;
% inter
%l=0.3
% large
%l=2


addpath('../PlottingRoutine/cbrewer','../PlottingRoutine','../QQ(t)')  
cd('..');
cd('./data/Fig5_Alignement/');
mkdir(FolderName, 'Video'); 
mkdir(FolderName, 'Figures'); 
cd('..');
cd('..');

learnrate = 0.01;
eta = learnrate;
tau=(1/eta);
N_x_t = 5;    %teacher input dimension
N_y_t = 5;  %teacher input dimension
RSA_timepoints = [140];
rsa_idx = 1;
target_precision_1 = 4.08e-12;
% generate training examples
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
[U_a,S,V_a]=svd(A'*A);
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
    
            %Check if we have that the constraint given in eq. is true.  
            Checkw1= W1*W1';
            Checkw2= W2'*W2;
            minus=Checkw1-Checkw2;
            if minus < 1e-11
                a='satisfied';
                fprintf(fileID, 'W1TW1=W2W2T');
            end 
            fclose(fileID);
            
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

line=linspace(2,t1,300) ;
line=round(line)-1;
szl=size(line);

%spetial 2*2  
sigma_xy_tilde = (Y*X');
[U, S, V] = svd(sigma_xy);
[U_tilde, S_tilde, V_tilde] = svd(sigma_xy_tilde);
AAT= A'*A;
a_1=AAT(1,1);
a_2= AAT(1,2);
a_3=AAT(2,2);
s1=S_tilde(1,1);
s2=S_tilde(2,2);
fact_1= (1/((a_1*a_3)-a_2^2));
t_ntk=1/s2;
t_clem=real((tau/(2*s1))*log((-2 *a_1 + s1)/(a_1* (-2 + a_1^2))));
t_ntk_fsmax= real((tau/s1)*log(s1/(a_1)^2));
t_bumps=(-tau/(4*s1))*(log(((a_1*a_3)-a_2^2)/(s1*(s1-a_1-a_3))));
t_andrew=(tau/s2)*log(s2/(a_1));



QtQtTst1=QtQtTs(1:t1,:,:);

figure(2000); 
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
name=horzcat(FolderName,'/Video/Full_projection_W1W2_AA.avi');

vid = VideoWriter(name);
open(vid);
v=1:1:t1;
for i=1:1:Ni
     for j=1:1:Ni
        hold on
        if i==j
            plot(v,W2W1_ij(:,i,j),'LineWidth',4,'Color','#BD2230');
        else
            plot(v,W2W1_ij(:,i,j),'LineWidth',4,'Color','#276FB4');
             hold on
         end
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
        frame = getframe(gcf);
        writeVideo(vid,frame);
     end
end
close(vid)

xline(t_andrew,'Color','#276FB4','LineStyle','--','LineWidth',2);
xline(t_ntk,'Color','#BD2230','LineStyle','--','LineWidth',2);
xline(t_bumps,'Color','k','LineStyle','--','LineWidth',2);
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Arial','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Arial','fontsize',14)
y_label = ylabel('Projection of W1W2 and AAT on to task1 Eigen-Basis '); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
set(0, 'DefaultFigureRenderer', 'painters');
name=horzcat(FolderName,'/Figures/Full_projection_W1W2_AA.svg');
filename_2000  = sprintf(name);
saveas(2000,filename_2000);


% Plot the Analitical and Simulation solution as well as their
% difference.¨¨
figure(2001); 
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
name=horzcat(FolderName,'/Figures/six.svg');
filename_2001  = sprintf(name);
saveas(2001,filename_2001);

figure(2002); 
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
name=horzcat(FolderName,'/Video/Full_projection_W1W2_AA.avi');

vid = VideoWriter(name);
open(vid);
v=1:1:t1;
for i=1:1:Ni
     for j=1:1:Ni
        hold on
        if i==j
            plot(v,Projection_W2W1_ij(:,i,j),'LineWidth',4,'Color','#BD2230');
        else
            plot(v,Projection_W2W1_ij(:,i,j),'LineWidth',4,'Color','#276FB4');
             hold on
         end
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
        frame = getframe(gcf);
        writeVideo(vid,frame);
     end
end
close(vid)
xline(t_andrew,'Color','#276FB4','LineStyle','--','LineWidth',2);
xline(t_ntk,'Color','#BD2230','LineStyle','--','LineWidth',2);
xline(t_bumps,'Color','k','LineStyle','--','LineWidth',2);
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Arial','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Arial','fontsize',14)
y_label = ylabel('Projection of W1W2 and AAT on to task1 Eigen-Basis '); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
set(0, 'DefaultFigureRenderer', 'painters');
name=horzcat(FolderName,'/Figures/Full_projection_W1W2_AA.svg');
filename_2002  = sprintf(name);
saveas(2002,filename_2002);


figure(2003); 
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
name=horzcat(FolderName,'/Video/NTK.avi');
vid = VideoWriter(name);
open(vid);
v=1:1:t1;
for i=1:1:Ni
     for j=1:1:Ni
        hold on
        if i==j
            plot(v,NTK_ij(:,i,j),'LineWidth',4,'Color','#BD2230');
              else
            plot(v,NTK_ij(:,i,j),'LineWidth',4,'Color','#276FB4');
            hold on
             end
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
        frame = getframe(gcf);
        writeVideo(vid,frame);
     end
end
legend('Diagonat1=l','Off-Diagonal')
close(vid)
xline(t_andrew,'Color','#276FB4','LineStyle','--','LineWidth',2);
xline(t_ntk,'Color','#BD2230','LineStyle','--','LineWidth',2);
xline(t_bumps,'Color','k','LineStyle','--','LineWidth',2);
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Arial','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Arial','fontsize',14)
y_label = ylabel('NTK'); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); 
set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
set(0, 'DefaultFigureRenderer', 'painters');
name=horzcat(FolderName,'/Figures/NTK.svg');
filename_2003  = sprintf(name);
saveas(2003,filename_2003);


figure(2004); 
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
name=horzcat(FolderName,'/Video/projection_NTK.avi');
vid = VideoWriter(name);
open(vid);
v=1:1:t1;
for i=1:1:Ni
     for j=1:1:Ni
        hold on
        if i==j
            plot(v,Projection_WNTKP_ij(:,i,j),'LineWidth',4,'Color','#BD2230');
               else
            plot(v,Projection_WNTKP_ij(:,i,j),'LineWidth',4,'Color','#276FB4');
            hold on
                end
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
       
        frame = getframe(gcf);
        writeVideo(vid,frame);
     end
end
legend('Diagonat1=l','Off-Diagonal')
close(vid)
xline(t_ntk_fsmax,'Color','#276FB4','LineStyle','--','LineWidth',2);
xline(t_andrew,'Color','#276FB4','LineStyle','--','LineWidth',2);
xline(t_ntk,'Color','#BD2230','LineStyle','--','LineWidth',2);
xline(t_bumps,'Color','k','LineStyle','--','LineWidth',2);
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Arial','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Arial','fontsize',14)
y_label = ylabel('NTK Proj'); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); 
set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
set(0, 'DefaultFigureRenderer', 'painters');
name=horzcat(FolderName,'/Figures/NTK_Proj.svg');
filename_2004  = sprintf(name);
saveas(2004,filename_2004);



tau_squares=1;
plot_variable_qqt=0;
Qqt_square=q_tq_t_square(t1,tau, X, Y_a, X, Y,FolderName,plot_variable_qqt,tau_squares);

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
                plot(v,Qqt_square(:,i,j),'LineWidth',2);
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
                plot(v,QtQtTst1(:,i,j)-Qqt_square(:,i,j),'LineWidth',2);
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
figure(509); 
Projection_W_1_ij=zeros(t1,Ni,Ni);
for t=1:1:t1
     W_2_t(:,:)=W_2_overtime(t,:,:);
     Projection_W_2=U_task_1_w2'*W_2_t*W_2_t'*Vv_task_1_w2;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_W_2_ij(t,i,j)=Projection_W_2(i,j);

          end
     end 
end 
name=horzcat(FolderName,'/Video/Full_projection_W2W2_SVD_task1_line.avi');
vid = VideoWriter(name);
open(vid);
v=1:1:t1;
for i=1:1:Ni
     for j=1:1:Ni
        hold on
        if i==j
        plot(v,Projection_W_2_ij(:,i,j),'LineWidth',2,'Color','#BD2230');
        else
        hold on
        plot(v,Projection_W_2_ij(:,i,j),'LineWidth',2,'Color','#276FB4');
        end
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
        frame = getframe(gcf);
        writeVideo(vid,frame);
        pause(0.1);
     end
end
legend('Diagonal','Off-Diagonal')
close(vid)
xline(t1,'Color','k','LineStyle','--','LineWidth',2);
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
y_label = ylabel('Projection of W2 on to task1 Eigen-Basis '); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
set(0, 'DefaultFigureRenderer', 'painters');
name=horzcat(FolderName,'/Figures/Full_projection_W2W2_SVD_task1_line.svg');
filename_509  = sprintf(name);
saveas(509,filename_509);


% Plot Projection of W1 onto SVD basis of W1 at the end of task 1
% Off-Diagonal terms
% During task 2

figure(510);
name=horzcat(FolderName,'/Video/Full_projection_W2W2_SVD_task1_off_diagonal.avi');
vid = VideoWriter(name);
open(vid);
v=1:1:t1;
for i=1:1:Ni
     for j=1:1:Ni
          if i ~=j  
        hold on
        plot(v,Projection_W_2_ij(:,i,j),'LineWidth',2,'Color','#276FB4');
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
        frame = getframe(gcf);
        writeVideo(vid,frame);
        pause(0.1);
          end
     end
end
close(vid)
xline(t1,'Color','k','LineStyle','--','LineWidth',2);
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
y_label = ylabel('Off-diagonal Projection of W2 on to task1 EigenBasis '); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); 
set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
name=horzcat(FolderName,'/Figures/Full_projection_W2_SVD_task1_off_diagonal.svg');
filename_510  = sprintf(name);
saveas(510,filename_510);

% Plot Projection of W1 onto SVD basis of W1 at the end of task 1
% During task 2
figure(511); 
Projection_W_1_ij=zeros(t1,Ni,Ni);
for t=1:1:t1
     W_1_t(:,:)=W_1_overtime(t,:,:);
     Projection_W_1=U_task_1_w1'*W_1_t'*W_1_t*Vv_task_1_w1;
     for i=1:1:Ni
          for j=1:1:Ni
                 Projection_W_1_ij(t,i,j)=Projection_W_1(i,j);

          end
     end 
end 
name=horzcat(FolderName,'/Video/Full_projection_W1W1_SVD_task1_line.avi');
vid = VideoWriter(name);
open(vid);
v=1:1:t1;
for i=1:1:Ni
     for j=1:1:Ni
        hold on
        if i==j
        plot(v,Projection_W_1_ij(:,i,j),'LineWidth',2,'Color','#BD2230');
        else
        hold on
        plot(v,Projection_W_1_ij(:,i,j),'LineWidth',2,'Color','#276FB4');
        end
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
        frame = getframe(gcf);
        writeVideo(vid,frame);
        pause(0.1);
     end
end
legend('Diagonal','Off-Diagonal')
close(vid)
xline(t1,'Color','k','LineStyle','--','LineWidth',2);
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
y_label = ylabel('Projection of W1 on to task1 Eigen-Basis '); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
set(0, 'DefaultFigureRenderer', 'painters');
name=horzcat(FolderName,'/Figures/Full_projection_W1W1_SVD_task1_line.svg');
filename_511  = sprintf(name);
saveas(511,filename_511);


% Plot Projection of W1 onto SVD basis of W1 at the end of task 1
% Off-Diagonal terms
% During task 2

figure(512);
name=horzcat(FolderName,'/Video/Full_projection_W1W1_SVD_task1_off_diagonal.avi');
vid = VideoWriter(name);
open(vid);
v=1:1:t1;
for i=1:1:Ni
     for j=1:1:Ni
          if i ~=j  
        hold on
        plot(v,Projection_W_1_ij(:,i,j),'LineWidth',2,'Color','#276FB4');
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
        frame = getframe(gcf);
        writeVideo(vid,frame);
        pause(0.1);
          end
     end
end
close(vid)
xline(t1,'Color','k','LineStyle','--','LineWidth',2);
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
y_label = ylabel('Off-diagonal Projection of W1 on to task1 EigenBasis '); 
set(y_label, 'FontSize', 20,'FontName' , 'Arial')
x_label = xlabel('Epochs'); 
set(x_label, 'FontSize', 20,'FontName' , 'Arial') 
name=horzcat(FolderName,'/Figures/Full_projection_W1_SVD_task1_off_diagonal.svg');
filename_512  = sprintf(name);
saveas(512,filename_512);           

 QtQtTst1=QtQtTs(1:t1,:,:);

figure(2000); 
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

Qqt_square_W2W1=Qqt_square(:,6:10,1:5);


QtQtTst1=QtQtTs(1:t1,:,:);

figure(2000); 
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

