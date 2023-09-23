% Function to test the analitical solutions of the q_tq_t with square-
% random input-output dimentions 
%% HouseKeeping
close all
clear all
cd('..');

addpath('../PlottingRoutine/cbrewer','../PlottingRoutine','../QQ(t)')  
addpath('./THETNS','./PlottingRoutine/cbrewer','./PlottingRoutine')  
FolderName=['Test',datestr(now,'mm-dd-yyyy HH-MM')];
cd('./data/Test-Q/');
mkdir(FolderName, 'Video'); 
mkdir(FolderName, 'Figures'); 
% Figure 8..
plot_variable_qqt=0;
% Figure 9..
plot_variable_new=0;
% Figure 1...
plot_variable_q=0;
% Figure 11..
plot_variable_q_u=0;
% Figure 11..
% Could add more for this 

%% Inputs
rng(4);
dim = 2;
sigma = 1;
samples_n =50;
tau_squares=1;

% Task 1
X =  normrnd(0, sigma, [dim, samples_n]);
X = X - mean(X,2);
cov_1 = cov(X',1);
[eigvecs,eigvals] = eig(cov_1);
X =(diag((diag(sqrt(tau_squares)))))*(diag((1 ./ diag(sqrt(samples_n)))))*(diag((1 ./ diag(sqrt(eigvals)))))*eigvecs'*X ;
W1_target = normrnd(0, sigma, [dim, dim]);
W2_target = normrnd(0, sigma, [dim,dim]);
Y = W2_target * W1_target * X;

% Task 1
X_tilde = normrnd(0, sigma, [dim, samples_n]);
X_tilde= X_tilde - mean(X_tilde,2);
cov_2 = cov(X_tilde',1);
[ eigvecs,eigvals] = eig(cov_2);
X_tilde = (diag((diag(sqrt(tau_squares)))))* (diag((1 ./ diag(sqrt(samples_n)))))*(diag((1 ./ diag(sqrt(eigvals)))))*eigvecs'*(X_tilde);
W1_tilde_target = normrnd(0, sigma, [dim, dim]);
W2_tilde_target = normrnd(0, sigma, [dim, dim]);
Y_tilde = W2_tilde_target * W1_tilde_target * X_tilde;

% Weight init


W1 = 2 * eye(size(W1_target));
W2 = eye(size(W2_target));
% W1 = normrnd(0, 0.00001, size(W1_target));
% W2 = normrnd(0, 0.0011, size(W2_target));

inputs =  X;
targets = Y;
training_steps_max = 2550000;
points=[];
H_vector=0; 
x=0;
% Train
eta = 0.001;
tau=1/eta;
inputs = X;
targets = Y;
training_steps_max = 25000000;
losses = [];
losses_tilde = [];
target_precision_1 = 1e-7;
target_precision_2 = 1e-7;
QtQtTs=zeros(training_steps_max,2*dim,2*dim);
Ss=zeros(training_steps_max,dim);

%% Run
for training_step=1:1:training_steps_max
    
    % Compute Q_TQ_T
    Qt = [W1'; W2];
    Q(training_step,:,:)= Qt;
    QtQtTs(training_step,:,:)=(Qt*Qt');
    
    % Compute loss
    h1 = W1 *inputs;
    Y_hat = W2 * h1;
    losses= [losses trace((0.5 * (Y_hat - Y)*(Y_hat - Y)'))];
    losses_tilde=[losses_tilde trace(((0.5 * (Y_hat - Y_tilde)*(Y_hat - Y_tilde)')))];
    
    % Gradient Update
    delta = (Y_hat - targets);
    delta_W2 = -eta .* delta * h1';
    delta = W2' * delta ;
    delta_W1 = -eta .* delta * inputs';
    
    W1 = W1 + delta_W1;
    W2 =  W2+ delta_W2;
    W_1_overtime(training_step,:,:)=W1;
    W_2_overtime(training_step,:,:)=W2;
    
     % SVD
    [U, S, V] = svd(W1);   
    Ss(training_step,:)= diag(S);
    
    % Switch from task 1 to task 2
    if losses(end) < target_precision_1
        t1 = training_step;
        QtQtTs_1 = QtQtTs(1:1:t1,:,:);
       
        inputs = X_tilde;
        targets = Y_tilde;
        [U,S,V]= svd(W2*W1);
        
    end
    % End of task 2
    if losses_tilde(end) < target_precision_2
        t2 = training_step;
        
        break
    end   
end

t3 =t2-t1;

%% Saving the Work space

cd(horzcat(FolderName,'/'));
save('run')
cd('..')



%% Plot Loss 

figure(1);  
plot(losses(1:1:t2),'LineWidth',2,'Color','b');
hold on 
plot(losses_tilde(1:1:t2),'LineWidth',2,'Color','r');
hold on 
ax = gca;
ax.XAxis.Exponent = 0;
tick2 =  num2str(get(gca,'YTick')','%g');
tick1 =  num2str(get(gca,'XTick')','%g');
set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
xline(t1,'Color','k','LineStyle','--','LineWidth',2);
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
y_label = ylabel('Loss'); %or h=get(gca,'xlabel')
set(y_label,'FontSize', 20,'FontName' , 'Times New Roman') 
legend('Task1','Task2')
name=horzcat(FolderName,'/Figures/loss.svg');
filename_1  = sprintf(name);
saveas(1,filename_1)
        

%% Analitical Solution for QTQT with equal input-output dimentions (See equation number.)
% My derivations
W_1(:,:)=W_1_overtime(t1,:,:);
W_2(:,:)=W_2_overtime(t1,:,:);
[R,S,V]=svd(W_1);


%% Analitical Solution for QTQT (See equation number.)
% Andrew's derivations
Qqt=q_tq_t_square(t3,tau, X, Y, X_tilde, Y_tilde,FolderName,plot_variable_qqt,tau_squares);
QtQtTs_2=QtQtTs(t1+1:t2,:,:);

% Plot the Analitical and Simulation solution as well as their difference.
figure(2000); 
subplot(3,1,1) 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim*2
                hold on
                plot(v,QtQtTs_2(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('QQ_T Simulation'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
name=horzcat(FolderName,'/Figures/QQt.svg');
subplot(3,1,2) 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim*2
                hold on
                plot(v,Qqt(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('QQ_T Analytical'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
name=horzcat(FolderName,'/Figures/QQt.svg');
hold on
subplot(3,1,3) 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim*2
                hold on
                plot(v,QtQtTs_2(:,i,j)-Qqt(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Difference'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
name=horzcat(FolderName,'/Figures/QQt.svg');
filename_2000  = sprintf(name);
saveas(2000,filename_2000);



%% Compute full Qt Analytical Solution without Orthogonal transformation (See Equation .)

W_1(:,:)=W_1_overtime(t1,:,:);
W_2(:,:)=W_2_overtime(t1,:,:);
[R,S,V]=svd(W_1);
line=linspace(3,t3,500) ;
line=round(line)-1;
szl=size(line);

Qt=q_tsquare(t3,tau, X, Y, X_tilde, Y_tilde,FolderName,R,plot_variable_q);

% Plot the Analitical and Simulation solution of Q_t as well as their difference.
% (Up to a orthogonal transformation)
Q_2=Q(t1+1:1:t2,:,:);
figure(4000); 
subplot(3,1,1) 
title('Without Orthogonal transform for Ana Solution') 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim
                hold on
                plot(v,Q_2(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Q Simulation'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
subplot(3,1,2) 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim
                hold on
                plot(v,Qt(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Q Analytical'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
subplot(3,1,3) 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim
                hold on
                plot(v,Q_2(:,i,j)-Qt(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Difference'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
name=horzcat(FolderName,'/Figures/Qt.svg');
filename_4000  = sprintf(name);
saveas(4000,filename_4000);

% Compute D(t) the orthogonal matrix from QT Analytical Solution
figure(2); 
v=1:1:t3;
k=0;
for t=v
        hold on
        k=k+1;
        Q_2_plot(:,:)=Q(t1+t,:,:);
        Q_3(:,:)=Qt(t,:,:);
        Qt_plot= pinv(Q_3);
        D(k,:,:)=(Qt_plot)*Q_2_plot;
end

 v=1:1:t3;
  for i=1:1:2
     for j=1:1:2
        plot(v,D(:,i,j),'LineWidth',2);
     
     end
  end
  
y_label = ylabel('quotient_of_q'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
name=horzcat(FolderName,'/Figures/quotient_of_q.svg');
filename_2  = sprintf(name);
saveas(2,filename_2);


% Plot to check that D(T) is an orthogonal matrix 
%  Should be an identity
% fig_num=3;
% % name_vid=horzcat('./data/',FolderName,'/Video/Disdentity.avi');
% k=0;
% for i=1:1:t3
  %   k=k+1;
  %   diff_i(:,:)=D(i,:,:);
  %   identity(k,:,:)= diff_i*diff_i';
% end
% name_plot='D*D';
% y_label = '' ;
% x_label = '';
% name_fig=horzcat('./data/',FolderName,'/Figures/Disdentity.svg');
% plot_rates_maps(fig_num,name_plot,name_vid,name_fig,identity,line,x_label,y_label)

% Plot the Simulation of Q_t solution as a colourmap.
% (Up to a orthogonal transformation)
% fig_num=4;
% name_vid=horzcat('./data/',FolderName,'/Video/qt_simul_img.avi');
% qtqtT_plot_1(:,:,:)= Q(1:t3,1:2*dim,1:dim);
% name_plot='Q(t)T Analytical Without Orthogonal transform for Ana Solution';
% x_label='Hidden';
% y_label='Output                    Input';
% name_fig=horzcat('./data/',FolderName,'/Figures/qt_simul_img.svg');
% plot_rates_maps(fig_num,name_plot,name_vid,name_fig,qtqtT_plot_1,line,x_label,y_label)

% Plot the difference between simulation and Analytical solution
% of Q_t as a colourmap.
% (Up to a orthogonal transformation)
% fig_num=5;
% name_vid=horzcat('./data/',FolderName,'/Video/qt_simul_img.avi');
% qtqtT_plot_1(:,:,:)= Q(1:t3,1:2*dim,1:dim)-Qt(1:t3,1:2*dim,1:dim);
% name_plot='Difference Without Orthogonal transform for Ana Solution';
% x_label='Hidden';
% y_label='Output                    Input';
% name_fig=horzcat('./data/',FolderName,'/Figures/Difference_Q.svg');
% plot_rates_maps(fig_num,name_plot,name_vid,name_fig,qtqtT_plot_1,line,x_label,y_label)

%% Compute full Qt Analytical Solution with Orthogonal transformation D (See Equation .)

Qt_d=q_tsquare_D(t3,tau, X, Y, X_tilde, Y_tilde,FolderName,R,D,plot_variable_q_u);


% Plot the difference between Simulation and Analytical solution
% of Q_t as a colourmap.
% with Orthogonal transformation D
% fig_num=6;
% name_vid=horzcat('./data/',FolderName,'/Video/Difference_QD.avi');
% qtqtT_plot_1(:,:,:)= Q(1:t3,1:dim*2,1:dim)-Qt_d(1:t3,1:2*dim,1:dim);
% name_plot='Q(t)T Analytical With Orthogonal transform for Ana Solution';
% x_label='Hidden';
% y_label='Output                    Input';
% name_fig=horzcat('./data/',FolderName,'/Figures/Difference_QD.svg');
% plot_rates_maps(fig_num,name_plot,name_vid,name_fig,qtqtT_plot_1,line,x_label,y_label)

% Plot the Analitical and Simulation solution of Q_t as well as their difference.
% (Up to a orthogonal transformation)
Q_2=Q(t1+1:1:t2,:,:);
figure(5000); 
subplot(3,1,1) 
title('With Orthogonal transform for Ana Solution') 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim
                hold on
                plot(v,Q_2(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Q Simulation'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
subplot(3,1,2) 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim
                hold on
                plot(v,Qt_d(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Q Analytical'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
subplot(3,1,3) 
v=1:1:t3;
 for i=1:1:dim*2
     for j=1:1:dim
                hold on
                plot(v,Q_2(:,i,j)+Qt_d(:,i,j),'LineWidth',2);
     end
 end
y_label = ylabel('Difference'); 
set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
x_label = xlabel('Epochs'); %or h=get(gca,'xlabel') 
set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')
name=horzcat(FolderName,'/Figures/Q_Dt.svg');
filename_5000  = sprintf(name);
saveas(5000,filename_5000);

for i=1:1:t3
    Q_(:,:)=Qt(i,:,:);
    QQ_2(i,:,:)=Q_*Q_';
end 
 


%% Compute full Qt Analytical Solution with Orthogonal transformation D (See Equation .)

%Qt_d=q_tsquare_D(t3,tau, X, Y, X_tilde, Y_tilde,FolderName,R,D,plot_variable_q_u);





for i=1:1:t3
    Q_d(:,:)=Qt_d(i,:,:);
    QQ_2_d(i,:,:)=Q_d*Q_d';
end 

 

% Compute D(t) the orthogonal matrix from QT Analytical Solution
figure(2); 
v=1:1:t3;
k=0;
for t=v
        k=k+1;
        Q_2_plot(:,:)=Q(t1+t,:,:);
        Q_3(:,:)=Qt(t,:,:);
        Qt_plot= pinv(Q_3);
        D(k,:,:)=(Qt_plot)*Q_2_plot;
end







%% Saving the Work space
cd(horzcat(FolderName,'/'));
save('run')

