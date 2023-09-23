%% House Keeping
close all
clear all 
addpath('../THETNS','../PlottingRoutine/cbrewer','../PlottingRoutine')  
FolderName=['General',datestr(now,'mm-dd-yyyy HH-MM')];
cd('..');
cd('./data/Fig 4-Generalisaiton/');
mkdir(FolderName, 'Video'); 
mkdir(FolderName, 'Figures'); 
cd('..');
cd('..');

learnrate = 0.01;
eta = learnrate;
tau=(1/eta);
samples_n=1;
N_x_t = 8;    %teacher input dimension
N_y_t = 8;  %teacher input dimension
target_precision_1 = 8e-10;
Nh = 10;

% generate training examples
%sigma=0.1;
%X =  normrnd(0, sigma, [N_x_t, samples_n]);
%X = X - mean(X,2);
%cov_1 = cov(X',1);
%[eigvecs,eigvals] = eig(cov_1);
%X =(diag((1 ./ diag(sqrt(eigvals)))))*eigvecs'*X ;
%W1_target = normrnd(0, sigma, [Nh, N_x_t]);
%W2_target = normrnd(0, sigma, [N_y_t,Nh]);
%Y = W2_target * W1_target * X;

X=eye(N_x_t,N_x_t);
Y=[-4,-4,-4,-4,-4,-4,-4,-4;-3,-3,-3,-3,3,3,3,3;-4,-4,4,4,0,0,0,0;0,0,0,0,-3,-3,3,3;0,0,0,0,0,0,-4,4;0,0,0,0,-3,3,0,0;0,0,-2,2,0,0,0,0;-1,1,0,0,0,0,0,0];

k=0;
line=linspace(0.001,1,25) ;
% Saving
cd(horzcat('./data/Fig 4-Generalisaiton/',FolderName,'/'));
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
path_y_t_new=horzcat('y_t_new.csv');
csvwrite(path_y_t_new,Y);

 
Ni = size(X,1);
No = size(Y,1); 
smaller_dim =min( Ni, No);
larger_dim =max( Ni, No);
[U,S,V]=svd(Y*X');
S=diag(diag(S))        ;
if Ni<No
    RSA_W2=((U(:,1:smaller_dim)*S*U(:,1:smaller_dim)'))/samples_n;
    RSA_W1=((V(:,:)*S*V(:,:)'))/samples_n;
else
    RSA_W2=((U(:,:)*S*U(:,:)'))/samples_n;
    RSA_W1=((V(:,1:smaller_dim)*S*V(:,1:smaller_dim)'))/samples_n;
end
                
% Choose weight scheme
for g=1:1:2
    % Choose std
    k=0;
    for l=line
         k=k+1
        % Number of trial for the mean
        for j=1:1:100
            if g==1
                Ni = size(X,1);
                No = size(Y,1);
                r(:,:)= normrnd(0,0.0000001,Nh,Nh);
                [R,R_2 ,R_3]=svd(r);
                A(:,:)= normrnd(0,l,No,Ni);
                [U, S, V ]=svd(A);
                SS=diag(diag(S./(sqrt(S))));
                S1 = [SS; zeros(Nh- smaller_dim, smaller_dim)];
                S0 = [SS, zeros(smaller_dim, Nh - smaller_dim)]; 
                if Ni<No
                    W1 =  R*S1 * V;
                    W2 =  U(:,1:smaller_dim) * S0 * R';
                else
                    W1 =  R*S1*V(1:smaller_dim,:);
                    W2 =  U * S0 * R';
                end
                assert( all(all(round(W1 *W1')==round(W2' *W2))),'oops')
            end
      
            if g==2
                sigma_w=l;
                W1 =  normrnd(0,l,Nh,Ni);
                W2 = normrnd(0,l,No,Nh);
                Y_a=W2*W1*X;
            end   
            gain_W1_j(j)= std(W1,1,'all')/sqrt(Nh);
            gain_W2_j(j)= std(W2,1,'all')/sqrt(Nh);

            inputs =  X;
            targets = Y;
            training_steps_max = 255000000;
            x=0;
            training_step=0;

            %% Run
               while x==0
                   training_step= training_step+1;
                    % Compute Q_TQ_T
                    Qt = [W1'; W2];
                    Q(training_step,:,:)= Qt;
                    QtQtTs(training_step,:,:)=(Qt*Qt');
                    QQT(:,:)=QtQtTs(training_step,:,:);
                    h1 = W1 *inputs;
                    Y_hat = W2 * h1;
                    x_t_input_e = eye(N_x_t); 
                    h_1 = W1* x_t_input_e;

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
                   % Switch from task 1 to task 2
                    if losses(training_step) < target_precision_1
                        x=x+1;
                        if x==1
                            t1 = training_step;
                        end     
                    end
               end
          x=0;
          Error_RSA_W1_j(k,j) = (mean(mean((W1'*W1-RSA_W1).^2)));
          Error_RSA_W2_j(k,j) = (mean(mean((W2*W2'-RSA_W2).^2)));
          Error_RSA_W1_jf(k,j) = norm(W1'*W1-RSA_W1,"fro")^2;
          Error_RSA_W2_jf(k,j) = norm(W2*W2'-RSA_W2,"fro")^2;

          
        end    
        gain_W1(g,k)= mean(gain_W1_j);
        gain_W2(g,k)= mean(gain_W2_j);
        std_RSA_W1(g,k)= std(Error_RSA_W1_j(k,:),1);
        std_RSA_W2(g,k)= std(Error_RSA_W2_j(k,:),1);
        Error_RSA_W1(g,k) = (mean(Error_RSA_W1_j(k,:)));
        Error_RSA_W2(g,k) = (mean(Error_RSA_W2_j(k,:)));
        std_RSA_W1_f(g,k)= std(Error_RSA_W1_jf(k,:),1);
        std_RSA_W2_f(g,k)= std(Error_RSA_W2_jf(k,:),1);
        Error_RSA_W1_f(g,k) = (mean(Error_RSA_W1_jf(k,:)));
        Error_RSA_W2_f(g,k) = (mean(Error_RSA_W2_jf(k,:)));
        
    end
    line=linspace(0.01,0.5,25)
    
end 
std=[line, gain_W1(2,:)]

save('run')
cd('..');
cd('..');

