function plot_inp=plot_inp(Y,X,Y_tilde,X_tilde,tau,plot_inputs,FolderName)
% Plot properties of the inputs
addpath('../THETNS','../PlottingRoutine/cbrewer','../PlottingRoutine')  
sigma_xy_tilde = (Y_tilde*X_tilde');
sigma_xy = (Y*X');
[U, S, V] = svd(sigma_xy);
G=U'*sigma_xy_tilde*V;
 
figure(103); 
        subplot(2,3,2)
        Maxi= max(G,[],'all');
        Mini= min(G,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(G);
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',12)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',12)
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('U') 
 if plot_inputs== 1
        
        Ni = size(X,1);
        
        % Plot the predicted singular value of the first task over time
        figure(101);
        [u,s,~] = svd(Y);
        t = linspace(1,500,500);
        a0 = .001;
        sd = diag(s);
        tmp = exp(2*bsxfun(@times,sd,t/tau));
        a = bsxfun(@times,sd,tmp) ./ bsxfun(@plus, tmp, sd/a0-1);
        plot(t,a','-b','LineWidth',3);
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',12)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',12)
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Singular Values of Y'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        name=horzcat('./data/',FolderName,'/Figures/SVD_of_task_1.svg');
        filename_101  = sprintf(name);
        saveas(101,filename_101)

        % Plot the predicted singular value of the first task and the 
        % 2nd task over time
        figure(102);
        [u_y,s,v_y] = svd(Y);
        t = linspace(1,500,50);
        a0 = .001;
        sd = diag(s);
        tmp = exp(2*bsxfun(@times,sd,t/tau));
        a = bsxfun(@times,sd,tmp) ./ bsxfun(@plus, tmp, sd/a0-1);
        p1=plot(t,a','-b','LineWidth',3);
        [u_new,s_new,v_new] = svd(Y_tilde);
        a0_new = .001;
        sd_new = diag(s_new);
        tmp_new = exp(2*bsxfun(@times,sd_new,t/tau));
        a_new = bsxfun(@times,sd_new,tmp_new) ./ bsxfun(@plus, tmp_new, sd_new/a0_new-1);
        t_new = linspace(1,500,50);
        hold on
        p2=plot(t_new,a_new','-r','LineWidth',1);
        legend([p1(1),p2(1)],'Y','Y_{tilde}')
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',12)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',12)
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Singular Values'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        name=horzcat('./data/',FolderName,'/Figures/SVD_of_task1_task2.svg');
        filename_102  = sprintf(name);
        saveas(102,filename_102)

        sigma_xy =  (Y*X');
        sigma_xy_tilde = (Y_tilde*X_tilde');
        [U, S, V] = svd(sigma_xy);
        [U_tilde, S_tilde, V_tilde] = svd(sigma_xy_tilde);
        
        for i=1:1:8
            if U(i,i) < 0
                U(i,:)= -U(i,:);
                V(:,i)= -V(:,i);
            end
        end
        for i=1:1:8
            if U_tilde(i,i) < 0
                U_tilde(i,i)= -U_tilde(i,i);
                V_tilde(:,i)= -V_tilde(:,i);
            end
        end
        
        L =(sigma_xy_tilde- sigma_xy)
        % Plot the singular matrix of the first task and the second task.
        % Plot the product of the singular matrix of the first and second task.
        
        % Choose the order of the collumns.
        k=[1,2,3,4,5,6,7,8];
        figure(103); 
        subplot(2,3,2)
        Maxi= max(U,[],'all');
        Mini= min(U,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(U(1:Ni,k(1:8)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',12)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',12)
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('U') 

        subplot(2,3,3)
        Maxi= max(V,[],'all');
        Mini= min(V,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(V(1:Ni,k(1:8)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',12)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',12)
        y_label = ylabel('Input'); %or h=get(gca,'xlabel')
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('V') 
        

        subplot(2,3,5);
        Maxi= max(U_tilde,[],'all');
        Mini= min(U_tilde,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(U_tilde(1:Ni,k(1:8)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('U_{tilde}') 
        
        subplot(2,3,6)
        Maxi= max(V_tilde,[],'all');
        Mini= min(V_tilde,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 1001; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(V_tilde(1:Ni,k(1:8)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Input'); %or h=get(gca,'xlabel')
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('V_{tilde}') 

        subplot(2,3,1)
        L=V'*V_tilde;
        Maxi= max(L,[],'all');
        Mini= min(L,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 1001; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(L(1:Ni,:));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Modes'); %or h=get(gca,'xlabel')
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') ;
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') ;
        title('V*V_{tilde}')

        subplot(2,3,4)
        L=U'*U_tilde;
        Maxi= max(L,[],'all');
        Mini= min(L,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 1001; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(L(1:Ni,:));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Modes'); %or h=get(gca,'xlabel')
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') ;
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') ;
        title('U*U_{tilde}')
        name=horzcat('./data/',FolderName,'/Figures/y_y_tilde_V_U.svg');
        filename_103  = sprintf(name);
        saveas(103,filename_103)
    
        % Plot the singular value of the first and the 2nd task. 
        % Plot the difference and addition of the singular value of 
        % the first and second task
        figure(104); 
        subplot(2,2,1)
        Maxi= max(S_tilde,[],'all');
        Mini= min(S_tilde,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(S_tilde(1:Ni,k(1:Ni)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('S_{tilde}')

        subplot(2,2,2)
        Maxi= max(S,[],'all');
        Mini= min(S,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(S(1:Ni,k(1:Ni)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Input'); %or h=get(gca,'xlabel')
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('S')
      
        
        minu=S-S_tilde;
        if minu==0
            
        else
        subplot(2,2,3);
        Maxi= max(minu,[],'all');
        Mini= min(minu,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(minu(1:Ni,k(1:Ni)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('S-S_{tilde}')
        end 
        subplot(2,2,4)
        plus=S+S_tilde;
        Maxi= max(plus,[],'all');
        Mini= min(plus,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 1001; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(plus(1:Ni,k(1:Ni)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Input'); %or h=get(gca,'xlabel')
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        name=horzcat('./data/',FolderName,'/Figures/y1-ytilde-S.svg');
        title('S+S_{tilde}')
        filename_104  = sprintf(name);
        saveas(104,filename_104)

        F=[zeros(8,8),sigma_xy_tilde;sigma_xy_tilde ,zeros(8,8)];
        [Uf,Sf,Vf] = svd(F);
         
        figure(105); 
        subplot(2,2,1)
        Maxi= max(Uf,[],'all');
        Mini= min(Uf,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(Sf(1:2*Ni,1:2*Ni));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('S_f')

        subplot(2,2,2)
        Maxi= max(Vf,[],'all');
        Mini= min(Vf,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(Vf(1:2*Ni,1:2*Ni));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Input'); %or h=get(gca,'xlabel')
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('Vf')
      
        minu=Uf;
        subplot(2,2,3);
        Maxi= max(minu,[],'all');
        Mini= min(minu,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(minu(1:2*Ni,1:2*Ni));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        x_label = xlabel('Modes'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('Uf')
        name=horzcat('./data/',FolderName,'/Figures/y1-ytilde-F.svg');
        filename_105  = sprintf(name);
        saveas(105,filename_105)

        sigma_xy =  (Y*X');
        sigma_xy_tilde = (Y_tilde*X_tilde');
        figure(106); 
        subplot(1,2,1)
        Maxi= max(sigma_xy_tilde,[],'all');
        Mini= min(sigma_xy_tilde,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(sigma_xy_tilde(1:Ni,k(1:8)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',12)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',12)
        x_label = xlabel('Input'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('Task B') 
        
        subplot(1,2,2)
        Maxi= max(sigma_xy,[],'all');
        Mini= min(sigma_xy,[],'all') ;
        Vector_max_min_squared=[Maxi.^2,Mini.^2];
        l=sqrt(max(Vector_max_min_squared));
        n = 101; % should be odd 
        cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
        caxis('manual')
        colormap(cmap); 
        imagesc(sigma_xy(1:Ni,k(1:8)));
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l],'FontSize',15)
        caxis([-l,+l])
        axis square
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',12)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',12)
        x_label = xlabel('Input'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Output'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        title('Task A') 
        name=horzcat('./data/',FolderName,'/Figures/autocorolation.svg');
        filename_106  = sprintf(name);
        saveas(106,filename_106)

 end 
