function plot_loss=plot_loss_RSA(losses,losses_tilde,t1,t2,RSA_timepoints,RSA_mat,FolderName,plotting)
 
%function to plot the loss and the RSA matrix
    if plotting==1

        % Plot the loss curve
        figure(201);  
        plot(losses,'LineWidth',2,'Color','#276FB4');
        hold on 
        plot(losses_tilde,'LineWidth',2,'Color','#276FB4');
        hold on 
        plot(RSA_timepoints,losses(RSA_timepoints),'o','LineWidth',2,'Color','k')
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
        name=horzcat('./data/',FolderName,'/Figures/loss.svg');
        legend('Task1','Task2','RSA points')
        name=horzcat('./data/',FolderName,'/Figures/loss.svg');
        set(0, 'DefaultFigureRenderer', 'painters');
        filename_201  = sprintf(name);
        saveas(201,filename_201)

        % Plot RSA matrices over training
        figure(202)
         n = 1001; % should be odd
         cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom
         caxis('manual')
         colormap(cmap);
        
          lin = (0:.01:1)';
          for j = 1:length(RSA_timepoints)
          Maxi_1= max(RSA_mat{j}(:),[],'all');
          Mini_1= min(RSA_mat{j}(:),[],'all');
          Vector_max_min_squared_1=[Maxi_1.^2,Mini_1.^2];
          l_1=sqrt(max(Vector_max_min_squared_1));
          subplot(2,3,j)
          imagesc(RSA_mat{j})
 
           caxis([-l_1,+l_1])
          tick2 =  num2str(get(gca,'YTick')','%g');
          tick1 =  num2str(get(gca,'XTick')','%g');
          set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
          set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
          x_label = xlabel('Input'); %or h=get(gca,'xlabel')
          set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
          y_label = ylabel('Input'); %or h=get(gca,'xlabel')
          set(y_label,'FontSize', 20,'FontName' , 'Times New Roman') 
          axis square
        end
        name=horzcat('./data/',FolderName,'/Figures/RSA.svg');
        set(0, 'DefaultFigureRenderer', 'painters');
        filename_202  = sprintf(name);
        saveas(202,filename_202);
    end 
end 