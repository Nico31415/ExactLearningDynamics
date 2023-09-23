function plot_weights_svd= plot_weights_svd(Ss,Ss_2,Uu,Uu_2,Vv,Vv_2,t1,t2, FolderName, plot_weight_SVD,line,X)
% Plot the SVD of the weights overtime

if plot_weight_SVD==1
    
    
     
         % Plot the singular values of the 1st weights matrix overtime
        figure(410); 
        v=1:1:t2;
        size_x=size(X);
        for i=1:1:size_x(1)
            hold on
            plot(v,Ss(1:1:t2,i),'LineWidth',2,'Color','#BD2230');
            ax = gca;
            ax.XAxis.Exponent = 0;
            x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
            tick2 =  num2str(get(gca,'YTick')','%g');
            tick1 =  num2str(get(gca,'XTick')','%g');
            set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
            set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
            set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
            y_label = ylabel('Singular Values W1'); %or h=get(gca,'xlabel')
            set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')   
        end
        hold on 
        xline(t1,'Color','k','LineStyle','--','LineWidth',2);
        set(0, 'DefaultFigureRenderer', 'painters');
        name=horzcat('./data/',FolderName,'/Figures/S_W1_t1.svg');
        filename_410  = sprintf(name);
        saveas(410,filename_410)
        
         % Plot the singular values of the 2nd weights matrix overtime
        figure(407); 
        v=1:1:t1;
        SS1(:,:)=Ss_2(1:1:t1,:);
        plot(v,SS1,'LineWidth',2,'Color','#BD2230');
        hold on 
        xline(t1,'Color','k','LineStyle','--','LineWidth',2);
        ax = gca;
        ax.XAxis.Exponent = 0;
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Singular Values W2'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')   
        name=horzcat('./data/',FolderName,'/Figures/S_W2_t1.svg');
       set(0, 'DefaultFigureRenderer', 'painters');
        filename_407  = sprintf(name);
        saveas(407,filename_407)
        
       
        
       
        figure(408); 
        v=t1:1:t2;
        size_x=size(X);
        for i=1:1:size_x(1)
        hold on
         plot(v,Ss(t1:1:t2,i),'LineWidth',2,'Color','#BD2230');
         ax = gca;
         ax.XAxis.Exponent = 0;
            x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
            tick2 =  num2str(get(gca,'YTick')','%g');
            tick1 =  num2str(get(gca,'XTick')','%g');
            set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
            set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
            set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
            y_label = ylabel('Singular Values W1'); %or h=get(gca,'xlabel')
            set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')   
        end
        hold on 
        xline(t1,'Color','k','LineStyle','--','LineWidth',2);
        set(0, 'DefaultFigureRenderer', 'painters');
        name=horzcat('./data/',FolderName,'/Figures/S_W1_last.svg');
        filename_408  = sprintf(name);
        saveas(408,filename_408)

        % Plot the singular values of the 2nd weights matrix overtime
        figure(409); 
        v=t1:1:t2;
        SS3(:,:)=Ss_2(t1:1:t2,:);
        plot(v,SS3,'LineWidth',2,'Color','#BD2230');
        hold on 
        xline(t1,'Color','k','LineStyle','--','LineWidth',2);
        ax = gca;
        ax.XAxis.Exponent = 0;
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Singular Values W2'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')   
        name=horzcat('./data/',FolderName,'/Figures/S_W2_last.svg');
       set(0, 'DefaultFigureRenderer', 'painters');
        filename_409  = sprintf(name);
        saveas(409,filename_409)
        
        figure(401); 
        v=1:1:t1;
        size_x=size(X);
        for i=1:1:size_x(1)
            hold on
            plot(v,Ss(1:1:t1,i),'LineWidth',2,'Color','#BD2230');
            ax = gca;
            ax.XAxis.Exponent = 0;
            x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
            tick2 =  num2str(get(gca,'YTick')','%g');
            tick1 =  num2str(get(gca,'XTick')','%g');
            set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
            set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
            set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
            y_label = ylabel('Singular Values W1'); %or h=get(gca,'xlabel')
            set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')   
        end
        hold on 
        xline(t1,'Color','k','LineStyle','--','LineWidth',2);
        set(0, 'DefaultFigureRenderer', 'painters');
        name=horzcat('./data/',FolderName,'/Figures/S_W1.svg');
        filename_401  = sprintf(name);
        saveas(401,filename_401)

       

        % Plot the singular values of the 2nd weights matrix overtime
        figure(402); 
        v=1:1:t2;
        SS_1(:,:)=Ss_2(1:1:t2,:);
        plot(v,SS_1,'LineWidth',2,'Color','#BD2230');
        hold on 
        xline(t1,'Color','k','LineStyle','--','LineWidth',2);
        ax = gca;
        ax.XAxis.Exponent = 0;
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel('Singular Values W2'); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')   
        name=horzcat('./data/',FolderName,'/Figures/S_W2.svg');
       set(0, 'DefaultFigureRenderer', 'painters');
        filename_402  = sprintf(name);
        saveas(402,filename_402)

        % Plot the singular matrix U of the 1st weights matrix overtime
        fig_num=403;
        Uu_plot_2(:,:,:)=Uu(t1:t2,1:size_x(1),1:size_x(1));
        name_plot='U W1 simulation';
        name_vid=horzcat('./data/',FolderName,'/Video/U_W1.avi');
        name_fig=horzcat('./data/',FolderName,'/Figures/U_W1.svg');
        x_label='Modes';
        y_label='Hidden';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,Uu_plot_2,line,x_label,y_label);
        
        % Plot the singular matrix U of the 2nd weights matrix overtime
        fig_num=404;
        Uu_plot_3(:,:,:)=Uu_2(t1:t2,1:size_x(1),1:size_x(1));
        name_plot='U W2 simulation';
        name_vid=horzcat('./data/',FolderName,'/Video/U_W2.avi');
        name_fig=horzcat('./data/',FolderName,'/Figures/U_W2.svg');
        x_label='Modes';
        y_label='Output';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,Uu_plot_3,line,x_label,y_label)

        % Plot the singular matrix V of the 1st weights matrix overtime
        fig_num=405;
        VV_plot_3(:,:,:)=Vv(t1:t2,1:size_x(1),1:size_x(1));
        name_plot='V W1 simulation';
        name_vid=horzcat('./data/',FolderName,'/Video/V_W1.avi');
        name_fig=horzcat('./data/',FolderName,'/Figures/V_W1.svg');
        x_label='Modes';
        y_label='Input';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,VV_plot_3,line,x_label,y_label)

        % Plot the singular matrix V of the 2st weights matrix overtime
        fig_num=406;
        VV_plot(:,:,:)=Vv_2(t1:t2,1:size_x(1),1:size_x(1));
        name_plot='V W2 simulation';
        name_vid=horzcat('./data/',FolderName,'/Video/V_W2.avi');
        name_fig=horzcat('./data/',FolderName,'/Figures/V_W2.svg');
        x_label='Modes';
        y_label='Hidden';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,VV_plot,line,x_label,y_label)
        
        
        
       
        
end

