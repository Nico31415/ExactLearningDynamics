function plot_weights_svd= plot_q_svd(Sq,Uq,Vq,t1,t2, FolderName, plot_weight_SVD,line,X)
% Plot the SVD of the weights overtime
if plot_weight_SVD==1
        % Plot the singular values of the 1st weights matrix overtime
        figure(1201); 
        v=1:1:t2;
        size_x=size(X);
        for i=size_x(1):1:2*size_x(1)
            hold on
            plot(v,Sq(1:1:t2,i),'LineWidth',2);
            ax = gca;
            ax.XAxis.Exponent = 0;
            x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
            tick2 =  num2str(get(gca,'YTick')','%g');
            tick1 =  num2str(get(gca,'XTick')','%g');
            set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
            set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
            set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
            y_label = ylabel('Singular Values QQ'); %or h=get(gca,'xlabel')
            set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')   
        end
        hold on 
        xline(t1,'Color','k','LineStyle','--','LineWidth',2);
        name=horzcat('./data/',FolderName,'/Figures/S_qq.svg');
        filename_1201  = sprintf(name);
        saveas(1201,filename_1201)

        % Plot the singular matrix U of the 1st weights matrix overtime
        fig_num=1202;
        Uu_plot_2(:,:,:)=Uq(t1:t2,1:2*size_x(1),1:2*size_x(1));
        name_plot='U QQ simulation';
        name_vid=horzcat('./data/',FolderName,'/Video/U_qq.avi');
        name_fig=horzcat('./data/',FolderName,'/Figures/U_qq.svg');
        x_label='Modes';
        y_label='Hidden';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,Uu_plot_2,line,x_label,y_label);
        
        % Plot the singular matrix V of the 1st weights matrix overtime
        fig_num=1203;
        VV_plot_3(:,:,:)=Vq(t1:t2,1:2*size_x(1),1:2*size_x(1));
        name_plot='V QQ simulation';
        name_vid=horzcat('./data/',FolderName,'/Video/V_qq.avi');
        name_fig=horzcat('./data/',FolderName,'/Figures/V_qq.svg');
        x_label='Modes';
        y_label='Input';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,VV_plot_3,line,x_label,y_label)

end

