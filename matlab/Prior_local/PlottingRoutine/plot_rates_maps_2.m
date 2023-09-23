function []=plot_rates_maps_2(fig_num,name,name_fig,matrix1,matrix2,line,name_x,name_y)
% Plot and save 2 coulour map

    n = 1001; % should be odd 
    cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom 
    figure(fig_num)
    v = VideoWriter(name);
    open(v);

    Maxi_1= max(matrix1,[],'all');
    Mini_1= min(matrix1,[],'all');
    Maxi_2= max(matrix2,[],'all');
    Mini_2= min(matrix2,[],'all');

    Vector_max_min_squared_1=[Maxi_1.^2,Mini_1.^2];
    l_1=sqrt(max(Vector_max_min_squared_1));
    Vector_max_min_squared_2=[Maxi_2.^2,Mini_2.^2];
    l_2=sqrt(max(Vector_max_min_squared_2));

    for i=line
        
        subplot(2,1,1)
        Plot_matrix(:,:)=matrix1(i,:,:);
        caxis('manual')
        colormap(cmap); 
        imagesc(Plot_matrix);
        colorbar('Ticks', [-l_1, 0, +l_1], 'TickLabels',[-l_1,0,l_1],'FontSize',12)
        caxis([-l_1,+l_1])
        set(gca,'XTickLabel',[1,2,3,4,5,6,7,8],'FontName','Times','fontsize',12)   
        set(gca,'YTickLabel',[1,2,3,4,5,6,7,8],'FontName','Times','fontsize',12)
        set(gca,'XTickLabelMode','auto')
        x_label = xlabel(name_x); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel(name_y); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')  
        axis square

       
        subplot(2,1,2)
        Plot_matrix(:,:)=matrix2(i,:,:);
        caxis('manual')
        colormap(cmap); 
        imagesc(Plot_matrix);
        colorbar('Ticks', [-l_2, 0, +l_2], 'TickLabels',[-l_2,0,l_2],'FontSize',12)
        caxis([-l_2,+l_2])
        t1 = get(gca,'XTickLabel');
        t2 = get(gca,'YTickLabel');
        set(gca,'XTickLabel',[1,2,3,4,5,6,7,8],'FontName','Times','fontsize',12)   
        set(gca,'YTickLabel',[1,2,3,4,5,6,7,8],'FontName','Times','fontsize',12)
        set(gca,'XTickLabelMode','auto')
        x_label = xlabel(name_x); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        y_label = ylabel(name_y); %or h=get(gca,'xlabel')
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')  
        axis square
          
        frame = getframe(gcf);
        writeVideo(v,frame);
    end  
     
    close(v)
    filename  = sprintf(name_fig);
    saveas(fig_num,filename);
end
