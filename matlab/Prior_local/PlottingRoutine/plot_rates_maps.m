function []=plot_rates_maps(fig_num,name_plot,name,name_fig,matrix,line,name_x,name_y)
% Plot and save 1 coulour map

    figure(fig_num)
    Maxi= max(matrix,[],'all');
    Mini= min(matrix,[],'all') ;
    Vector_max_min_squared=[Maxi.^2,Mini.^2];
    l=sqrt(max(Vector_max_min_squared));
    n = 1001; % should be odd
    cmap = flipud(cbrewer('div','RdBu',n)); % blues at bottom
    v = VideoWriter(name);
    open(v);

    for i=line
        Plot_matrix(:,:)=matrix(i,:,:);
        caxis('manual')
        colormap(cmap);
        imagesc(Plot_matrix);
        colorbar('Ticks', [-l, 0, +l], 'TickLabels',[-l,0,+l], 'FontSize',15)
        caxis([-l,+l])
        title(name_plot)
        a = get(gca,'XTickLabel');
        b = get(gca,'YTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',b,'FontName','Times','fontsize',14)
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
