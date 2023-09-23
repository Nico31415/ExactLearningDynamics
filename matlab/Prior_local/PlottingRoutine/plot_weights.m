function plot_weights_deep= plot_weights(W_1_overtime,W_2_overtime,t1,t2, FolderName, plot_weight,line)
% Plot the weights overtime

    if  plot_weight==1
         % Plot the simulated weight 1 dynamics as a colourmap overtime
        fig_num=301; 
        size_x=size(W_1_overtime);
        W1_plot_3(:,:,:)=W_1_overtime(t1:t2,1:size_x(2),1:size_x(2));
        name_vid=horzcat('./data/',FolderName,'/Video/W1.avi');
        name_plot='W1 over time simulation';
        name_fig=horzcat('./data/',FolderName,'/Figures/W1.svg');
        y_label='Input';
        x_label='Hidden';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,W1_plot_3,line,x_label,y_label)
        
        % Plot the simulated weight 2 dynamics as a colourmap overtime
        fig_num=302;
        W2_plot_3(:,:,:)=W_2_overtime(t1:t2,1:size_x(2),1:size_x(2));
        name_plot='W2 over time simulation';
        name_vid=horzcat('./data/',FolderName,'/Video/W2.avi');
        name_fig=horzcat('./data/',FolderName,'/Figures/W2.svg');
        x_label='Hidden';
        y_label='Output';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,W2_plot_3,line,x_label,y_label)
    end 
end 