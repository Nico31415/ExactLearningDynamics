function plot_qq_t=plot_qq_t(QtQtTs,t1,t2,FolderName,plot_qqt,line)
% Plot the simulated Q_TQ_t 
    if plot_qqt==1
        
        t3 =t2-t1;
        line_1=linspace(2,t1,100) ;
        line_1=round(line_1)-1;
        fig_num=703; 
        QtQtTs_1(:,:,:)=QtQtTs(1:t1,:,:);
        size_q=size(QtQtTs_1)
        QtQtTs_1_plots(:,:,:)= QtQtTs_1(1:1:t1,1:size_q(2),1:size_q(2));
        name_vid=horzcat('./data/',FolderName,'/Video/QQtsimul_task1.avi');
        name_plot='Q(t)Q(t)T 1 Simulation Task 1';
        name_fig=horzcat('./data/',FolderName,'/Figures/QQtsimul_task1.svg');
        x_label='Input                     Output';
        y_label='Output                     Input';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,QtQtTs_1_plots,line_1,x_label,y_label)
       
        line_1=linspace(2,t3,100) ;
        line_1=round(line_1)-1;
        fig_num=703; 
        QtQtTs_2(:,:,:)=QtQtTs(t1:t2,:,:);
        size_q=size(QtQtTs_2)
        QtQtTs_2_plots(:,:,:)= QtQtTs_2(1:1:t3,1:size_q(2),1:size_q(2));
        name_vid=horzcat('./data/',FolderName,'/Video/QQtsimul_task2.avi');
        name_plot='Q(t)Q(t)T 1 Simulation Task 2';
        name_fig=horzcat('./data/',FolderName,'/Figures/QQtsimul_task2.svg');
        x_label='Input                     Output';
        y_label='Output                     Input';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,QtQtTs_2_plots,line_1,x_label,y_label)
        
        line_1=linspace(1,4,4) ;
        fig_num=703; 
        QtQtTs_end(:,:,:)=QtQtTs(t2-3:t2,:,:);
        size_q=size(QtQtTs_end);
        QtQtTs_2_plots(:,:,:)= QtQtTs_2_end(:,1:size_q(2),1:size_q(2));
        name_vid=horzcat('./data/',FolderName,'/Video/QQtsimul_task2.avi');
        name_plot='Q(t)Q(t)T 1 Simulation Task 2';
        name_fig=horzcat('./data/',FolderName,'/Figures/QQtsimul_task2.svg');
        x_label='Input                     Output';
        y_label='Output                     Input';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,QtQtTs_2_plots,line_1,x_label,y_label)
        
     end
end 

