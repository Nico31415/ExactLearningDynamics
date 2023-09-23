function delta=delta_learning_projection(delta_W_1_overtime,delta_W_2_overtime,U_task_1,U_task_1_2,Vv_task_1,Vv_task_1_2,t1,t2,FolderName,plot_delta,line,line_matrix)
% Plot gradient step overtime for both weights
    if  plot_delta==1
    
        t3=t2-t1;
        size_delta=size(delta_W_1_overtime);
        Ni=size_delta(2);
        
        % Plot gradient step against time for weight 1
        figure(601)
        name=horzcat('./data/',FolderName,'/Video/Delta_W1.avi');
        vid = VideoWriter(name);
        open(vid);
        v=1:1:t3;
        for i=1:1:Ni
             for j=1:1:Ni
                  if i ~=j  
                    hold on
                    plot(v,delta_W_1_overtime(t1+1:t2,i,j),'LineWidth',2,'Color','b');
                    y_label = ylabel('Delta W1'); 
                    set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
                    x_label = xlabel('Epochs'); 
                    set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
                    frame = getframe(gcf);
                    writeVideo(vid,frame);
                    pause(0.1);
                  end
             end
        end
        close(vid)
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Delta W1'); 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
        x_label = xlabel('Epochs'); 
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        name=horzcat('./data/',FolderName,'/Figures/Delta_W1.png');
        filename_601  = sprintf(name);
        saveas(601,filename_601);
        
         % Plot gradient step against time for weight 2
        figure(602)
        name=horzcat('./data/',FolderName,'/Video/Delta_W2.avi');
        vid = VideoWriter(name);
        open(vid);
        v=1:1:t3;
        for i=1:1:Ni
             for j=1:1:Ni
                  if i ~=j  
                hold on
                plot(v,delta_W_2_overtime(t1+1:t2,i,j),'LineWidth',2,'Color','b');
                y_label = ylabel('Delta W2'); 
                set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
                x_label = xlabel('Epochs'); 
                set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
                frame = getframe(gcf);
                writeVideo(vid,frame);
                pause(0.1);
                  end
             end
        end
        close(vid)
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Delta W2'); 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
        x_label = xlabel('Epochs'); 
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        name=horzcat('./data/',FolderName,'/Figures/Delta_W2.png');
        filename_602  = sprintf(name);
        saveas(602,filename_602);

        % Plot gradient step overtime for weight 1 as colourmap
        fig_num=603;
        size_x=Ni;
        delta_W_1(:,:,:)=delta_W_1_overtime(t1:t2,1:size_x(1),1:size_x(1));
        name_vid=horzcat('./data/',FolderName,'/Video/Delta_W1_img.avi');
        name_plot='Delta W1';
        name_fig=horzcat('./data/',FolderName,'/Figures/Delta_W1_img.png');
        y_label='Detla Input';
        x_label='Delta Hidden';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,delta_W_1,line,x_label,y_label)

        % Plot gradient step overtime for weight 2 as colourmap
        fig_num=604;
        delta_W_2(:,:,:)=delta_W_2_overtime(t1:t2,1:size_x(1),1:size_x(1));
        name_vid=horzcat('./data/',FolderName,'/Video/Delta_W2_img.avi');
        name_plot='Delta W2';
        name_fig=horzcat('./data/',FolderName,'/Figures/Delta_W2_img.png');
        x_label='Detla Hidden';
        y_label='Detla Output';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,delta_W_2,line,x_label,y_label)

        % Plot projections of gradient step of weight 1 into the eigen-basis 
        % of the first task overtime (as colourmap)
        fig_num=605;
        k=0;
        for t=line
            k=k+1;
            delta_l(:,:)= delta_W_1_overtime(t1+t,:,:);
            delta_plot(k,:,:)=U_task_1'*delta_l*Vv_task_1;
        end 
        name_vid=horzcat('./data/',FolderName,'/Video/Projecction_Delta_W1_SVD_task1_img.avi');
        name_plot='Projected Delta W1';
        name_fig=horzcat('./data/',FolderName,'/Figures/Projection_Delta_W1_SVD_task1_img.png');
        x_label='Detla Modes';
        y_label='Delta Modes';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,delta_plot,line_matrix,x_label,y_label)
        
        % Plot projections of gradient step of weight 2 into the eigen-basis 
        % of the first task overtime (as colourmap)
        fig_num=606;
        k=0;
        for t=line
            k=k+1;
            delta_l(:,:)= delta_W_2_overtime(t1+t,:,:);
            delta_plot(k,:,:)=U_task_1_2'*delta_l*Vv_task_1_2;
        end
        name_vid=horzcat('./data/',FolderName,'/Video/Projection_Delta_W2_SVD_task1_img.avi');
        name_plot='Projected Delta W2';
        name_fig=horzcat('./data/',FolderName,'/Figures/Projection_Delta_W2_SVD_task1_img.png');
        x_label='Detla Modes';
        y_label='Delta Modes';
        plot_rates_maps(fig_num,name_plot,name_vid,name_fig,delta_plot,line_matrix,x_label,y_label)
        
        % Plot projections of gradient step of weight 1 into the eigen-basis 
        % of the first task. (Values against Epochs) (Values against Epochs)
        figure(607); 
        delta_l_ij=zeros(t3,Ni,Ni);
        for t=1:1:t3
             delta_w1_l(:,:)= delta_W_1_overtime(t1+t,:,:);
             delta_plot=U_task_1'*delta_w1_l*Vv_task_1;
               for i=1:1:Ni
                    for j=1:1:Ni
                         delta_l_ij(t,i,j)=delta_plot(i,j);
                   end
               end 
        end 
        name=horzcat('./data/',FolderName,'/Video/Projection_Delta_W1_SVD_task1_lines.avi');
        vid = VideoWriter(name);
        open(vid);
        v=1:1:t3;
        for i=1:1:Ni
             for j=1:1:Ni
                hold on
                if i==j
                    plot(v,delta_l_ij(:,i,j),'LineWidth',2,'Color','r')
                else
                plot(v,delta_l_ij(:,i,j),'LineWidth',2,'Color','b')
                end
                frame = getframe(gcf);
                writeVideo(vid,frame);
             end
        end
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Projection  Delta of W1 on to task1 Eigen-Basis '); 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        legend('Diagonal','Off-Diagonal')
        close(vid)
        name=horzcat('./data/',FolderName,'/Figures/Projection_Delta_W1_SVD_task1_lines.png');
        filename_608  = sprintf(name);
        saveas(607,filename_607); 


        % Plot projections of gradient step of weight 2 into the eigen-basis 
        % of the first task. (Values against Epochs)
        figure(608); 
        delta_l_ij=zeros(t3,Ni,Ni);
        for t=1:1:t3
             delta_w2_l(:,:)= delta_W_2_overtime(t1+t,:,:);
             delta_plot=U_task_1_2'*delta_w2_l*Vv_task_1_2;
               for i=1:1:Ni
                    for j=1:1:Ni
                         delta_l_ij(t,i,j)=delta_plot(i,j);
                   end
               end 
        end 
        name=horzcat('./data/',FolderName,'/Video/Projection_Delta_W2_SVD_task1_lines.avi');
        vid = VideoWriter(name);
        open(vid);
        v=1:1:t3;
        for i=1:1:Ni
             for j=1:1:Ni
                hold on
                if i==j
                    plot(v,delta_l_ij(:,i,j),'LineWidth',2,'Color','r')
                else
                plot(v,delta_l_ij(:,i,j),'LineWidth',2,'Color','b')
                end
                frame = getframe(gcf);
                writeVideo(vid,frame);
             end
        end
        tick2 =  num2str(get(gca,'YTick')','%g');
        tick1 =  num2str(get(gca,'XTick')','%g');
        set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
        set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
        y_label = ylabel('Projection Delta of W2 on to task1 Eigen-Basis'); 
        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
        legend('Diagonal','Off-Diagonal')
        close(vid)
        name=horzcat('./data/',FolderName,'/Figures/Projection_Delta_W2_SVD_task1_lines.png');
        filename_608  = sprintf(name);
        saveas(608,filename_608);    
           
    end 
end