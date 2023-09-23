function plotweightprojection=plot_weight_projection(W_1_overtime,W_2_overtime,U_task_1,U_task_1_2,Vv_task_1,Vv_task_1_2,Ss_task_1,Ss_task_1_2,Uu,Uu_2,Vv,Vv_2,Ss,Ss_2,t1,t2,FolderName,plot_weight_projections_n,line,line_matrix,Vv_task_3,U_task_1_3)
% Plot projections of the weights using the SVD of the weights at the end of the first task
      % Plot Projection of W2 onto SVD basis of W2 at the end of task 1
            % During task 2
                   % Plot Projection of W2 onto SVD basis of W2 at the end of task 1
            % During task 2
             
    if plot_weight_projections_n==1
        
        
            % Plot Projection of W1 onto SVD basis of W1 at the end of task 1
            % as colormap
            % During the 2nd task
           
            % Plot Projection of W2 onto SVD basis of W2 at the end of task 1
            % During task 2
            t3=t2-t1;
            figure(506); 
            size_W1=size(W_1_overtime);
            Ni=size_W1(2);
            Prosjection_W_2_ij=zeros(t3,Ni,Ni);
            for t=1:1:t3
                  W_2_t(:,:)=W_2_overtime(t1+t,:,:);
                  Projection_W_2= U_task_1_2'*W_2_t*Vv_task_1_2;
                   for i=1:1:Ni
                        for j=1:1:Ni
                             Projection_W_2_ij(t,i,j)=Projection_W_2(i,j);
                       end
                   end 
            end 
            name=horzcat('./data/',FolderName,'/Video/Full_projection_W2_SVD_task1_lines.avi');
            vid = VideoWriter(name);
            open(vid);
            v=1:1:t3;
            for i=1:1:Ni
                 for j=1:1:Ni
                    hold on
                    if i==j
                        plot(v,Projection_W_2_ij(:,i,j),'LineWidth',2,'Color','#BD2230')
                    else
                    plot(v,Projection_W_2_ij(:,i,j),'LineWidth',2,'Color','#276FB4')
                    end

                    frame = getframe(gcf);
                    writeVideo(vid,frame);
                 end
            end
            xline(0,'Color','k','LineStyle','--','LineWidth',2);
            tick2 =  num2str(get(gca,'YTick')','%g');
            tick1 =  num2str(get(gca,'XTick')','%g');
            set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
            set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
            y_label = ylabel('Projection of W2 on to task1 Eigen-Basis '); 
            set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
            x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
            set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman')       
            legend('Diagonal','Off-Diagonal')
            close(vid)
            set(0, 'DefaultFigureRenderer', 'painters');
            name=horzcat('./data/',FolderName,'/Figures/Full_projection_W2_SVD_task1_lines.svg');
            filename_506  = sprintf(name);
            saveas(506,filename_506);

            % Plot Projection of W2 onto SVD basis of W2 at the end of task 1
            % Off-Diagonal terms
            % During task 2
            figure(508)
            name=horzcat('./data/',FolderName,'/Video/Full_projection_W2_SVD_task1_offdiagonal.avi');
            vid = VideoWriter(name);
            open(vid);
            v=1:1:t3;
            for i=1:1:Ni
                 for j=1:1:Ni
                    if i~=j 
                        hold on
                        plot(v,Projection_W_2_ij(:,i,j),'LineWidth',2,'Color','#276FB4')
                        x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
                        set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
                        y_label = ylabel('Mode Value'); %or h=get(gca,'xlabel')
                        set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
                        frame = getframe(gcf);
                        writeVideo(vid,frame);
                        pause(0.1);
                    end 
                 end
            end
            close(vid)
            xline(0,'Color','k','LineStyle','--','LineWidth',2);
            tick2 =  num2str(get(gca,'YTick')','%g');
            tick1 =  num2str(get(gca,'XTick')','%g');
            set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
            set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
            y_label = ylabel('Off-diag Projection of W2 on to task1 Eigen-Basis '); 
            set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
            x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
            set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
            name=horzcat('./data/',FolderName,'/Figures/Full_projection_W2_SVD_task1_offdiagonal.svg');
            filename_508  = sprintf(name);
            saveas(508,filename_508);


            % Plot Projection of W1 onto SVD basis of W1 at the end of task 1
            % During task 2
            figure(509); 
            Projection_W_1_ij=zeros(t3,Ni,Ni);
            for t=1:1:t3
                 W_1_t(:,:)=W_1_overtime(t1+t,:,:);
                 Projection_W_1=U_task_1'*W_1_t*Vv_task_1;
                 for i=1:1:Ni
                      for j=1:1:Ni
                             Projection_W_1_ij(t,i,j)=Projection_W_1(i,j);

                      end
                 end 
            end 
            name=horzcat('./data/',FolderName,'/Video/Full_projection_W1_SVD_task1_line.avi');
            vid = VideoWriter(name);
            open(vid);
            v=1:1:t3;
            for i=1:1:Ni
                 for j=1:1:Ni
                    hold on
                    if i==j
                    plot(v,Projection_W_1_ij(:,i,j),'LineWidth',2,'Color','#BD2230');
                    else
                    hold on
                    plot(v,Projection_W_1_ij(:,i,j),'LineWidth',2,'Color','#276FB4');
                    end
                    x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
                    set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
                    y_label = ylabel('Mode Value'); %or h=get(gca,'xlabel')
                    set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
                    frame = getframe(gcf);
                    writeVideo(vid,frame);
                    pause(0.1);
                 end
            end
            legend('Diagonal','Off-Diagonal')
            close(vid)
            xline(0,'Color','k','LineStyle','--','LineWidth',2);
            tick2 =  num2str(get(gca,'YTick')','%g');
            tick1 =  num2str(get(gca,'XTick')','%g');
            set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
            set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
            y_label = ylabel('Projection of W1 on to task1 Eigen-Basis '); 
            set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
            x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
            set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
            set(0, 'DefaultFigureRenderer', 'painters');
            name=horzcat('./data/',FolderName,'/Figures/Full_projection_W1_SVD_task1_line.svg');
            filename_509  = sprintf(name);
            saveas(509,filename_509);


            % Plot Projection of W1 onto SVD basis of W1 at the end of task 1
            % Off-Diagonal terms
            % During task 2
            
            figure(510);
            name=horzcat('./data/',FolderName,'/Video/Full_projection_W1_SVD_task1_off_diagonal.avi');
            vid = VideoWriter(name);
            open(vid);
            v=1:1:t3;
            for i=1:1:Ni
                 for j=1:1:Ni
                      if i ~=j  
                    hold on
                    plot(v,Projection_W_1_ij(:,i,j),'LineWidth',2,'Color','#276FB4');
                    x_label = xlabel('Epochs'); %or h=get(gca,'xlabel')
                    set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
                    y_label = ylabel('Mode Value'); %or h=get(gca,'xlabel')
                    set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
                    frame = getframe(gcf);
                    writeVideo(vid,frame);
                    pause(0.1);
                      end
                 end
            end
            close(vid)
            xline(0,'Color','k','LineStyle','--','LineWidth',2);
            tick2 =  num2str(get(gca,'YTick')','%g');
            tick1 =  num2str(get(gca,'XTick')','%g');
            set(gca,'XTickLabel',tick1,'FontName','Times','fontsize',14)
            set(gca,'YTickLabel',tick2,'FontName','Times','fontsize',14)
            y_label = ylabel('Off-diagonal Projection of W1 on to task1 EigenBasis '); 
            set(y_label, 'FontSize', 20,'FontName' , 'Times New Roman')
            x_label = xlabel('Epochs'); 
            set(x_label, 'FontSize', 20,'FontName' , 'Times New Roman') 
            name=horzcat('./data/',FolderName,'/Figures/Full_projection_W1_SVD_task1_off_diagonal.svg');
            filename_510  = sprintf(name);
            saveas(510,filename_510);
            
            fig_num=501;
            k=0;
            for t=line
                k=k+1;
                W_1_t(:,:)=W_1_overtime(t1-2+t,:,:);
                Projection_W_1_t(k,:,:)=U_task_1'*W_1_t*Vv_task_1;
            end 
            name_plot='Projection W1';
            name_vid=horzcat('./data/',FolderName,'/Video/full_Projection_W1_SVD_task1.avi');
            name_fig=horzcat('./data/',FolderName,'/Figures/full_Projection_W1_SVD_task1.svg');
            x_label='Modes';
            y_label='Modes';
            plot_rates_maps(fig_num,name_plot,name_vid,name_fig,Projection_W_1_t,line_matrix,x_label,y_label)

            % Plot Projection of W2 onto SVD basis of W2 at the end of task 1
            % as colormap
            % During the 2nd task
            fig_num=502;
            k=0;
            for t=line
                k=k+1;
                W_2_t(:,:)=W_2_overtime(t1-2+t,:,:);
                Projection_W_2_t(k,:,:)=U_task_1_2'*W_2_t*Vv_task_1_2;
            end 
            name_plot='Projection W2';
            name_vid=horzcat('./data/',FolderName,'/Video/full_Projection_W2_SVD_task1.avi');
            name_fig=horzcat('./data/',FolderName,'/Figures/full_Projection_W2_SVD_task1.svg');
            x_label='Modes';
            y_label='Modes';
            plot_rates_maps(fig_num,name_plot,name_vid,name_fig,Projection_W_2_t,line_matrix,x_label,y_label);

            % Projection of the eigen-matrices of W1/W2 during the 2nd task
            % on to their respective eigenbasis at the end of first task
            fig_num=503;
            k=0;
            for i=line
                k=k+1;
                Uu_t(:,:)=Uu(t1-1+i,:,:);
                Proj_U_w_1(k,:,:)=U_task_1'*Uu_t;

                Vv_t(:,:)=Vv(t1+1+i,:,:);
                Proj_V_w_1(k,:,:)=Vv_task_1'*Vv_t;

                Uu_t(:,:)=Uu_2(t1-1+i,:,:);
                Proj_U_w_2(k,:,:)=U_task_1_2'*Uu_t;

                Vv_t(:,:)=Vv_2(t1-1+i,:,:);
                Proj_V_w_2(k,:,:)=Vv_task_1_2'*Vv_t;
            end 
            x_label='Modes';
            y_label='Modes';
            name_vid=horzcat('./data/',FolderName,'/Video/Projection_UU_SS_VV.avi');
            name_fig=horzcat('./data/',FolderName,'/Figures/Projections_UU_SS_VV.svg');
            plot_rates_maps_4(fig_num,name_vid,name_fig,Proj_U_w_2,Proj_V_w_2,Proj_V_w_1,Proj_U_w_1,line_matrix,x_label,y_label)

            % Plot of partial projection of W1 during task 2
            % onto SVD basis of W1 at the end of task 1. 
            % Plot comparaison with SVD overtime during task 2
            fig_num=504;
            k=0;
            for i=line
                k=k+1;

                W_1_t(:,:)=W_1_overtime(t1-1+i,:,:);
                Projection_W_1_2(k,:,:)= W_1_t*Vv_task_1;

                Ss_t(:)=Ss(t1-1+i,:);
                u_t(:,:)=Uu(t1-2+i,:,:);
                h_features(k,:,:)=(u_t*diag(Ss_t));

                W_1_t(:,:)=W_1_overtime(t1-1+i,:,:);
                Projection_W_1(k,:,:)=U_task_1'*W_1_t;

                Ss_t(:)=Ss(t1-1+i,:);
                Vv_t(:,:)=Vv(t1-1+i,:,:);
                h_items(k,:,:)=diag(Ss_t)*Vv_t';
            end 
            name_fig=horzcat('./data/',FolderName,'/Figures/Partial_Projection_W1_SVD_task1.svg');
            name_vid=horzcat('./data/',FolderName,'/Video/Partial_Projection_W1_SVD_task1.avi');
            x_label='Modes';
            y_label='Modes';
            plot_rates_maps_4(fig_num,name_vid,name_fig,Projection_W_1_2,h_features,Projection_W_1,h_items,line_matrix,x_label,y_label)

            % Plot of partial projection of W2 during task 2
            % onto SVD basis of W2 at the end of task 1. 
            % Plot comparaison with SVD overtime during task 2
            fig_num=505;
            k=0;
            for i=line
                k=k+1;
                W_2_t(:,:)=W_2_overtime(t1-2+i,:,:);
                Projection_W_2(k,:,:)= W_2_t*Vv_task_1_2;

                Ss_2_t(:)=Ss_2(t1-2+i,:);
                u_t_2(:,:)=Uu_2(t1-2+i,:,:);
                h_features_2(k,:,:)=(u_t_2*diag(Ss_2_t));

                W_2_t(:,:)=W_2_overtime(t1-2+i,:,:);
                Projection_W_2_2(k,:,:)=U_task_1_2'*W_2_t;

                Ss_2_t(:)=Ss_2(t1-2+i,:);
                Vv_2_t(:,:)=Vv_2(t1-2+i,:,:);
                h_items_2(k,:,:)=diag(Ss_2_t)*Vv_2_t';

            end
            name_vid=horzcat('./data/',FolderName,'/Video/Partial_Projection_W2_SVD_task1.avi');
            name_fig=horzcat('./data/',FolderName,'/Figures/Partial_Projection_W2_SVD_task1.svg');
            x_label='Modes';
            y_label='Modes';
            plot_rates_maps_4(fig_num,name_vid,name_fig,Projection_W_2,h_features_2,Projection_W_2_2,h_items_2,line_matrix,x_label,y_label)


    end 
end

