if ~exist('dataset','var')
    load('data/dataset.mat')
    load('data/landmarks.mat')
end
figure
options.face_vertex_color = texture1;
for i=1:size(labels_expr,1)
    
    defShape = def_shapes(:,i);
    defShape = reshape(defShape,length(defShape)/3,3)';
    
    
    plot_mesh(defShape,compute_delaunay(defShape),options);
    disp([labels_expr{i} ' - ID: ' num2str(labels_id(i))])
    pause
    clf
end