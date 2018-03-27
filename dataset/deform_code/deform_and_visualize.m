function done = deform_and_visualize(def_neutral, def_v2, def_v3, def_v4, def_v5, expr, tec, filename, index_texture)
    
    figure1 = figure;
    
    if ~exist('def_coeff','var')
        load data/avgModel.mat
        load data/processed_ck.mat
        load data/components_DL_300.mat
        load data/processed_ck_colors.mat
        addpath(genpath('toolbox_general/'))
        addpath(genpath('toolbox_graph/'))
    end

    texture = colors_all(:,:,index_texture); 
    options.face_vertex_color = texture;

    subplot(1,5,1)
    defNeutral = deform_3D_shape_fast(avgModel',Components, def_neutral');
    plot_mesh(defNeutral,compute_delaunay(defNeutral), options);
    title('neutral model')

    subplot(1,5,2)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v2');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with mode"))

    subplot(1,5,3)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v3');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with linear regression"))

    subplot(1,5,4)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v4');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with svr regression"))

    subplot(1,5,5)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v5');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with NN regression"))

    saveas(figure1,filename);

    done = 'done';
end

