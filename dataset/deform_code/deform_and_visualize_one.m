function done = deform_and_visualize(def_neutral, def_v1, expr, tec, filename, index_texture)
    
    figure1 = figure;
    
    if ~exist('def_coeff','var')
        load data/avgModel.mat
        load data/processed_ck.mat
        load data/components_DL_300.mat
        load data/processed_ck_colors.mat
        addpath(genpath('toolbox_general/'))
        addpath(genpath('toolbox_graph/'))
    end
    

    defNeutral = deform_3D_shape_fast(avgModel',Components, def_neutral');
    
    texture = colors_all(:,:,index_texture); 
    options.face_vertex_color = texture;

    subplot(1,2,1)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v1');
    plot_mesh(defNeutral,compute_delaunay(defNeutral), options);
    title('neutral model')

    subplot(1,2,2)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v1');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title('neutral model')

    done = 'done';
end

