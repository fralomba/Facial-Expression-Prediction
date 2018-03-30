function done = deform_and_visualize(def_neutral, def_vs, def_v2, def_v3, def_v4, def_v5, expr, tec, filename, index_texture)
    
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

    x0=10;
    y0=10;
    width=1500;
    height=400;
    set(figure1,'units','points','position',[100,800,width,height])

    subplot(1,6,1)
    defNeutral = deform_3D_shape_fast(avgModel',Components, def_neutral');
    plot_mesh(defNeutral,compute_delaunay(defNeutral), options);
    title(strcat(expr, " model with mean"))

    subplot(1,6,2)
    defShape = deform_3D_shape_fast(avgModel',Components, def_vs');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with median"))

    subplot(1,6,3)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v2');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with mode"))

    subplot(1,6,4)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v3');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with linear regression"))

    subplot(1,6,5)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v4');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with svr regression"))

    subplot(1,6,6)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v5');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model with NN regression"))

    %saveas(figure1,filename);

    done = 'done';
end

