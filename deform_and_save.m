function done = deform_and_visualize(def_neutral, def_vs1, def_vs2, def_v2, def_v3, def_v4, def_v5, expr, tec, filename, index_texture)
    
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
    height=200;
    set(figure1,'units','points','position',[0,800,width,height])

    subplot(1,7,1)
    defNeutral = deform_3D_shape_fast(avgModel',Components, def_neutral');
    plot_mesh(defNeutral,compute_delaunay(defNeutral), options);
    title("neutral model")

    subplot(1,7,2)
    defShape = deform_3D_shape_fast(avgModel',Components, def_vs1');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " with mean"))

    subplot(1,7,3)
    defShape = deform_3D_shape_fast(avgModel',Components, def_vs2');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " with median"))

    subplot(1,7,4)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v2');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " with mode"))

    subplot(1,7,5)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v3');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " with linear regression"))

    subplot(1,7,6)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v4');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " with svr regression"))

    subplot(1,7,7)
    defShape = deform_3D_shape_fast(avgModel',Components, def_v5');
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " with NN regression"))

    mkdir(char(strcat("results/", expr)));

    saveas(figure1, strcat("results/", expr, "/", filename));

    done = 'done';
end

