function done = deform_and_visualize_one(def_v, expr, tec, path, filename, index_texture)
    
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


    defShape = deform_3D_shape_fast(avgModel',Components, def_v');
    x0=10;
    y0=10;
    width=550;
    height=600;
    set(figure1,'units','points','position',[x0,y0,width,height])
    plot_mesh(defShape,compute_delaunay(defShape), options);
    title(strcat(expr, " model ", tec))

    mkdir(path)

    saveas(figure1, strcat(path, "/", filename));

    done = 'done';
end

