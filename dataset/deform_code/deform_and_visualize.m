function done = deform_and_visualize(def_neutral,def_v)
    
    figure;
    
    if ~exist('def_coeff','var')
        load data/avgModel.mat
        load data/processed_ck.mat
        load data/components_DL_300.mat
        addpath(genpath('toolbox_general/'))
        addpath(genpath('toolbox_graph/'))
    end
    
    defShape = deform_3D_shape_fast(avgModel',Components, def_v');
    defNeutral = deform_3D_shape_fast(avgModel',Components, def_neutral');

    subplot(1,2,1)
    plot_mesh(defNeutral,compute_delaunay(defNeutral));
    title('Neutral Model')
    subplot(1,2,2)
    plot_mesh(defShape,compute_delaunay(defShape));
    title('Expression Model')

    done = 'done';
end

