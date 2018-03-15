if ~exist('def_coeff','var')
    load data/avgModel.mat
    load data/processed_ck.mat
    load data/components_DL_300.mat
    addpath(genpath('toolbox_general/'))
    addpath(genpath('toolbox_graph/'))
end

idx = find(contains(labels_expr,'neutral'));
def_neutral = def_coeff(:,idx);

n_examples = 6;
expr = "sadness";
technique = "mode";

indexes = randi(300, n_examples,1);

for i=1:n_examples
    figure;
    def_v = def_neutral(:,indexes(i)) + prediction(expr, technique);

    defShape = deform_3D_shape_fast(avgModel',Components, def_v);
    defNeutral = deform_3D_shape_fast(avgModel',Components, def_neutral(:,indexes(i)));

    visualize(defNeutral, defShape); 
end

