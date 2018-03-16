%% Example code to deform the average 3D model
%  by means of a dictionary of deformation components and a set of deformation coefficients
%  Load data
if ~exist('def_coeff','var')
    load data/avgModel.mat
    load data/processed_ck.mat
    load data/components_DL_300.mat
    addpath(genpath('toolbox_general/'))
    addpath(genpath('toolbox_graph/'))
end

% Params
index = 12;

% Select an arbitrary coefficients vector.
% The coefficients are pre-computed by fitting the average model on a 2D
% image
def_v = def_coeff(:,index);

% Deform the average model ( or any aritrary model ) by summing to the
% average model a linear combination of the dictionary elements
defShape = deform_3D_shape_fast(avgModel',Components, def_v);

% Visualization
subplot(1,2,1)
plot_mesh(avgModel,compute_delaunay(avgModel));
title('Average Model')
subplot(1,2,2)
plot_mesh(defShape,compute_delaunay(defShape));
title('Deformed Model')

