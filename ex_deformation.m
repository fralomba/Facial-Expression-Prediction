%% Example code to deform the average 3D model
%  by means of a dictionary of deformation components and a set of deformation coefficients
%  Load data
if ~exist('def_coeff','var')
    load data/avgModel.mat
    load data/processed_ck.mat
    load data/components_DL_300.mat
    load data/processed_ck_colors.mat
    addpath(genpath('toolbox_general/'))
    addpath(genpath('toolbox_graph/'))
end

% Params


use_avgModel = false;
render_texture = true;
index_shape = 247;
index_texture = index_shape;
index_coeff = index_shape + 1;



% Select an arbitrary coefficients vector.
% The coefficients are pre-computed by fitting the average model on a 2D
% image


% Select an arbitrary shape.
% The coefficients are pre-computed by fitting the average model on a 2D
% image

shape_neutral = def_shapes(:,index_shape);
shape_neutral = reshape(shape_neutral,length(shape_neutral)/3,3);

if use_avgModel
    shape_base = avgModel;
    def_v = def_coeff(:,index_coeff);
else
    shape_base = def_shapes(:,index_shape);
    shape_base = reshape(shape_base,length(shape_base)/3,3);
    def_v = def_coeff(:,index_coeff);
end

% Deform the average model ( or any aritrary model ) by summing to the
% average model a linear combination of the dictionary elements
defShape = deform_3D_shape_fast(shape_base',Components, def_v);

% Get arbitrary texture
texture = colors_all(:,:,index_texture); 
if render_texture
    options.face_vertex_color = texture;
else
    options.face_vertex_color = [];
end

% Visualization
figure;
subplot(1,2,1)
plot_mesh(shape_neutral,compute_delaunay(shape_neutral),options);
title('Base Model')
subplot(1,2,2)
plot_mesh(defShape,compute_delaunay(defShape),options);
title('Deformed Model')


