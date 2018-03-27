%% Load data
if ~exist('dataset','var')
    load('data/dataset.mat')
    load('data/landmarks.mat')
    load('data/emotions.mat')
    load('data/components_DL_300.mat')
    Components_res = reshape_components(Components);
    Weights = scaledata(Weights,0.1,1);
    addpath(genpath('toolbox_graph/'))
    addpath(genpath('miscellaneous/'))
end
load('data/avgModel_NE.mat')
%% Set Params
rounds_opt = 1;             % 3DMM optimization iterations
lambda_opt = 0.01;          % 3DMM regularization
remove_boundary = false;    % Remove boundary facial landmarks
debug1 = false;

%% Set structures
labels_id = [];
labels_expr = {};
def_coeff = [];
def_shapes = [];
colors_all = [];

%% Loop over test identities
for i=1:size(dataset,2)
    % Loop over sequences
    for j = 1 : size(emotions(i).seq,2)
        % some sequences do not have expression label, so skip them
        if ~isempty(emotions(i).seq(j).data)
            % Loop over frames
            for k = [1 size(dataset(i).seq(j).data,2)]              % First frame: Neutral - Last Frame: Expressive
                
                im = dataset(i).seq(j).data(k).img;
                lm = landmarks(i).seq(j).data(k).landImage;
                
                % Adjust landmarks configurations
                lm(65,:) = [];
                lm(61,:) = [];
                
                if remove_boundary
                    lm(1:17,:) = [];
                end
                
                labels_id = [labels_id; i];
                if k == 1   % Neutral
                    labels_expr = [labels_expr; 'neutral'];
                    base_model = avgModel;
                else        % Expressive
                    labels_expr = [labels_expr; emotions(i).seq(j).data];
                    base_model = defShape;
                end
                
                %% 3DMM Fitting
                % Pose estimation
                [A, S, R, t] = estimatePose(base_model(idxLandmarks3D,:),lm);
                % Landmark projection
                lm3D_p = getProjectedVertex(base_model(idxLandmarks3D,:),S,R,t)';
                % Alpha coefficients estimation
                alpha  = alphaEstimation_fast(lm,lm3D_p,Components_res,idxLandmarks3D,S,R,t,Weights,lambda_opt);
                % Deform the 3D average Model
                defShape = deform_3D_shape_fast(base_model',Components,alpha)';
                % Save data
                def_coeff = [def_coeff, alpha];
                def_shapes = [def_shapes, defShape(:)];
                % Get Colors
                if size(im,3) == 1
                    im(:,:,1) = im;
                    im(:,:,2) = im(:,:,1);
                    im(:,:,3) = im(:,:,1);
                end
                proj3dmm = getProjectedVertex(defShape,S,R,t)';
                colors = getRGBtexture(round(proj3dmm),im);
                
                colors_all = cat(3,colors_all,colors);
                
                if debug1
                    % Deform the 3D average Model from the average model
                    options.face_vertex_color= colors;
                    defShape_o = deform_3D_shape_fast(avgModel',Components,alpha)';
                    subplot(1,3,1)
                    plot_landmarks(im,lm,'r.',0);
                    subplot(1,3,2)
                    plot_mesh(defShape,compute_delaunay(defShape),options);
                    title('From the subject model')
                    subplot(1,3,3)
                    plot_mesh(defShape_o,compute_delaunay(defShape_o),options);
                    title('From the average model')
                    pause
                    clf
                end
                
            end
            clear defShape defShape_o;
        end
    end
    disp(['Subjects ' num2str(i) ' done!'])
end






