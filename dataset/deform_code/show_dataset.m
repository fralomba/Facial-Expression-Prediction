if ~exist('dataset','var')
    load('data/dataset.mat')
    load('data/landmarks.mat')
end

for i=1:size(dataset,2)
    
    for j = 1 : size(dataset(i).seq,2)
        
        for k = size(dataset(i).seq(j).data,2)
            
            test_image = dataset(i).seq(j).data(k).img;
            test_landmarks = landmarks(i).seq(j).data(k).landImage;
            
            plot_landmarks(test_image, test_landmarks,'r.',1 )
            pause()
            clf
            
        end
    end
end