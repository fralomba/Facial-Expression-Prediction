function visualize(defNeutral,defShape)
    subplot(1,2,1)
    plot_mesh(defNeutral,compute_delaunay(defNeutral));
    title('Average Model')
    subplot(1,2,2)
    plot_mesh(defShape,compute_delaunay(defShape));
    title('Deformed Model')
end

