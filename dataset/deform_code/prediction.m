function result = prediction(expr,technique)

    script_text = strcat("python3 ../../py/prediction.py '", expr, "' '", technique, "'");

    [status, result] = system(script_text);
    if status ~= 0
        disp("Wrong folder! Open deform_code folder")
        return
    end

    result = reshape(str2num(result)', 300, 1);
    
end


