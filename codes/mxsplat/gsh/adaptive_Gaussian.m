function [gsh_model,optimizer,len] = adaptive_Gaussian(gsh_model,optimizer,grad)

   
    params = gsh_model.Learnables;
    mom1 = optimizer.mom1;
    mom2 = optimizer.mom2;
    
    criteria = sin(params.Value{5}).^2 .* ...
        sqrt( exp(params.Value{3}(1,:)) .* exp(params.Value{3}(2,:)) );

    idx = (criteria < 0.0003);


    for con = 1:5
        params.Value{con}(:,idx) = [];
        mom1.Value{con}(:,idx) = [];
        mom2.Value{con}(:,idx) = [];
        grad.Value{con}(:,idx) = [];
    end


    



    grad_size = grad.Value{3};
    grad_amp  = sqrt(grad_size(1,:).^2 + grad_size(2,:).^2);

    idx = grad_amp > 5e-6;
   
    c1 = grad_size(1,idx) ./ (grad_amp(:,idx) + 1e-4);
    c2 = grad_size(2,idx) ./ (grad_amp(:,idx) + 1e-4);


    temp1 = params.Value{1}(1,idx);
    temp2 = params.Value{1}(2,idx);
    temp1 = temp1 + 1.3*c1.*randn(size(temp1));
    temp2 = temp2 + 1.3*c2.*randn(size(temp2));

    temp3 = params.Value{3}(:,idx) * 0.8;
    temp4 = params.Value{4}(:,idx) * 0.5;
    temp5 = params.Value{5}(:,idx) * 0.8;

    params.Value{1} = cat(2,params.Value{1},[temp1;temp2]);
    params.Value{2} = cat(2,params.Value{2},params.Value{2}(:,idx));
    params.Value{3} = cat(2,params.Value{3},temp3);
    params.Value{4} = cat(2,params.Value{4},temp4);
    params.Value{5} = cat(2,params.Value{5},temp5);

    for con = 1:5
        mom1.Value{con} = cat(2,mom1.Value{con},mom1.Value{con}(:,idx));
        mom2.Value{con} = cat(2,mom2.Value{con},mom2.Value{con}(:,idx));
    end


    len = size(params.Value{5});
    len = len(2);

    gsh_model.Learnables = params;
    optimizer.mom1 = mom1;
    optimizer.mom2 = mom2;
end