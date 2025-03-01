%{
    2D Gaussian splatting fits a single image.
%}


clc
clear

addpath(genpath("optimizers"));
addpath(genpath("mxsplat"));

%% load an image
img = single(imread("cameraman.tif"))/255;
img = mean(img,3);
img_GT = img;

img_sz = size(img);
get_GPU = gpuDevice();

num_Gaussian = 10000;

GSH_model = dlnetwork([
                        inputLayer(([1,num_Gaussian]),'UU');
                        gSplat_Hologram("device", get_GPU,...
                                        "img_sz", img_sz);
                    ]);


%% Main training loop
iter = 0;
iter_max = 4800;
optimizer = optimizer_AdaBelief();
lr = 0.006;

while iter < iter_max
    iter = iter + 1;
    
    tic;
    % forward the 2D Gaussian and get gradients 
    [loss,grad,img_out] = dlfeval(@model_loss, GSH_model, img_GT); 
    tt = toc;  
    
    % learning for parameters
    GSH_model.Learnables = optimizer.step(GSH_model.Learnables,... 
                                          grad,...
                                          iter,...
                                          lr);

    fprintf("at %d-iter, 2d gs takes: %4.5f, " + ...
            "loss: %4.4f, remain gs: %d, \n",iter, tt, loss, num_Gaussian);

    %Adaptive Gaussian density
    if mod(iter,10) == 0
        [GSH_model, optimizer, num_Gaussian] = ...
                            adaptive_Gaussian(GSH_model,...
                                              optimizer,...
                                              grad);
    end
    
    if mod(iter,5) == 1
        figure(122);
        imshow(img_out{1},[]);
        drawnow;
    end

    % learning rate decay
    if mod(iter,120) == 0 
        lr = max(lr * 0.7,0.0001);
    end
end

save('results/test_non.mat','GSH_model','img_out');

rmpath(genpath("optimizers"));
rmpath(genpath("gsplat"));
reset(gpuDevice());


% helper functions
function [loss,grad,img_out] = model_loss(model,img_GT)
    canvas = model.forward(dlarray(1,'U'));

    loss = loss_fun.l2_loss(canvas, img_GT,'mean');

    grad = dlgradient(loss, model.Learnables);

    img_out{1} = extractdata(stripdims(canvas));
end