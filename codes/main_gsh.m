clc
clear

reset(gpuDevice());

img = single(imread("cameraman.tif"))/255;
img = mean(img,3);
img = imresize(img,[256,256]);
img = padarray(img,[128,128],'both');

upsam = 1;

img_sz = size(img);
get_GPU = gpuDevice();

% init Gaussian splatting hologram
num_Gaussian = 5000;
GSH_model = dlnetwork([
                        inputLayer(([1,num_Gaussian]),'UU');
                        mxsplat.gSplat_Hologram("device", get_GPU,...
                                        "img_sz", img_sz);
                    ]);

diffractor_u = mxsplat.diffractor("pix_size", 2.7,...
                          "lambda",   0.532,...
                          "toz",      6000,...
                          "img_sz",   img_sz);

prop = diffractor_u.set_propagation();
GT = img;

img = exp(1i*pi*img);
img_GT = abs(prop(img));
img_GT = dlarray(gpuArray(img_GT));

iter = 0;
iter_max = 4800;
optimizer = optimizers.Adam(0.9,0.999,1e-15);
lr = 0.01;

%% Main training loop
while iter < iter_max
    iter = iter + 1;
    
    tic;
    [loss,grad,img_out] = dlfeval(@model_loss, GSH_model, img_GT, prop);
    tt = toc;  

    GSH_model.Learnables = optimizer.step(GSH_model.Learnables,...
                                          grad,...
                                          iter,...
                                          lr);

    %Adaptive Gaussian density
    if (mod(iter,10) == 0) && (num_Gaussian < 50000)
        [GSH_model, optimizer, num_Gaussian] = ...
                            adaptive_Gaussian(GSH_model,...
                                              optimizer,...
                                              grad);
    end

    fprintf("at %d-iter, 2d gs takes: %4.5f, " + ...
            "loss: %4.4f, remain gs: %d, \n",iter, tt, loss, num_Gaussian);

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

reset(gpuDevice());


% helper functions
function [loss,grad,img_out] = model_loss(model,img_GT,prop)
    canvas = model.forward(dlarray(1,'U'));

    u0 = exp(1i*2*pi*canvas);

    img_PR = abs(prop(stripdims(u0)));
    
    loss = loss_fun.l2_loss(img_PR, img_GT,'mean');

    grad = dlgradient(loss, model.Learnables);

    img_out{1} = extractdata(stripdims(canvas));
    img_out{2} = extractdata(img_PR);
end
