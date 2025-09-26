classdef gSplat_Hologram < nnet.layer.Layer & nnet.layer.Formattable
    properties 
        num_gaussian;
        device;
        bd;
        img_sz
        
    end

    properties (Learnable)
        xy_pos;
        rotate;
        scales;
        colors;
        opacis;
    end

    methods 
        function layer = gSplat_Hologram(args)
            arguments
                args.device;
                args.img_sz = [256,256];
            end
            layer.Description = "GaussianSplatting";

            layer.img_sz = args.img_sz;
            layer.device = args.device;

            layer.NumInputs = 1;
            layer.NumOutputs = 1;  
        end

        function layer = initialize(layer,layout)
            
            layer.num_gaussian = layout.Size(2);
            num_points = layer.num_gaussian;
            
            if isempty(layer.xy_pos) % xy_pos
                layer.xy_pos = asin(init_rand([2,num_points]));
            end
            if isempty(layer.rotate) % rotate
                layer.rotate = asin(init_rand([1,num_points]));
            end
            if isempty(layer.scales) % scales
                layer.scales = 2*init_rand([2,num_points]);
            end
            if isempty(layer.colors) % colors
                layer.colors = init_rand([1,num_points]);
            end
            if isempty(layer.opacis) % opacis
                layer.opacis = init_rand([1,num_points]);
            end
        end
        
        function Y = predict(layer,X)  
            xys_get  = sin(layer.xy_pos);
            rot_get  = layer.rotate .* pi ./ 4;
            scl_get  = exp(layer.scales) + 0.01;
            col_get  = abs(layer.colors);
            % opa_get  = gpuArray(single(ones(size(sin(layer.opacis).^4))));
            opa_get  = sin(layer.opacis).^2;

            if ~(isgpuarray(xys_get))
                xys_get = gpuArray(xys_get);
                rot_get = gpuArray(rot_get);
                scl_get = gpuArray(scl_get);
                col_get = gpuArray(col_get);
                opa_get = gpuArray(opa_get);
            end
            wait(layer.device);

            % forward projection
            [   xys, ...
                depths, ...
                radii, ...
                conics, ...
                num_tiles_hit] = ...
                       project_2dgs_func(...
                                            xys_get, ...
                                            rot_get, ...
                                            scl_get, ...
                                            layer.img_sz);
            wait(layer.device);

            % forward rasterization
            canvas = rasterize_2dgs_func(...
                                            xys, ...
                                            depths, ...
                                            radii, ...
                                            conics, ...
                                            num_tiles_hit,...
                                            col_get,...
                                            opa_get,...
                                            layer.img_sz);        
            wait(layer.device);
            Y = dlarray(canvas,'SSCB');
        end
    end
end

function out = init_rand(sz)
    out = 2 * single(rand(sz,'gpuArray')) - 1;
end

%{
    Two differentiable supporting functions
    differentiable project_2dgs_function
    differentiable rasterize_2dgs_function
%}

function [xys, depths, radii, conics, num_tiles_hit] = ...
                       project_2dgs_func(xy_pos, rotation, scales, img_sz)

format = dims(xy_pos);

foo = project_2dgs_fused(format);

[xys, depths, radii, conics, num_tiles_hit] = ...
                                    foo(xy_pos, rotation, scales, img_sz);


xys     = dlarray(xys,format);
depths  = dlarray(depths,format);
radii   = dlarray(radii,format);
conics  = dlarray(conics,format);

if ~isgpuarray(num_tiles_hit)
    num_tiles_hit = gpuArray(num_tiles_hit);
end

end


function canvas = rasterize_2dgs_func(xys,depths,radii,conics,...
                                       num_tiles_hit,colors,opacity,img_sz)

format = dims(xys);

foo = rasterize_2dgs_fused(format);

canvas = foo(xys,depths,radii,conics,...
                                      num_tiles_hit,colors,opacity,img_sz);

canvas = dlarray(canvas);

end