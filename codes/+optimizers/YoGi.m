classdef YoGi < handle
    properties
        mom1
        mom2
        beta1
        beta2
        epsilon
        iteration
    end
    
    methods
        function self = YoGi(beta1,beta2,epsilon)

            arguments
                beta1(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1, 0), mustBeLessThan(beta1, 1)} = 0.9;
                beta2(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta2, 0), mustBeLessThan(beta2, 1)} = 0.999;
                epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
            end

            self.mom1       = [];
            self.mom2       = [];
            self.beta1      = beta1;
            self.beta2      = beta2;
            self.epsilon    = epsilon;
        end

        function para = step(self,para,grad,iteration,lr)
            [para, self.mom1, self.mom2] = update_core(para, grad, ...
                                                            self.mom1, ...
                                                            self.mom2, ...
                                                            iteration, ...
                                                            lr, ...
                                                            self.beta1, ...
                                                            self.beta2, ...
                                                            self.epsilon);

        end
    end
end




function [p, avg_g, avg_gsq] = update_core(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon)
%update_core Update parameters via adaptive moment estimation


arguments
    p
    g
    avg_g
    avg_gsq
    t(1,1) {mustBeNumeric, mustBePositive, mustBeInteger}
    
    lr(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 0.001;
    beta1(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1, 0), mustBeLessThan(beta1, 1)} = 0.9;
    beta2(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta2, 0), mustBeLessThan(beta2, 1)}= 0.999;
    epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
end

persistent func
if isempty(func)
    func = deep.internal.LearnableUpdateFunction( ...
        @iSingleStepValue, ...
        @iSingleStepParameter );
end

if isempty(avg_g) && isempty(avg_gsq)
    % Execute a first-step update with g_av and g_sq_av set to 0.  The step
    % will create arrays for these that are the correct size
    paramArgs = {g};
    fixedArgs = {0, 0, t, lr, beta1, beta2, epsilon};
else
    % Execute the normal update
    paramArgs = {g, matlab.lang.internal.move(avg_g), matlab.lang.internal.move(avg_gsq)};
    fixedArgs = {t, lr, beta1, beta2, epsilon};
end

[p, avg_g, avg_gsq] = deep.internal.networkContainerFixedArgsFun(func, ...
    p, matlab.lang.internal.move(paramArgs), fixedArgs);
end


function [p, avg_g, avg_gsq] = iSingleStepParameter(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon)
% Apply per-parameter learn-rate factor
lr = lr .* p.LearnRateFactor;

% Short-circuit for 0 learn rate.
if lr == 0
    return;
end

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate
biasCorrection = sqrt(1-beta2.^t)./(1-beta1.^t);
effectiveLearnRate = biasCorrection.*lr;

[step, avg_g, avg_gsq] = yogi_step(...
    g, avg_g, avg_gsq, effectiveLearnRate, beta1, beta2, epsilon);

v = p.Value;
p.Value = [];
v = v + step;
p.Value = v;
end


function [p, avg_g, avg_gsq] = iSingleStepValue(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon)
% Short-circuit for 0 learn rate.
if lr == 0
    return;
end

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate

biasCorrection = sqrt(1-beta2.^t)./(1-beta1.^t);
effectiveLearnRate = biasCorrection.*lr;

[step, avg_g, avg_gsq] = yogi_step(...
    g, avg_g, avg_gsq, effectiveLearnRate, beta1, beta2, epsilon);
p = p + step;
end

function [step, avg_g, avg_gsq] = yogi_step(g, avg_g, avg_gsq, learnrate, beta1, beta2, epsilon)
% YoGistep   Calculate YoGi update step for a single parameter

%   Copyright 2019-2022 The MathWorks, Inc.

% iAssertNumericAndReal(g,avg_g,avg_gsq);

if beta1~=1
    % If beta1 is 1 then the new gradients do not contribute to the trailing
    % averages.

    % Add g to an appropriately scaled avg_g so that we don't need an
    % additional temporary array that is a scaled copy of g.
    avg_g = beta1 .* avg_g + (1 - beta1) .* g;
end

avg_gsq = avg_gsq - (1 - beta2) .* sign(avg_gsq - abs(g).^2) .* (abs(g).^2);

step = -learnrate.*( avg_g ./ (sqrt(avg_gsq) + epsilon) );
end

function iAssertNumericAndReal(x,y,z)
% Assert that each of x,y,z are real numeric. 
% The condition is written such that valid cases short-circuit as soon as
% possible.
if ~(isreal(x) || ~isnumeric(x)) || ~(isreal(y)||~isnumeric(y)) || ~(isreal(z)||~isnumeric(z))
    error(message('nnet_cnn:solver:complexGradientYoGi'));
end
end
