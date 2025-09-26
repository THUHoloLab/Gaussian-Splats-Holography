%{
    Adan optimizers matlab implementation
    Paper: https://arxiv.org/pdf/2208.06677
%}
classdef Adan < handle
    properties
        r1;
        r2;
        r3;
        mom1;
        mom2;
        mom3;
        epsilon;
        g_last;
    end

    methods
        function self = Adan(args)
            arguments
                args.r1(1,1) {mustBeNumeric, ...
                    mustBeGreaterThanOrEqual(args.r1, 0), ...
                    mustBeLessThan(args.r1, 1)} = 0.98;

                args.r2(1,1) {mustBeNumeric, ...
                    mustBeGreaterThanOrEqual(args.r2, 0), ...
                    mustBeLessThan(args.r2, 1)} = 0.92;

                args.r3(1,1) {mustBeNumeric, ...
                    mustBeGreaterThanOrEqual(args.r3, 0), ...
                    mustBeLessThan(args.r3, 1)} = 0.99;
                
                args.eps(1,1) {mustBeNumeric, ...
                    mustBeFinite, ...
                    mustBePositive} = 1e-8;
            end

            self.r1 = args.r1;
            self.r2 = args.r2;
            self.r3 = args.r3;

            self.mom1 = [];
            self.mom2 = [];
            self.mom3 = [];

            self.epsilon = args.eps;
            self.g_last = 0;
        end

        function para = step(self,para,grad,iteration,lr)
            [para, self.mom1, self.mom2, self.mom3] = update_core(para, ...
                                                            grad, ...
                                                            self.g_last,...
                                                            self.mom1, ...
                                                            self.mom2, ...
                                                            self.mom3, ...
                                                            iteration, ...
                                                            lr, ...
                                                            self.r1, ...
                                                            self.r2, ...
                                                            self.r3, ...
                                                            self.epsilon);

            self.g_last = grad;
        end

    end
end

function [p, mom1, mom2, mom3] = update_core(p, g, g_last, mom1, mom2, mom3, ...
                                      t, lr, beta1, beta2, beta3, epsilon)
%update_core Update parameters via adaptive moment estimation


arguments
    p
    g
    g_last
    mom1
    mom2
    mom3
    t(1,1) {mustBeNumeric, mustBePositive, mustBeInteger}
    
    lr(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 0.001;
    beta1(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1, 0), mustBeLessThan(beta1, 1)} = 0.98;
    beta2(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta2, 0), mustBeLessThan(beta2, 1)} = 0.92;
    beta3(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta3, 0), mustBeLessThan(beta3, 1)} = 0.99;
    epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
end

persistent func
if isempty(func)
    func = deep.internal.LearnableUpdateFunction( ...
        @iSingleStepValue, ...
        @iSingleStepParameter);
end

if isempty(mom1) && isempty(mom2) && isempty(mom3)
    % Execute a first-step update with g_av and g_sq_av set to 0.  The step
    % will create arrays for these that are the correct size
    paramArgs = {g};
    fixedArgs = {0, 0, 0, 0, t, lr, beta1, beta2, beta3, epsilon};
else
    % Execute the normal update
    paramArgs = {g, g_last, matlab.lang.internal.move(mom1), ...
                    matlab.lang.internal.move(mom2),...
                    matlab.lang.internal.move(mom3)};

    fixedArgs = {t, lr, beta1, beta2, beta3, epsilon};
end

[p, mom1, mom2, mom3] = deep.internal.networkContainerFixedArgsFun(func, ...
    p, matlab.lang.internal.move(paramArgs), fixedArgs);
end


function [p, mom1, mom2, mom3] = iSingleStepParameter(p, g, g_last, mom1, mom2, ...
                                 mom3, t, lr, beta1, beta2, beta3, epsilon)
% Apply per-parameter learn-rate factor
lr = lr .* p.LearnRateFactor;

% Short-circuit for 0 learn rate.
if lr == 0
    return;
end

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate
biasCorrection = sqrt(1-beta3.^t)./(1-beta1.^t);
effectiveLearnRate = biasCorrection.*lr;

[step, mom1, mom2, mom3] = adan_step(...
    g, g_last, mom1, mom2, mom3, effectiveLearnRate, beta1, beta2, beta3, epsilon);

v = p.Value;
p.Value = [];
v = v + step;
p.Value = v;
end


function [p, mom1, mom2, mom3] = iSingleStepValue(p, g, g_last, mom1, mom2, mom3,...
                                       t, lr, beta1, beta2, beta3, epsilon)
% Short-circuit for 0 learn rate.
if lr == 0
    return;
end

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate

biasCorrection = sqrt(1-beta3.^t)./(1-beta1.^t);
effectiveLearnRate = biasCorrection.*lr;

[step, mom1, mom2, mom3] = adan_step(...
    g, g_last, mom1, mom2, mom3, effectiveLearnRate, beta1, beta2, beta3, epsilon);

p = p + step;
end

function [step, mom1, mom2, mom3] = adan_step(g, g_last, mom1, mom2, mom3, ...
                                            learnrate, ...
                                            beta1, beta2, beta3, epsilon)
% Adanstep   Calculate Adan update step for a single parameter

%   Copyright 2019-2022 The MathWorks, Inc.

% iAssertNumericAndReal(g,mom1,mom2);
temp = beta2 .* (g - g_last);

mom1 = (1 - beta1) .* mom1 + beta1 .* g;
mom2 = (1 - beta2) .* mom2 + temp;
mom3 = (1 - beta3) .* mom3 + beta3 .* abs(2 * g - g_last - temp).^2;

eta = learnrate ./ (sqrt(mom3) + epsilon);

step = -eta .* (mom1 + (1 - beta2) .* mom2);
end

function iAssertNumericAndReal(x,y,z)
% Assert that each of x,y,z are real numeric. 
% The condition is written such that valid cases short-circuit as soon as
% possible.
if ~(isreal(x) || ~isnumeric(x)) || ~(isreal(y)||~isnumeric(y)) || ~(isreal(z)||~isnumeric(z))
    error(message('nnet_cnn:solver:complexGradientAdan'));
end
end
