% TAC 2024 Youla-REN Project
%
% Examine linear cart-pole system with varying pole mass, and design base
% controller and weighting filters for Youla-REN learning.
%
% Author: Nic Barbara


close all;
clear;
clc;

do_plots = false;

% Model parameters
len = 10;
g = 9.81;
mc = 1;
mp_list = linspace(0.14, 0.35, 20);
mp_nom = 0.2;

% Construct the linear model as an uncertain state-space system
Afunc = @(mp) [0, 1, 0, 0;
               0, 0, -mp*g/mc, 0;
               0, 0, 0, 1;
               0, 0, (mc+mp)*g / (mc*len), 0];
B = [0, 1/mc, 0, -1/mc]';
C = [1 0 0 0];
D = 0;

nu = 1;
nx = 4;
ny = 1;

% Discrete system
dt = 0.02;
Ad_func = @(mp) eye(nx) + dt * Afunc(mp);
B = dt * B;

% Design an LQG controller
A_nom = Ad_func(mp_nom);

% Weights for nominal (robust) controller
Q = diag([1, 0, 1, 0]);
R = 1;
Sw = diag([1, 1e3, 1, 1e3]) * dt;
Sv = 1e-3 / dt;

L = dlqr(A_nom', C', Sw, Sv)';
K = dlqr(A_nom, B, Q, R);

% Weights for known controller
Sw = 1e-3 * diag([1, 1, 1, 1]) * dt;
Sv = 1e-4 / dt;

% Design output-weighting filter
w = -3;
Qfilter_inv = zpk([-50, -50, -50, -50], [w, w, w, w], 0.0005);
Ffilter_inv = zpk([-50, -50, -50, -50], [w, w, w, w], 0.01);

Qfilter = c2d(1 / Qfilter_inv, dt);
Ffilter = c2d(1 / Ffilter_inv, dt);

for mp = mp_list
    
    % Build up closed-loop systems, G: u~ -> y and Gd: u~ -> y~
    Amp = Ad_func(mp);
    Ag = [Amp, -B*K; L*C, A_nom - B*K - L*C];
    Bg = [B; B];
    Cg1 = [C zeros(ny,nx)];
    Cg2 = [C -C];
    G  = ss(Ag, Bg, Cg1, D, dt);
    Gd = ss(Ag, Bg, Cg2, D, dt);

    % Look at the known-mass controller too
    L_known = dlqr(Amp', C', Sw, Sv)';
    K_known = dlqr(Amp, B, Q, R);
    Cs = ss(Amp - B*K_known - L_known*C, L_known, K_known, 0, dt);
    Ak = [Amp, -B*K_known; L_known*C, Amp - B*K_known - L_known*C];
    Gk = ss(Ak,Bg,Cg1,D,dt);

    % Modulate the u~ -> y~ transfer function
    G_filt = series(Ffilter, G);
    Gd_filt = series(Qfilter, Gd);
    
    fprintf("mass: %.3f eig: %.4f\n", mp, max(abs(eig(Ag))));

    % Put on a bodeplot
    if do_plots
        figure(1); 
        hold on;
        bodeplot(G, c2d(Ffilter_inv, dt));
        title('u~ -> y')
    
        figure(2);
        hold on;
        bodeplot(Gd, c2d(Qfilter_inv, dt));
        title('u~ -> y~')
        
        figure(3);
        hold on;
        step(G, 50);
        title('Step response (base)')
    
        figure(4);
        hold on;
        step(Gk,50);
        title('Step response (optimal - known mass)')
    
        figure(5);
        hold on;
        bodeplot(Cs);
        title('Optimal (known mass) controller')
    
        figure(6);
        hold on;
        bodeplot(G_filt);
        title('u~ -> y (filtered)')
    
        figure(7);
        hold on;
        bodeplot(Gd_filt);
        title('u~ -> y~ (filtered)')
    
        figure(8);
        hold on;
        step(G_filt, 50);
        title('u~ -> y (filtered)')
    
        figure(9);
        hold on;
        step(Gd_filt, 50);
        title('u~ -> y~ (filtered)')
    end

    if mp == mp_list(1) || mp == mp_list(end)
        disp(hinfnorm(G));
        disp(hinfnorm(G_filt));
        disp(hinfnorm(Gd));
        disp(hinfnorm(Gd_filt));
    end
end
