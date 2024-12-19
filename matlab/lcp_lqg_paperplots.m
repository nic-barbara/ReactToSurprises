% TAC 2024 Youla-REN Project
%
% Same as lcp_lqg_design.m, but just for making nice
% plots of the base-controlled transfer functions.
%
% Author: Nic Barbara

close all;
clear;
clc;

addpath("plotting/")
startup_plotting;

% Model parameters
len = 10;
g = 9.81;
mc = 1;
mp_list = linspace(0.14, 0.36, 20);
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
Qfilter_inv = zpk([-50, -50, -50, -50], [w, w, w, w], 0.0005) / 100;
Ffilter_inv = zpk([-50, -50, -50, -50], [w, w, w, w], 0.01) / 10;

Qfilter = c2d(1 / Qfilter_inv, dt);
Ffilter = c2d(1 / Ffilter_inv, dt);

% Plotting options
freqs = {2*pi*5e-3, 2*pi*1e2};
opts = struct;
opts.fsize = 24;
opts.lwidth = 2;
opts.PhaseVisible = 'off';
opts.phasematch = 'on';

figure('Units', 'normalized', 'Position', [0.25, 0.25, 0.34, 0.37]);
hold on;
figure('Units', 'normalized', 'Position', [0.25, 0.25, 0.34, 0.37]);
hold on;

for mp = mp_list
    
    % Build up closed-loop systems, G: u~ -> y and Gd: u~ -> y~
    Amp = Ad_func(mp);
    Ag = [Amp, -B*K; L*C, A_nom - B*K - L*C];
    Bg = [B; B];
    Cg1 = [C zeros(ny,nx)];
    Cg2 = [C -C];
    G  = ss(Ag, Bg, Cg1, D, dt);
    Gd = ss(Ag, Bg, Cg2, D, dt);
    
    fprintf("mass: %.3f eig: %.4f\n", mp, max(abs(eig(Ag))));

    % Put on a bodeplot
    figure(1);
    hG1 = my_bodeplot(opts, G, freqs);

    figure(2);
    hG2 = my_bodeplot(opts, Gd, freqs);
end

% Change the colours and plot the weighting filters
greys = flip(linspace(0.75, 0.1, length(mp_list)));
greys = [greys, greys];

change_bodeplot_colours(greys, 1);
hF1 = my_bodeplot(opts, c2d(Ffilter_inv, dt), freqs);
% title('$\tilde{u} \mapsto y$')

change_bodeplot_colours(greys, 2);
hF2 = my_bodeplot(opts, c2d(Qfilter_inv, dt), freqs);
% title('$\tilde{u} \mapsto \tilde{y}$')

function change_bodeplot_colours(greys, fid)
    figure(fid);
    lines = findall(gcf, 'Type', 'line');
    i = 0;
    for k = 1:length(lines)
        lk = lines(k);
        if ~(isinf(lk.YData(1)) || isnan(lk.YData(1)))
            if ~(lk.XData(1) == lk.XData(2))
                i = i + 1;
                set(lk, 'Color', greys(i)*[1, 1, 1])
            end
        end
    end
end
