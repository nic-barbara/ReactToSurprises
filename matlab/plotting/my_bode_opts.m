% Nearmap Thesis Project
% 
% Nice options for Bode plots. Still have to change the linewidth
% separately though!
%
% Author:   Nicholas Barbara
% Email:    nbar5346@uni.sydney.edu.au

function h = my_bode_opts(fsize,phasewrap,phasematch)
    
    % Default options
    h = bodeoptions;
    if nargin < 2, phasewrap = 'off'; end
    if nargin < 3, phasematch = 'off'; end
    
    % Units
    h.FreqUnits = 'Hz';
    
    % Grid
    h.Grid = 'on';
    
    % Axis labels
    h.XLabel.Interpreter = 'latex';
    h.YLabel.Interpreter = 'latex';
    h.Title.Interpreter  = 'latex';
    h.XLabel.FontSize    = fsize;
    h.YLabel.FontSize    = fsize;
    h.Title.FontSize     = fsize;
    
    % Tick Labels
    h.TickLabel.FontSize = fsize - 2;
    h.TickLabel.Color    = 0.4*[1 1 1];
    
    % Input/output labels
    h.InputLabels.Interpreter = 'latex';
    h.InputLabels.FontSize = fsize - 3;
    h.OutputLabels.Interpreter = 'latex';
    h.OutputLabels.FontSize = fsize - 3;
    
    % Remove title - we know it's a Bode Diagram!
    h.Title.String = '';
    
    % Handle phase wrapping/matching
    h.PhaseWrapping = phasewrap;
    h.PhaseMatching = phasematch;
end