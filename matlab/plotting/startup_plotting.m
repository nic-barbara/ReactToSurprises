% Default plot formatting - text interpreter
%
% Function changes the text interpreter to the specified string, with the
% default as 'latex'
%
% Author:   Nicholas Barbara
% Email:    nbar5346@uni.sydney.edu.au

function startup_plotting(str)
    
    % Inputs
    if nargin < 1
        str = 'latex';
    end
    
    % Change all the defaults
    set(groot,'defaulttextinterpreter',str);
    set(groot,'defaultAxesTickLabelInterpreter',str);
    set(groot,'defaultLegendInterpreter',str);
    
    % Set colours
    colors = GiveMeColors(8);
%     c2     = GiveMeColors(10);
%     colors = [colors;c2(5:end)];
    set(groot,'defaultAxesColorOrder',[colors{1};colors{2};...
        colors{3};colors{4};colors{5};colors{6};colors{7};colors{8}]);
end