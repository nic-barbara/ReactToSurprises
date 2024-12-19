% Formatting plots nicely. 
% Figure set to white, grid on, box on.
%
% INPUTS:
%   - fsize:    (Numeric) Font size. Default 16.
%   - lwidth:   (Numeric) Line Width. Default 1.5.
%   - axHndl:   (Axes) Axis handle. Default from axHndl = gca.
%   - figHndl:  (Figure) Figure handle. Default from figHndl = gcf.
%
% Author:   Nicholas Barbara
% Email:    nbar5346@uni.sydney.edu.au

function formatNice(fsize,lwidth,axHndl,figHndl)
    
    % Inputs
    if nargin < 1
        fsize = 14;
    end
    if nargin < 2
        lwidth = 1.5;
    end
    if nargin < 3
        axHndl = gca;
    end
    if nargin < 4
        figHndl = gcf;
    end
        
    % Format figure
    set(figHndl,'color','w');
    grid(axHndl,'minor');
    axHndl.Box = 'on';

    % Linewidths and font size
    set(findall(figHndl, '-property', 'FontSize'), 'FontSize', fsize);
    set(findall(figHndl,'Type','Line'), 'LineWidth',lwidth);
    
    % Ditch box around legend
    leg = findobj(figHndl,'Type','Legend');
    if ~isempty(leg)
        for k = 1:length(leg)
            set(leg(k),'EdgeColor',leg(k).Color);
            set(leg(k),'FontSize',0.85*fsize);
            set(leg(k).BoxFace,'ColorType','truecoloralpha',...
                'ColorData',uint8(255*[1;1;1;0.8]));
        end
    end
end