% Nearmap Thesis Project
% 
% Script to set nice formatting for bode plots. The opts struct allows you
% to configure the bodeplot - fontsize, linewidth, whether to create a new
% figure, how big the figure should be, phase wrapping and matching.
%
% All inputs after the opts struct must follow the same syntax as bodeplot!
%
% Author:   Nicholas Barbara
% Email:    nbar5346@uni.sydney.edu.au
%
% b = my_bodeplot(opts,varargin)


function b = my_bodeplot(opts,varargin)

    % Different defaults for mac and windows
    if ismac
        d_fsize  = 18;
        d_lwidth = 1.5;
    else
        d_fsize  = 14;
        d_lwidth = 1.2;
    end
    
    % Default plotting options
    if isempty(opts)
        opts = struct;
    end
    if ~isfield(opts,'fsize')
        opts.fsize = d_fsize;
    end
    if ~isfield(opts,'lwidth')
        opts.lwidth = d_lwidth;
    end
    if ~isfield(opts,'PhaseVisible')
        opts.PhaseVisible = 'on';
    end
    if ~isfield(opts,'phasewrap')
        opts.phasewrap = 'off';
    end
    if ~isfield(opts,'phasematch')
        opts.phasematch = 'off';
    end
    
    % Make the plot using custom options
    b = bodeplot(varargin{:});
    h = my_bode_opts(opts.fsize,opts.phasewrap,opts.phasematch);
    h.PhaseVisible = opts.PhaseVisible;
    setoptions(b,h);
    
    % Change the damn lineiwdth and figure background colour
    figHndl = gcf;
    set(figHndl,'color','w');
    set(findall(figHndl,'Type','Line'), 'LineWidth',opts.lwidth);
end