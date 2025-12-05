function [ previous_text_len ] = mat_progress_bar(varargin)
%MAT_PROGRESS_BAR_T
%   Input:
%       - front_text:  a char array, the text before the progress bar
%       - current_index: index of current loop
%       - max_index: total number of the loop
%       - t: time estimation of current loop, acquired by tic/toc
%       - previous_text_len: previous text width
%       - fid (optional): file ID for writing
%       - custom_name, custom_value (optional): additional display
%
%   Output:
%       - previous_text_len
%
%   Author: Modified by ChatGPT

    persistent total_t

    marker_solid = '#';
    marker_empty = '-';

    %% default
    write_text_file = 0;
    custom_str = '';

    %% param parsing
    if nargin >= 5
        front_text = varargin{1};
        current_index = varargin{2};
        max_index = varargin{3};
        t  = varargin{4};
        previous_text_len  = varargin{5};
    else
        error('Must input at least 5 arguments.');
    end

    % Remaining arguments
    if nargin == 6
        arg6 = varargin{6};
        if isnumeric(arg6)
            % 6th argument is fid
            fid = arg6;
            write_text_file = 1;
        elseif ischar(arg6) || isstring(arg6)
            % Only name given, need its value → error
            error('Must provide value along with custom name');
        end

    elseif nargin == 7
        % Must be (name, value)
        custom_name = varargin{6};
        custom_value = varargin{7};
        custom_str = [' [', custom_name, ': ', num2str(custom_value, '%.3e'), ']'];

    elseif nargin == 8
        % form:  front,cur,max,t,prev, fid, name,value
        fid = varargin{6};
        write_text_file = 1;

        custom_name = varargin{7};
        custom_value = varargin{8};
        custom_str = [' [', custom_name, ': ', num2str(custom_value, '%.3e'), ']'];
    end

    %% accumulate total time
    if isempty(total_t)
        total_t = 0;
    else
        total_t = total_t + t;
    end

    %% remaining time estimation
    remain_t = t*(max_index - current_index);
    remain_t_h = floor(remain_t/3600);
    remain_t_m = floor((remain_t - remain_t_h*3600)/60);
    remain_t_s = round(remain_t - remain_t_h*3600 - remain_t_m*60);
    remain_t_str = [' [RT: ',num2str(remain_t_h), 'h ', num2str(remain_t_m), 'min ', num2str(remain_t_s,'%02d'), 's]'];

    %% total time display
    total_t_h = floor(total_t/3600);
    total_t_m = floor((total_t - total_t_h*3600)/60);
    total_t_s = round(total_t - total_t_h*3600 - total_t_m*60);
    total_t_str = [' [UT: ', num2str(total_t_h), 'h ', num2str(total_t_m), 'min ', num2str(total_t_s, '%02d'), 's]'];

    %% loop info
    index_str = [' [Loop: ', num2str(current_index), '/', num2str(max_index), ']'];

    %% percentage
    percent_done = 100 * current_index / max_index;
    perc_str = num2str(percent_done,'%3.1f');

    %% progress bar
    total_bar_num = 20;
    prc_step = 100/total_bar_num;

    dot_num = round(percent_done./prc_step);
    bar_str = ['[', repmat(marker_solid,1,dot_num), repmat(marker_empty,1,total_bar_num-dot_num), '] '];

    %% final text
    total_str = [front_text, ': ', bar_str, perc_str, '%%', index_str, custom_str, remain_t_str, total_t_str];

    %% update displayed string
    if current_index == 1
        s = total_str;

    elseif current_index > 1 && current_index < max_index
        backspace_matrix = repmat(sprintf('\b'), 1, previous_text_len);
        s = [backspace_matrix, total_str];

    elseif current_index == max_index
        backspace_matrix = repmat(sprintf('\b'), 1, previous_text_len);
        s = [backspace_matrix, total_str, '\n'];
    end

    previous_text_len = length(sprintf(total_str));

    %% display output
    fprintf(s);

    if write_text_file
        fprintf(fid, s);
    end

end
