function [simpot_matrix, fs_out, pre_samp, ansum_matrix] = roenne2012_varied_batch(levels_db, stimulus_family, flow, fhigh, num_cf, cohc, cihc, spont, ur_delay_ms)
%ROENNE2012_VARIED_BATCH Local Roenne helper with configurable stimulus, CF coverage, and UR delay.
%
%   [simpot_matrix, fs_out, pre_samp, ansum_matrix] = roenne2012_varied_batch( ...
%       levels_db, stimulus_family, flow, fhigh, num_cf, cohc, cihc, spont,
%       ur_delay_ms)
%
%   This helper reproduces the Roenne 2012 pipeline:
%     stimulus -> zilany2007 AN model -> across-CF sum -> data_roenne2012 UR
%   It needs the zilany2007.m of amtoolbox-1.6.0/, which takes cohc, cihc and
%   spont (README.md).
%
%   Unlike the stock roenne2012.m entry point, this wrapper exposes:
%     - stimulus selection across Elberling click / chirp waveforms
%     - cochlear coverage via flow / fhigh / numCF
%     - one delayed unitary-response kernel for readout latency variation
%
%   The Roenne UR already represents the compound ABR generator shape. This
%   helper keeps that single kernel intact and varies only its overall delay.
%
%   Inputs
%   ------
%   levels_db       : Row or column vector of stimulus levels in dB peSPL.
%   stimulus_family : 'click', 'chirp1', ..., 'chirp5'
%   flow            : Lowest CF passed to zilany2007, in Hz.
%   fhigh           : Highest CF passed to zilany2007, in Hz.
%   num_cf          : Number of CF channels passed to zilany2007.
%   cohc            : OHC scaling factor.
%   cihc            : IHC scaling factor.
%   spont           : Spontaneous firing rate.
%   ur_delay_ms     : Added delay of the single Roenne unitary response, in ms.
%
%   Outputs
%   -------
%   simpot_matrix   : [n_samples x n_levels] simulated ABR potentials.
%   fs_out          : Native sample rate of simpot_matrix (30 kHz).
%   pre_samp        : Sample index at fs_out corresponding to the 15 ms
%                     Elberling stimulus reference used for cropping.
%   ansum_matrix    : [n_samples x n_levels] across-channel auditory-nerve population response.

levels_db = levels_db(:).';
num_levels = length(levels_db);
fsmod = 200e3;
min_modellength_ms = 40;

[ur, fs_out] = data_roenne2012;
[stim, fsstim, pre_samp] = local_prepare_stimulus(stimulus_family, fs_out);
readout_ur = local_delay_kernel(ur(:), fs_out, ur_delay_ms);

if length(stim) / fsstim < min_modellength_ms / 1000
    stim_temp = zeros(1, round(fsstim * min_modellength_ms / 1000));
    stim_temp(1:length(stim)) = stim;
    stim = stim_temp;
end

simpot_matrix = [];
ansum_matrix = [];
for idx = 1:num_levels
    level_db = levels_db(idx);
    [ANout, ~] = zilany2007(level_db, stim, fsstim, fsmod, ...
        'flow', flow, ...
        'fhigh', fhigh, ...
        'numCF', num_cf, ...
        'cohc', cohc, ...
        'cihc', cihc, ...
        'spont', spont);

    ANout = ANout - spont;
    ANsum = sum(ANout, 2);
    ANsum = resample(ANsum, fs_out, fsmod);
    simpot = filter(readout_ur, 1, ANsum);

    if isempty(simpot_matrix)
        simpot_matrix = zeros(length(simpot), num_levels);
        ansum_matrix = zeros(length(ANsum), num_levels);
    end
    simpot_matrix(:, idx) = simpot;
    ansum_matrix(:, idx) = ANsum;
end

end


function [stim, fsstim, pre_samp] = local_prepare_stimulus(stimulus_family, fs_out)
%LOCAL_PREPARE_STIMULUS Resolve an Elberling click/chirp waveform by name.

[stim_data, fsstim] = data_elberling2010('stim');

family = lower(strtrim(char(stimulus_family)));
switch family
    case 'click'
        stim = stim_data.c0;
    case 'chirp1'
        stim = stim_data.c1;
    case 'chirp2'
        stim = stim_data.c2;
    case 'chirp3'
        stim = stim_data.c3;
    case 'chirp4'
        stim = stim_data.c4;
    case 'chirp5'
        stim = stim_data.c5;
    otherwise
        error('Unsupported stimulus_family: %s', family);
end

stim = stim(:).';

% Roenne2012_chirp subtracts a fixed 15 ms reference for both click and chirps.
% We keep the same alignment here so Python can crop from the stimulus reference.
pre_samp = round(15e-3 * fs_out);
end


function y = local_delay_kernel(x, fs, delay_ms)
%LOCAL_DELAY_KERNEL Shift a kernel to later latency without changing length.

shift = round(delay_ms * fs / 1000);
if shift <= 0
    y = x;
    return;
end

y = zeros(size(x));
if shift < length(x)
    y(shift + 1:end) = x(1:end - shift);
end
end
