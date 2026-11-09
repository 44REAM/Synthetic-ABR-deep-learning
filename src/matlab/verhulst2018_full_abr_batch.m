function [w1_matrix, w3_matrix, w5_matrix, fs_abr, pre_samp, an_matrix] = verhulst2018_full_abr_batch(levels_db, stimulus_family, hearing_profile, subject_seed, numH, numM, numL, varargin)
%VERHULST2018_FULL_ABR_BATCH  Build configurable stimuli and return W1, W3, and W5.
%
%   [w1_matrix, w3_matrix, w5_matrix, fs_abr, pre_samp, an_matrix] = verhulst2018_full_abr_batch(
%       levels_db, stimulus_family, hearing_profile, subject_seed, numH, numM, numL, varargin)
%
%   Input parameters:
%     levels_db      : Row vector of stimulus levels in dB peSPL.
%     stimulus_family: 'verhulst_click' to preserve the original
%                      demo_verhulst2018-style click, or one of the
%                      Elberling toolbox families 'click', 'chirp1', ...,
%                      'chirp5'.
%     hearing_profile: Hearing-profile string, e.g. 'Flat00' or 'Flat00_Slope35'.
%     subject_seed   : Integer seed for cochlear-irregularity randomness.
%     numH           : Number of high-SR AN fibres (1-13).
%     numM           : Number of medium-SR AN fibres (0-3).
%     numL           : Number of low-SR AN fibres (0-3).
%     varargin       : Optional parameter-value pairs passed directly to verhulst2018
%                      including:
%                        - Cochlear: 'IrrPct', 'irr_on', 'non_linear_type'
%                        - IHC: 'ihc_scal_constant'
%                        - AN: 'kSR_H', 'kSR_M', 'kSR_L', 'kmax_H', 'kmax_M', 'kmax_L'
%                        - CN: 'Tex_cn', 'Tin_cn', 'dly_cn', 'Acn', 'Scn'
%                        - IC: 'Tex_ic', 'Tin_ic', 'dly_ic', 'Aic', 'Sic'
%                        - Scaling: 'M1', 'M3', 'M5'
%
%   Output parameters:
%     w1_matrix : Wave-I   (cochlear nerve) population response, (n_samples x n_levels), Volts.
%     w3_matrix : Wave-III (cochlear nucleus) population response, (n_samples x n_levels), Volts.
%     w5_matrix : Wave-V   (inferior colliculus) population response, (n_samples x n_levels), Volts.
%     fs_abr    : Scalar sample rate shared by all three matrices (default 20000 Hz).
%     pre_samp  : Stimulus-reference sample index at fs_abr used by the
%                 Python caller for time-axis alignment and delay insertion.
%     an_matrix : Across-channel summed combined auditory-nerve population response, (n_samples x n_levels).
%
%   The default 'verhulst_click' stimulus is identical to that of AMT's
%   demo_verhulst2018.m:
%     - 100 kHz sampling rate, 50 ms duration
%     - 100 us positive polarity click starting at sample 10 (1-based)
%     - Amplitude converted from dB peSPL using p0 = 2e-5 Pa
%
%   The Elberling-family options reuse the toolbox click/chirp waveforms
%   also used by src/synthetic_roenne.py, calibrated exactly as the Roenne
%   path calibrates them (roenne2012_varied_batch.m -> zilany2007): the
%   waveform is zero-padded to 40 ms, resampled to 200 kHz, and its
%   non-zero samples are scaled to the requested level (dB re 20 uPa). The
%   calibrated pressure waveform is then resampled to 100 kHz, so both
%   models receive the same physical stimulus at the same nominal level.
%
%   The Python caller (src/synthetic_verhulst.py) delays each wave by its
%   brainstem delay before summing them into the composite ABR signal.
%
%   This helper is used by src/synthetic_verhulst.py.

if nargin == 6
    numL = numM;
    numM = numH;
    numH = subject_seed;
    subject_seed = hearing_profile;
    hearing_profile = stimulus_family;
    stimulus_family = 'verhulst_click';
    extra_opts = {};
elseif nargin >= 7
    extra_opts = varargin;
else
    error('verhulst2018_full_abr_batch expects at least 6 input arguments.');
end

levels_db = levels_db(:).';
N_levels = length(levels_db);
[insig, fs, pre_time_s] = local_prepare_stimulus_matrix(levels_db, stimulus_family);

output = verhulst2018(insig, fs, 'abr', ...
    'numH', numH, 'numM', numM, 'numL', numL, ...
    'hearing_profile', hearing_profile, ...
    'subject', subject_seed, ...
    extra_opts{:}, ...
    'silent');

n_out     = length(output(1).w5);
w1_matrix = zeros(n_out, N_levels);
w3_matrix = zeros(n_out, N_levels);
w5_matrix = zeros(n_out, N_levels);
an_matrix = zeros(n_out, N_levels);
for j = 1:N_levels
    w1_matrix(:, j) = output(j).w1;
    w3_matrix(:, j) = output(j).w3;
    w5_matrix(:, j) = output(j).w5;
    an_matrix(:, j) = sum(output(j).an_summed, 2);
end

fs_abr = output(1).fs_abr;
pre_samp = round(pre_time_s * fs_abr);


function [insig, fs, pre_time_s] = local_prepare_stimulus_matrix(levels_db, stimulus_family)
%LOCAL_PREPARE_STIMULUS_MATRIX Build the requested Verhulst input family.

fs   = 100e3;
p0   = 2e-5;
dur  = 50e-3;
N_levels = length(levels_db);
family = lower(strtrim(char(stimulus_family)));

switch family
    case {'verhulst_click', 'legacy_click'}
        dur_click_samples = round(100e-6 * fs);
        N_samples = round(dur * fs);
        insig = zeros(N_samples, N_levels);
        for j = 1:N_levels
            amp = p0 * 10^(levels_db(j) / 20) * sqrt(2) * 2;
            insig(10:10+dur_click_samples, j) = amp;
        end
        pre_time_s = (10 - 1) / fs;

    case {'click', 'chirp1', 'chirp2', 'chirp3', 'chirp4', 'chirp5'}
        [stim_data, fsstim] = data_elberling2010('stim');

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
        end

        % Same calibration as the Roenne path: roenne2012_varied_batch.m pads
        % the stimulus to 40 ms, and zilany2007 resamples it to 200 kHz and
        % scales its non-zero samples to the level with scaletodbspl.
        stim = stim(:).';
        min_len = round(40e-3 * fsstim);
        if length(stim) < min_len
            stim(min_len) = 0;
        end
        fs_cal = 200e3;
        lvlref = 20 * log10(1 / p0);
        base = resample(stim, fs_cal, fsstim);
        idnz = base ~= 0;

        N_samples = max(round(dur * fs), ceil(length(base) * fs / fs_cal));
        insig = zeros(N_samples, N_levels);
        for j = 1:N_levels
            pressure = base;
            pressure(idnz) = scaletodbspl(base(idnz), levels_db(j), 'dboffset', lvlref);
            pressure = resample(pressure, fs, fs_cal);
            insig(1:length(pressure), j) = pressure(:);
        end

        % Match the 15 ms toolbox reference used by the Roenne helper path.
        pre_time_s = 15e-3;

    otherwise
        error('Unsupported stimulus_family: %s', family);
end
