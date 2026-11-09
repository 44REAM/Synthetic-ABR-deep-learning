function [ANdata,vFreq] = zilany2007(stim_level,stim,fsstim,fsmod,varargin)
%zilany2007  Nonlinear model of auditory periphery (basic version)
%   Usage: [ANdata,vFreq] = zilany2007(stim_level,stim,fsstim,fsmod,varargin)
%
%   Input parameters:
%     stim_level  : SPL of the stimulus (in dB).
%     stim        : Pressure waveform of stimulus (a time series).
%     fsstim      : Sampling frequency of the stimulus (in Hz).
%     fsmod       : Sampling frequency of the model (in Hz, often 200 kHz).
%
%   Output parameters:
%     ANdata     : Matrix with the excitation in 500 different AN fibers at 
%                  center frequencies spaced equally on the basilar membrane.
%     vFreq      : Vector with the 500 center frequencies.
%
%   ZILANY2007(stim_level,stim,fsstim,fsmod) returns simulations
%   from Rønne et al. (2012). It calls the mex'ed C code containing the
%   humanized version of the AN model from Zilany et al. (2007). The humanization
%   is described in Rønne et al. (2012). The AN model is called 500 times to
%   simulate 500 fibers tuned to different center frequencies.
%
%   This function takes the following optional parameters:
%
%     'flow',flow     Lowest centre frequency (in Hz). Default is 100 Hz.
%
%     'fhigh',fhigh   Highest centre frequency (in Hz). Default is 16000 Hz.
%
%     'numCF',nf    Number of fibers between lowest and highest
%                   frequency. The fibers will be equidistantly spaced
%                   on the basilar membrane. Default is 500.
%  
%   
%   References:
%     F. M. Rønne, T. Dau, J. Harte, and C. Elberling. Modeling auditory
%     evoked brainstem responses to transient stimuli. The Journal of the
%     Acoustical Society of America, 131(5):3903--3913, 2012.
%     
%     M. S. A. Zilany and I. C. Bruce. Representation of the vowel (epsilon)
%     in normal and impaired auditory nerve fibers: Model predictions of
%     responses in cats. The Journal of the Acoustical Society of America,
%     122(1):402--417, jul 2007.
%     
%
%   Url: http://amtoolbox.org/amt-1.6.0/doc/models/zilany2007.php


%   #StatusDoc: Perfect
%   #StatusCode: Good
%   #Verification: Unknown
%   #Requirements: MEX M-Signal 
%   #Author: Peter L. Sondergaard (2012)

% This file is licensed unter the GNU General Public License (GPL) either 
% version 3 of the license, or any later version as published by the Free Software 
% Foundation. Details of the GPLv3 can be found in the AMT directory "licences" and 
% at <https://www.gnu.org/licenses/gpl-3.0.html>. 
% You can redistribute this file and/or modify it under the terms of the GPLv3. 
% This file is distributed without any warranty; without even the implied warranty 
% of merchantability or fitness for a particular purpose. 
%
% Changed from AMT 1.6.0 for the code of 'Physiologically Informed Synthetic Auditory Brainstem
% Responses Enable Synthetic-to-Real Wave V Classification' (2026): cohc, cihc and spont are
% key-value parameters (defaults 1, 1 and 50, the values the release fixes) instead of fixed values.

if nargin<4
  error('%s: Too few input parameters.',upper(mfilename));
end;

% Define input flags
definput.flags.disp = {'no_debug','debug'};
definput.flags.progress = {'progress','silent'};
definput.keyvals.flow    = 100;
definput.keyvals.fhigh   = 16000;
definput.keyvals.numCF = 500;
definput.keyvals.cohc   = 1;    % ADDED: OHC scaling factor
definput.keyvals.cihc   = 1;    % ADDED: IHC scaling factor
definput.keyvals.spont  = 50;   % ADDED: spontaneous firing rate
[flags,kv]  = ltfatarghelper({'flow','fhigh','numCF'},definput,varargin);

stim	= resample(stim,fsmod,fsstim);	% stim fs = mod fs
idnz	= stim ~= 0;                    % ignore pauses
lvlref = 20*log10(1/20e-6);           % Reference level: 20 micro Pa
stim(idnz) = scaletodbspl(stim(idnz),stim_level,'dboffset',lvlref); % Calibrate level

% stim must be a row vector
if size(stim,2) == 1
    stim = stim';
end

% location of lowest and highest centre frequency
xlo     = (1.0/0.06)*log10((kv.flow/165.4)+0.88);
xhi     = (1.0/0.06)*log10((kv.fhigh/165.4)+0.88);	

% equal spaced distances on the BM
vX      = linspace(xlo,xhi,kv.numCF);

% and the resulting frequency vector
vFreq   = 165.4*(10.^(0.06*vX)-0.88); 

% resolution in the time domain
tdres   = 1/fsmod;                

% spontaneous rate in sp/sec
spont   = kv.spont;    

% time in sec between stimulus repetitions - NOTE should be equal to or
% longer than the duration of the stimulus
reptime   = (2*length(stim)+1)/fsmod; 

% cohc is the ohc scaling factor: 1 is normal OHC function; 0 is complete
% OHC dysfunction
cohc    = kv.cohc;                                        

% cihc is the ihc scaling factor: 1 is normal IHC function; 0 is complete
% IHC dysfunction
cihc    = kv.cihc;                                        

% Call AN model - loop over the 500 fibers tuned to different CFs
for jj = 1:kv.numCF
  % Call AN model (mex'ed C model)
  [timeout,meout,c1filterout,c2filterout,c1vihc,c2vihc,vihc,synout,psth500k] ...
      = comp_zilany2007(stim,...
                                 vFreq(jj),...
                                 1,...
                                 tdres,...
                                 reptime,...
                                 cohc, ...
                                 cihc,...
                                 spont); 
  
  % Use the output of the synapse stage of the AN model. 
  ANdata(jj,:)           = synout;             
end

timeout = timeout.';
meout = meout.';
c1filterout = c1filterout.';
c2filterout = c2filterout.';
c1vihc = c1vihc.';
c2vihc = c2vihc.';
vihc = vihc.';
psth500k = psth500k.';
ANdata = ANdata.';


