# tilt-filter

* Fits the target slope to within +/- X dB (maybe 0.05 or 0.1 dB) between 20 Hz and 20 kHz.
Fixed for 48 kHz sample rate is fine.
0 dB gain at 1 kHz.
Adjustable in +/- 0.1 dB/octave steps up to +/- 6 dB/octave-- 120 table entries between -6dB/octave - 6dB/octave
Implemented as some number of second-order sections.  I would guess we need about 3 sections (6th order) to get the target accuracy.  So equivalent to 3 bands of EQ in terms of computational complexity.