import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as optimize
from scipy.signal import sosfreqz, zpk2sos

class TiltFilter:
    """Tilt filter using alternating real poles and zeros"""
    
    def __init__(self, fs=48000):
        self.fs = fs
        self.pivot_freq = 1000
        
    def tilt_target(self, f, slope_db_per_octave):
        return slope_db_per_octave * np.log2(f / self.pivot_freq)
    
    def design_tilt_filter(self, slope_db_per_octave, N=6):
        """
        KEY INSIGHT FROM PAPER: alternating real poles and zeros distributed logarithmically
        """
        print(f"\nDesigning {slope_db_per_octave} dB/octave tilt filter with {N} poles/zeros...")
        
        # 0 dB/octave case
        if abs(slope_db_per_octave) < 1e-10:
            print("Creating unity filter")
            return np.array([]), np.array([]), 1.0, np.array([[1, 0, 0, 1, 0, 0]])
        
        f_min = 20
        f_max = 20000
        
        abs_slope = abs(slope_db_per_octave)

        # adaptive order selection based on slope
        if abs_slope <= 0.5:
            # very small slopes - near cancelling pairs
            poles_freq = []
            zeros_freq = []
            
            n_pairs = N // 2
            frequencies = np.logspace(np.log10(100), np.log10(10000), n_pairs)
            
            for f in frequencies:
                # Separation factor used to determine slope -- smaller separation for small slopes
                sep_factor = 1 + 0.1 * abs_slope  # e.g., 1.01 for 0.1 dB/oct
                
                if slope_db_per_octave < 0:
                    zeros_freq.append(f / np.sqrt(sep_factor))
                    poles_freq.append(f * np.sqrt(sep_factor))
                else:
                    poles_freq.append(f / np.sqrt(sep_factor))
                    zeros_freq.append(f * np.sqrt(sep_factor))
                    
        elif abs_slope >= 5.0:
            n_low = N // 2
            n_high = N - n_low
            if slope_db_per_octave < 0:
                zeros_freq = np.logspace(np.log10(10), np.log10(100), n_low)
                poles_freq = np.logspace(np.log10(5000), np.log10(40000), n_high)
            else:
                poles_freq = np.logspace(np.log10(10), np.log10(200), n_low)
                zeros_freq = np.logspace(np.log10(2000), np.log10(40000), n_high)
                
        else:
            # medium slopes 
            poles_freq = []
            zeros_freq = []
            
            # Distribute more evenly across frequency range
            if abs_slope <= 1.5:
                # tight distribution around pivot for smaller medium slopes
                f_start = 50
                f_end = 20000
            else:
                # wider distribution for 1.5 - 5.0 slopes
                f_start = 20
                f_end = 20000
            
            frequencies = np.logspace(np.log10(f_start), np.log10(f_end), N)
            
            # Sort into poles and zeros based on slope sign
            for i, f in enumerate(frequencies):
                if slope_db_per_octave < 0:
                    if i % 2 == 0:
                        zeros_freq.append(f) # even = zeros
                    else:
                        poles_freq.append(f)
                else:
                    if i % 2 == 0:
                        poles_freq.append(f)
                    else:
                        zeros_freq.append(f)
        
        poles_freq = np.array(poles_freq)
        zeros_freq = np.array(zeros_freq)
        
        print(f"Initial pole frequencies: {poles_freq}")
        print(f"Initial zero frequencies: {zeros_freq}")
        
        poles_init = -2 * np.pi * poles_freq
        zeros_init = -2 * np.pi * zeros_freq
        
        abs_slope = abs(slope_db_per_octave)
        
        poles_opt, zeros_opt = self.optimize_placement(
            poles_init, zeros_init, slope_db_per_octave, abs_slope
        )
        
        # Calculate gain for 0 dB at 1 kHz
        gain = self.calculate_gain(zeros_opt, poles_opt)
        print(f"Gain: {gain:.4f} ({20*np.log10(gain):.2f} dB)")
        
        sos = zpk2sos(zeros_opt, poles_opt, gain)
        
        self.plot_response(zeros_opt, poles_opt, gain, slope_db_per_octave)
        
        return zeros_opt, poles_opt, gain, sos
    
    def optimize_placement(self, poles_init, zeros_init, slope_db_per_octave, abs_slope):
        
        # sample filter at many freqs, extra around pivot
        f_log = np.logspace(np.log10(20), np.log10(20000), 300)
        # Add extra points around pivot
        f_pivot = np.logspace(np.log10(500), np.log10(2000), 100)
        f = np.sort(np.concatenate([f_log, f_pivot]))
        
        target_db = self.tilt_target(f, slope_db_per_octave)
        
        x0 = np.concatenate([
            -poles_init / (2 * np.pi),
            -zeros_init / (2 * np.pi)
        ])
        
        def error_func(x):
            n_poles = len(poles_init)
            pole_freqs = np.abs(x[:n_poles])
            zero_freqs = np.abs(x[n_poles:])
            
            poles = -2 * np.pi * pole_freqs
            zeros = -2 * np.pi * zero_freqs
            
            gain = self.calculate_gain(zeros, poles)
            
            omega = 2 * np.pi * f
            s = 1j * omega
            
            H = gain * np.ones_like(s, dtype=complex)
            for z in zeros:
                H *= (s - z)
            for p in poles:
                H /= (s - p)
            
            response_db = 20 * np.log10(np.abs(H))
            
            # Adaptive weighting based on slope
            weights = np.ones_like(f)
            
            if abs_slope <= 1.0:
                # small slopes: emphasize entire band equally
                weights[:] = 1.0
                # Extra weight at test frequencies
                for test_f in [100, 1000, 10000]:
                    idx = np.argmin(np.abs(f - test_f))
                    weights[idx] *= 5
            else:
                # large slopes: emphasize mid-band more
                weights[(f >= 100) & (f <= 10000)] = 2.0
                # Extra weight around pivot
                weights[(f >= 500) & (f <= 2000)] = 3.0
            
            error = (response_db - target_db) * weights
            
            regularization = 0
            
            if abs_slope <= 0.5:
                # Poles and zeros should be close
                for i in range(min(len(pole_freqs), len(zero_freqs))):
                    ratio = max(pole_freqs[i], zero_freqs[i]) / min(pole_freqs[i], zero_freqs[i])
                    if ratio > 2: # more than 1 octave apart
                        regularization += (ratio - 2) ** 2
            
            # For all slopes, penalize clustering
            all_freqs = np.concatenate([pole_freqs, zero_freqs])
            all_freqs.sort()
            for i in range(len(all_freqs) - 1):
                ratio = all_freqs[i+1] / all_freqs[i]
                if ratio < 1.2:  # Too close (less than 1/3 octave)
                    regularization += 10 * (1.2 - ratio) ** 2
            
            # For medium slopes, encourage even distribution
            if 1.0 <= abs_slope <= 3.0:
                if len(all_freqs) > 2:
                    log_freqs = np.log10(all_freqs)
                    log_diffs = np.diff(log_freqs)
                    spacing_variance = np.var(log_diffs)
                    regularization += 100 * spacing_variance
            
            return np.sum(error**2) + regularization
        
        # Adaptive bounds based on slope
        bounds = []
        for i, freq in enumerate(x0):
            if abs_slope <= 0.5:
                bounds.append((50, 20000))
            elif abs_slope >= 5.0:
                bounds.append((5, 50000))
            else:
                bounds.append((10, 40000))
        
        best_result = None
        best_error = float('inf')
        
        if abs_slope <= 1.0:
            methods = ['L-BFGS-B', 'TNC', 'SLSQP']
            max_iter = 1000
        else:
            methods = ['L-BFGS-B', 'TNC']
            max_iter = 2000  # More iterations for harder problems
        
        for method in methods:
            try:
                if method == 'SLSQP':
                    result = optimize.minimize(
                        error_func,
                        x0,
                        method=method,
                        options={'ftol': 1e-10, 'maxiter': max_iter}
                    )
                else:
                    result = optimize.minimize(
                        error_func,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'ftol': 1e-10, 'gtol': 1e-10, 'maxfun': max_iter}
                    )
                
                if result.fun < best_error:
                    best_error = result.fun
                    best_result = result
                    
            except Exception as e:
                print(f"  Optimization with {method} failed: {e}")
                continue
        
            n_poles = len(poles_init)
            pole_freqs_opt = np.abs(best_result.x[:n_poles])
            zero_freqs_opt = np.abs(best_result.x[n_poles:])
            
            pole_freqs_opt.sort()
            zero_freqs_opt.sort()
            
            poles_opt = -2 * np.pi * pole_freqs_opt
            zeros_opt = -2 * np.pi * zero_freqs_opt
            
            print(f"  Optimization succeeded with {best_result.message}")
        
        print(f"Optimized pole frequencies: {-poles_opt/(2*np.pi)}")
        print(f"Optimized zero frequencies: {-zeros_opt/(2*np.pi)}")
        
        return poles_opt, zeros_opt
    
    def calculate_gain(self, zeros, poles):
        omega_pivot = 2 * np.pi * self.pivot_freq
        s_pivot = 1j * omega_pivot
        
        H_pivot = 1.0
        for z in zeros:
            H_pivot *= (s_pivot - z)
        for p in poles:
            H_pivot /= (s_pivot - p)
        
        return 1.0 / np.abs(H_pivot)
    
    def plot_response(self, zeros, poles, gain, slope_db_per_octave):
        
        f = np.logspace(np.log10(20), np.log10(20000), 500)
        target_db = self.tilt_target(f, slope_db_per_octave)
        
        omega = 2 * np.pi * f
        s = 1j * omega
        
        H = gain * np.ones_like(s, dtype=complex)
        for z in zeros:
            H *= (s - z)
        for p in poles:
            H /= (s - p)
        
        response_db = 20 * np.log10(np.abs(H))
        error = response_db - target_db
        
        fig = plt.figure(figsize=(12, 10))
        
        # Main response plot
        ax1 = plt.subplot(3, 1, 1)
        plt.semilogx(f, target_db, 'b-', label='Target', linewidth=2)
        plt.semilogx(f, response_db, 'r--', label='Actual', linewidth=2)
        plt.axhline(0, color='gray', linestyle=':', alpha=0.3)
        plt.axvline(1000, color='gray', linestyle=':', alpha=0.3)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title(f'Tilt Filter: {slope_db_per_octave} dB/octave')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(20, 20000)
        
        # Error plot
        ax2 = plt.subplot(3, 1, 2)
        plt.semilogx(f, error, 'g-', linewidth=2)
        plt.axhline(0.1, color='r', linestyle='--', alpha=0.5)
        plt.axhline(-0.1, color='r', linestyle='--', alpha=0.5)
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Error (dB)')
        plt.title('Error')
        plt.grid(True, alpha=0.3)
        plt.xlim(20, 20000)
        plt.ylim(-1, 1)
        
        # Pole-zero plot
        ax3 = plt.subplot(3, 1, 3)
        pole_freqs = -poles / (2 * np.pi)
        zero_freqs = -zeros / (2 * np.pi)
        
        # Plot poles and zeros on log frequency axis
        if len(pole_freqs) > 0:
            plt.scatter(pole_freqs, np.zeros_like(pole_freqs), 
                       marker='x', s=200, c='red', linewidth=3, label='Poles')
            for pf in pole_freqs:
                plt.axvline(pf, color='red', alpha=0.2, linestyle='--')
        
        if len(zero_freqs) > 0:
            plt.scatter(zero_freqs, np.ones_like(zero_freqs), 
                       marker='o', s=200, facecolors='none', edgecolors='blue', 
                       linewidth=3, label='Zeros')
            for zf in zero_freqs:
                plt.axvline(zf, color='blue', alpha=0.2, linestyle='--')
        
        plt.axvline(1000, color='gray', linestyle=':', alpha=0.5, label='Pivot (1 kHz)')
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Type')
        plt.title('Pole-Zero Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(10, 100000)
        plt.ylim(-0.5, 1.5)
        plt.yticks([0, 1], ['Pole', 'Zero'])
        
        plt.tight_layout()
        plt.show()
        
        max_error = np.max(np.abs(error))
        print(f"\nVerification:")
        print(f"Max error: {max_error:.3f} dB")
        
        test_freqs = [100, 1000, 10000]
        for tf in test_freqs:
            idx = np.argmin(np.abs(f - tf))
            print(f"  {tf} Hz: target={target_db[idx]:.2f} dB, "
                  f"actual={response_db[idx]:.2f} dB, error={error[idx]:.3f} dB")
        
        return max_error < 0.1


if __name__ == "__main__":
    designer = TiltFilter(fs=48000)
    
    test_slopes = [-6.0, -3.0, -2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 3.0, 6.0]
    
    print("="*80)
    print("TILT FILTER DESIGN")
    print("="*80)
    
    for slope in test_slopes:
        # Adaptive order selection based on slope
        abs_slope = abs(slope)
        if abs_slope < 0.1:
            N = 0
        elif abs_slope <= 0.5:
            N = 6 
        elif abs_slope <= 1.0:
            N = 8
        elif abs_slope <= 2.0:
            N = 10  
        elif abs_slope <= 4.0:
            N = 8  
        else:
            N = 6  
        
        zeros, poles, gain, sos = designer.design_tilt_filter(slope, N)
        print("-" * 80)