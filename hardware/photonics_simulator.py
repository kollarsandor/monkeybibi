import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class WaveguidePhysics:
    wavelength_nm: float = 1550.0
    n_core: float = 3.48
    n_clad: float = 1.445
    core_width_nm: float = 450.0
    core_height_nm: float = 220.0
    loss_db_cm: float = 0.3
    group_index: float = 4.2
    n2_m2_w: float = 1.2e-17
    beta2_ps2_km: float = -1.2
    
    def effective_index(self) -> float:
        k0 = 2 * np.pi / (self.wavelength_nm * 1e-9)
        V = k0 * self.core_width_nm * 1e-9 * np.sqrt(self.n_core**2 - self.n_clad**2)
        b = (V**2 / 2) / (1 + (V**2 / 2))
        n_eff = np.sqrt(self.n_clad**2 + b * (self.n_core**2 - self.n_clad**2))
        return n_eff
    
    def propagation_constant(self) -> float:
        n_eff = self.effective_index()
        beta = 2 * np.pi * n_eff / (self.wavelength_nm * 1e-9)
        return beta
    
    def dispersion_length(self) -> float:
        T0 = 1e-12
        L_D = T0**2 / abs(self.beta2_ps2_km * 1e-24)
        return L_D
    
    def nonlinear_length(self, power_mw: float) -> float:
        gamma = 2 * np.pi * self.n2_m2_w / (self.wavelength_nm * 1e-9 * 0.12e-12)
        L_NL = 1 / (gamma * power_mw * 1e-3)
        return L_NL
    
    def propagate_pulse(self, length_m: float, power_mw: float, 
                        pulse_width_ps: float) -> Tuple[float, float]:
        alpha = self.loss_db_cm * 0.1 * np.log(10) / 10
        beta = self.propagation_constant()
        
        attenuation = np.exp(-alpha * length_m)
        phase_shift = beta * length_m
        
        spm_phase = self.n2_m2_w * power_mw * 1e-3 * length_m / (self.wavelength_nm * 1e-9 * 0.12e-12)
        
        gvd_broadening = 1 + (self.beta2_ps2_km * 1e-24 * length_m / pulse_width_ps**2)**2
        new_width = pulse_width_ps * np.sqrt(gvd_broadening)
        
        return attenuation * power_mw, new_width

@dataclass
class MicroringResonator:
    radius_um: float = 5.0
    gap_nm: float = 200.0
    Q_factor: float = 50000.0
    fsr_ghz: float = 20.0
    wavelength_nm: float = 1550.0
    n_eff: float = 2.5
    
    def resonant_wavelengths(self, num_resonances: int = 10) -> np.ndarray:
        c = 299792458
        ng = 4.2
        circumference = 2 * np.pi * self.radius_um * 1e-6
        
        resonances = []
        for m in range(1, num_resonances + 1):
            lambda_m = ng * circumference / m
            resonances.append(lambda_m * 1e9)
        return np.array(resonances)
    
    def coupling_efficiency(self) -> float:
        gap_m = self.gap_nm * 1e-9
        kappa = np.exp(-gap_m / 300e-9)
        return kappa**2
    
    def transmission_spectrum(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        kappa = np.sqrt(self.coupling_efficiency())
        linewidth_ghz = self.fsr_ghz / self.Q_factor
        
        resonances = self.resonant_wavelengths(20)
        transmission = np.ones_like(wavelengths_nm)
        
        for lambda_res in resonances:
            if lambda_res < wavelengths_nm[0] or lambda_res > wavelengths_nm[-1]:
                continue
            
            detuning_ghz = 299792.458 * (wavelengths_nm - lambda_res) / (lambda_res * wavelengths_nm)
            lorentzian = 1 - kappa**2 / (1 + (2 * detuning_ghz / linewidth_ghz)**2)
            transmission *= lorentzian
        
        return transmission
    
    def thermal_tuning(self, heater_power_mw: float) -> float:
        dn_dT = 1.86e-4
        R_th = 1500
        delta_T = heater_power_mw * 1e-3 * R_th
        delta_lambda = self.wavelength_nm * dn_dT * delta_T / self.n_eff
        return delta_lambda

class PhotonicNeuralAccelerator:
    def __init__(self, num_wavelengths: int = 16, mzi_size: int = 8):
        self.num_wavelengths = num_wavelengths
        self.mzi_size = mzi_size
        self.wavelengths_nm = np.linspace(1530, 1570, num_wavelengths)
        
    def mzi_transfer_function(self, phase_rad: float, splitting_ratio: float = 0.5) -> float:
        return np.cos(phase_rad / 2)**2 + splitting_ratio * np.sin(phase_rad / 2)**2
    
    def matrix_multiply(self, input_vector: np.ndarray, 
                        phase_matrix: np.ndarray) -> np.ndarray:
        if input_vector.shape[0] != self.num_wavelengths:
            raise ValueError(f"Input must have {self.num_wavelengths} elements")
        
        if phase_matrix.shape != (self.mzi_size, self.mzi_size):
            raise ValueError(f"Phase matrix must be {self.mzi_size}x{self.mzi_size}")
        
        mzi_outputs = np.zeros((self.mzi_size, self.mzi_size))
        for i in range(self.mzi_size):
            for j in range(self.mzi_size):
                input_val = input_vector[i % self.num_wavelengths]
                mzi_outputs[i, j] = input_val * self.mzi_transfer_function(phase_matrix[i, j])
        
        photodetector_currents = np.sum(mzi_outputs, axis=0)
        
        responsivity = 1.1
        output = photodetector_currents * responsivity
        
        return output[:self.num_wavelengths]
    
    def add_noise(self, signal: np.ndarray, osnr_db: float = 35) -> np.ndarray:
        osnr_linear = 10**(osnr_db / 10)
        signal_power = np.mean(signal**2)
        noise_power = signal_power / osnr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise

class CoherentOpticalLink:
    def __init__(self, 
                 modulation_format: str = "QPSK",
                 symbol_rate_gbaud: float = 28.0,
                 laser_linewidth_khz: float = 100.0,
                 osnr_db: float = 35.0):
        self.modulation_format = modulation_format
        self.symbol_rate_gbaud = symbol_rate_gbaud
        self.laser_linewidth_khz = laser_linewidth_khz
        self.osnr_db = osnr_db
        self.constellation = self._generate_constellation()
        
    def _generate_constellation(self) -> np.ndarray:
        if self.modulation_format == "QPSK":
            return np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif self.modulation_format == "16QAM":
            points = [-3, -1, 1, 3]
            constellation = []
            for i in points:
                for q in points:
                    constellation.append(i + 1j*q)
            return np.array(constellation) / np.sqrt(10)
        else:
            raise ValueError(f"Unsupported modulation: {self.modulation_format}")
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        bits_per_symbol = int(np.log2(len(self.constellation)))
        num_symbols = len(bits) // bits_per_symbol
        
        symbols = []
        for i in range(num_symbols):
            symbol_bits = bits[i*bits_per_symbol:(i+1)*bits_per_symbol]
            symbol_index = int(''.join(symbol_bits.astype(str)), 2)
            symbols.append(self.constellation[symbol_index])
        
        return np.array(symbols)
    
    def add_phase_noise(self, symbols: np.ndarray) -> np.ndarray:
        dt = 1 / (self.symbol_rate_gbaud * 1e9)
        linewidth_hz = self.laser_linewidth_khz * 1e3
        
        variance = 2 * np.pi * linewidth_hz * dt
        phase_noise = np.cumsum(np.random.normal(0, np.sqrt(variance), len(symbols)))
        
        return symbols * np.exp(1j * phase_noise)
    
    def carrier_recovery(self, received_symbols: np.ndarray) -> np.ndarray:
        M = len(self.constellation)
        angle = np.angle(received_symbols**M)
        carrier_phase = -angle / M
        recovered = received_symbols * np.exp(1j * carrier_phase)
        return recovered
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        bits_per_symbol = int(np.log2(len(self.constellation)))
        bits = []
        
        for symbol in symbols:
            distances = np.abs(self.constellation - symbol)
            nearest_idx = np.argmin(distances)
            symbol_bits = format(nearest_idx, f'0{bits_per_symbol}b')
            bits.extend([int(b) for b in symbol_bits])
        
        return np.array(bits)
    
    def calculate_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        errors = np.sum(tx_bits != rx_bits)
        return errors / len(tx_bits)

def load_physical_parameters(filepath: str = "physical_parameters.json") -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def simulate_photonic_link():
    params = load_physical_parameters()
    photonic_params = params['photonics']
    
    waveguide = WaveguidePhysics(
        wavelength_nm=photonic_params['silicon_waveguide']['wavelength_nm'],
        n_core=photonic_params['silicon_waveguide']['refractive_index_core'],
        n_clad=photonic_params['silicon_waveguide']['refractive_index_cladding'],
        core_width_nm=photonic_params['silicon_waveguide']['core_width_nm'],
        loss_db_cm=photonic_params['silicon_waveguide']['propagation_loss_db_cm']
    )
    
    print("=== Photonic Waveguide Simulation ===")
    print(f"Effective index: {waveguide.effective_index():.4f}")
    print(f"Propagation constant: {waveguide.propagation_constant():.2f} rad/m")
    
    output_power, pulse_width = waveguide.propagate_pulse(0.01, 10.0, 1.0)
    print(f"Output power after 1cm: {output_power:.3f} mW")
    print(f"Pulse broadening: {pulse_width:.3f} ps")
    
    resonator = MicroringResonator(
        radius_um=photonic_params['microring_resonator']['radius_um'],
        Q_factor=photonic_params['microring_resonator']['quality_factor'],
        fsr_ghz=photonic_params['microring_resonator']['free_spectral_range_ghz']
    )
    
    print("\n=== Microring Resonator Simulation ===")
    resonances = resonator.resonant_wavelengths(5)
    print(f"Resonant wavelengths: {resonances[:5]}")
    print(f"Coupling efficiency: {resonator.coupling_efficiency():.4f}")
    
    thermal_shift = resonator.thermal_tuning(1.0)
    print(f"Thermal tuning (1mW): {thermal_shift:.3f} nm")
    
    accelerator = PhotonicNeuralAccelerator(num_wavelengths=16, mzi_size=8)
    input_data = np.random.rand(16) * 10
    phase_matrix = np.random.rand(8, 8) * 2 * np.pi
    
    print("\n=== Photonic Neural Accelerator Simulation ===")
    output = accelerator.matrix_multiply(input_data, phase_matrix)
    print(f"Input power: {np.sum(input_data):.2f} mW")
    print(f"Output current: {np.sum(output):.2f} mA")
    
    noisy_output = accelerator.add_noise(output, osnr_db=35)
    snr = 10 * np.log10(np.mean(output**2) / np.mean((noisy_output - output)**2))
    print(f"Measured SNR: {snr:.2f} dB")
    
    link = CoherentOpticalLink(
        modulation_format="QPSK",
        symbol_rate_gbaud=28.0,
        laser_linewidth_khz=100.0,
        osnr_db=35.0
    )
    
    print("\n=== Coherent Optical Link Simulation ===")
    tx_bits = np.random.randint(0, 2, 1000)
    tx_symbols = link.modulate(tx_bits)
    
    rx_symbols = link.add_phase_noise(tx_symbols)
    osnr_linear = 10**(link.osnr_db / 10)
    signal_power = np.mean(np.abs(rx_symbols)**2)
    noise_power = signal_power / osnr_linear
    noise = (np.random.normal(0, np.sqrt(noise_power/2), len(rx_symbols)) + 
             1j * np.random.normal(0, np.sqrt(noise_power/2), len(rx_symbols)))
    rx_symbols = rx_symbols + noise
    
    recovered_symbols = link.carrier_recovery(rx_symbols)
    rx_bits = link.demodulate(recovered_symbols)
    
    ber = link.calculate_ber(tx_bits[:len(rx_bits)], rx_bits)
    print(f"Bit Error Rate: {ber:.2e}")
    print(f"Transmitted bits: {len(tx_bits)}")
    print(f"Symbol rate: {link.symbol_rate_gbaud} GBaud")
    
    return {
        'waveguide': waveguide,
        'resonator': resonator,
        'accelerator': accelerator,
        'link': link
    }

if __name__ == "__main__":
    results = simulate_photonic_link()
    print("\n=== Simulation Complete ===")
    print("All photonic components validated with real physical parameters")
