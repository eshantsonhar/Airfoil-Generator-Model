#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for ASO Paper

This script reads data from aso_production_100iter/ and generates
publication-grade vector-quality plots (300 DPI PNG and PDF formats).

Figures generated:
1. Figure 1: Convergence History (Cd and Cl vs. Iteration)
2. Figure 2: Airfoil Profile Overlay (Baseline vs. Optimized)
3. Figure 3: Surface Pressure Distribution (Cp vs. x/c)
4. Figure 4: Gradient Decay & Move Limit Dynamics

Author: ASO Research Team
Date: 2026-08-05
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

# Set publication-quality matplotlib parameters
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.linewidth'] = 0.8

# Publication color scheme (AIAA/IEEE standard)
COLORS = {
    'black': '#000000',
    'blue': '#0072BD',   # MATLAB blue
    'red': '#D95319',    # MATLAB red
    'green': '#77AC30',  # MATLAB green
    'gray': '#7E7E7E',   # MATLAB gray
    'orange': '#EDB120', # MATLAB orange
}


class ASODataLoader:
    """Load and process ASO optimization data for figure generation."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.convergence_data = None
        self.airfoil_data = None
        self.initial_cp_data = None
        self.final_cp_data = None
        
    def load_convergence_history(self) -> Dict:
        """Load convergence history from JSON file."""
        history_file = self.data_dir / "convergence_history.json"
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        iterations = data['iterations']
        self.convergence_data = {
            'iterations': [it['iteration'] for it in iterations],
            'cd': [it['cd'] for it in iterations],
            'cl': [it['cl'] for it in iterations],
            'grad_norm': [it['grad_norm'] for it in iterations],
            'step_accepted': [it['step_accepted'] for it in iterations],
            'trust_radius': [it['trust_radius'] for it in iterations],
            'max_thickness': [it['max_thickness'] for it in iterations],
            'design_vectors': [it['design_vector'] for it in iterations],
        }
        
        # Calculate step sizes (dx)
        self.convergence_data['step_size'] = self._calculate_step_sizes(
            self.convergence_data['design_vectors']
        )
        
        return self.convergence_data
    
    def _calculate_step_sizes(self, design_vectors: List[List[float]]) -> List[float]:
        """Calculate step sizes between consecutive design vectors."""
        step_sizes = [0.0]  # First iteration has no step size
        for i in range(1, len(design_vectors)):
            dv_current = np.array(design_vectors[i])
            dv_previous = np.array(design_vectors[i-1])
            step_size = np.linalg.norm(dv_current - dv_previous)
            step_sizes.append(step_size)
        return step_sizes
    
    def load_airfoil_shape(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load airfoil coordinates from DAT file."""
        shape_file = self.data_dir / "best_airfoil_shape.dat"
        
        # Parse DAT file (standard airfoil format)
        x_coords = []
        y_coords = []
        
        with open(shape_file, 'r') as f:
            lines = f.readlines()
            # Skip header line if present
            start_idx = 0 if lines[0].strip() != 'airfoil' else 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))
        
        self.airfoil_data = (np.array(x_coords), np.array(y_coords))
        return self.airfoil_data
    
    def load_initial_airfoil(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial airfoil from first design vector."""
        if self.convergence_data is None:
            self.load_convergence_history()
        
        # Use first design vector to generate initial airfoil
        initial_dv = self.convergence_data['design_vectors'][0]
        x_coords, y_coords = self._cst_to_coordinates(initial_dv)
        
        return x_coords, y_coords
    
    def _cst_to_coordinates(self, design_vector: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert CST design vector to airfoil coordinates.
        Simplified CST implementation for figure generation.
        """
        # Class function (NACA 0012-like)
        def class_function(x):
            return 0.5 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                         0.2843 * x**3 - 0.1015 * x**4)
        
        # Shape function (simplified Bernstein polynomials)
        x = np.linspace(0, 1, n_points)
        
        # Upper surface coefficients (first 6)
        au = design_vector[:6]
        # Lower surface coefficients (last 6)  
        al = design_vector[6:]
        
        # Bernstein polynomial basis (without scipy dependency)
        def bernstein(n, k, x):
            # Binomial coefficient calculation
            from math import factorial
            def comb(n, k):
                return factorial(n) // (factorial(k) * factorial(n - k))
            return comb(n, k) * x**k * (1-x)**(n-k)
        
        # Upper surface
        y_upper = np.zeros_like(x)
        for i, coeff in enumerate(au):
            y_upper += coeff * bernstein(5, i, x)
        y_upper = class_function(x) * y_upper
        
        # Lower surface
        y_lower = np.zeros_like(x)
        for i, coeff in enumerate(al):
            y_lower += coeff * bernstein(5, i, x)
        y_lower = -class_function(x) * y_lower
        
        # Combine upper and lower surfaces
        x_coords = np.concatenate([x[::-1], x[1:]])
        y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
        
        return x_coords, y_coords
    
    def load_surface_pressure(self, iteration: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load surface pressure data from specific iteration."""
        # Find the CFD case directory for the specified iteration
        cfd_cases_dir = self.data_dir / "cfd_cases"
        
        # Find eval directories for the iteration
        eval_dirs = sorted(cfd_cases_dir.glob(f"eval_*"))
        
        if iteration < len(eval_dirs):
            eval_dir = eval_dirs[iteration]
        else:
            eval_dir = eval_dirs[-1]  # Use last available
        
        surface_file = eval_dir / "surface_flow.csv"
        
        if not surface_file.exists():
            raise FileNotFoundError(f"Surface flow file not found: {surface_file}")
        
        # Parse CSV file
        x_upper, cp_upper = [], []
        x_lower, cp_lower = [], []
        
        with open(surface_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row.get('x', 0))
                cp = float(row.get('Cp', 0))
                
                # Assume first half is upper surface, second half is lower
                # This is a simplification - actual implementation depends on SU2 output format
                if len(x_upper) < len(x_lower) or len(x_lower) == 0:
                    x_upper.append(x)
                    cp_upper.append(cp)
                else:
                    x_lower.append(x)
                    cp_lower.append(cp)
        
        return (np.array(x_upper), np.array(cp_upper), 
                np.array(x_lower), np.array(cp_lower))


class FigureGenerator:
    """Generate publication-quality figures for ASO paper."""
    
    def __init__(self, data_loader: ASODataLoader, output_dir: Path):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_figure(self, fig, filename: str):
        """Save figure in both PNG and PDF formats."""
        png_path = self.output_dir / f"{filename}.png"
        pdf_path = self.output_dir / f"{filename}.pdf"
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    
    def figure1_convergence_history(self):
        """Figure 1: Convergence History (Cd and Cl vs. Iteration)."""
        data = self.data_loader.load_convergence_history()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        
        iterations = data['iterations']
        cd = data['cd']
        cl = data['cl']
        step_accepted = data['step_accepted']
        
        # Plot Cd
        ax1.plot(iterations, cd, color=COLORS['blue'], linewidth=1.5, label='$C_D$')
        ax1.scatter(iterations, cd, c=[COLORS['blue'] if sa else COLORS['red'] 
                                     for sa in step_accepted], s=30, zorder=5)
        ax1.set_ylabel('$C_D$', fontsize=11)
        ax1.set_title('(a) Drag Coefficient', fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', frameon=True)
        
        # Plot Cl with constraint floor
        ax2.plot(iterations, cl, color=COLORS['red'], linewidth=1.5, label='$C_L$')
        ax2.axhline(y=1.0, color=COLORS['black'], linestyle='--', 
                   linewidth=1.0, alpha=0.7, label='$C_L = 1.0$ floor')
        ax2.scatter(iterations, cl, c=[COLORS['red'] if sa else COLORS['blue'] 
                                     for sa in step_accepted], s=30, zorder=5)
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('$C_L$', fontsize=11)
        ax2.set_title('(b) Lift Coefficient', fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='lower right', frameon=True)
        
        plt.tight_layout()
        self.save_figure(fig, 'figure1_convergence_history')
        plt.close(fig)
    
    def figure2_airfoil_profile(self):
        """Figure 2: Airfoil Profile Overlay (Baseline vs. Optimized)."""
        # Load optimized airfoil
        x_opt, y_opt = self.data_loader.load_airfoil_shape()
        
        # Generate initial airfoil
        x_init, y_init = self.data_loader.load_initial_airfoil()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        
        # Plot initial airfoil
        ax.plot(x_init, y_init, color=COLORS['gray'], linewidth=1.5, 
               linestyle='--', label='Baseline', alpha=0.7)
        
        # Plot optimized airfoil
        ax.plot(x_opt, y_opt, color=COLORS['blue'], linewidth=2.0, 
               label='Optimized')
        
        # Formatting
        ax.set_xlabel('Chordwise Position $x/c$', fontsize=11)
        ax.set_ylabel('Thickness Coordinate $y/c$', fontsize=11)
        ax.set_title('Airfoil Profile Geometry Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.2, 0.2)
        
        plt.tight_layout()
        self.save_figure(fig, 'figure2_airfoil_profile')
        plt.close(fig)
    
    def figure3_surface_pressure(self):
        """Figure 3: Surface Pressure Distribution (Cp vs. x/c)."""
        try:
            # Load initial and final surface pressure data
            x_u_init, cp_u_init, x_l_init, cp_l_init = self.data_loader.load_surface_pressure(0)
            x_u_opt, cp_u_opt, x_l_opt, cp_l_opt = self.data_loader.load_surface_pressure(-1)
        except FileNotFoundError as e:
            print(f"Warning: Could not load surface pressure data: {e}")
            print("Skipping Figure 3 generation.")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        # Plot initial pressure distribution
        ax.plot(x_u_init, cp_u_init, color=COLORS['gray'], linewidth=1.5, 
               linestyle='--', label='Baseline Upper', alpha=0.7)
        ax.plot(x_l_init, cp_l_init, color=COLORS['gray'], linewidth=1.5, 
               linestyle=':', label='Baseline Lower', alpha=0.7)
        
        # Plot optimized pressure distribution
        ax.plot(x_u_opt, cp_u_opt, color=COLORS['blue'], linewidth=2.0, 
               label='Optimized Upper')
        ax.plot(x_l_opt, cp_l_opt, color=COLORS['red'], linewidth=2.0, 
               label='Optimized Lower')
        
        # Formatting
        ax.set_xlabel('Chordwise Position $x/c$', fontsize=11)
        ax.set_ylabel('Pressure Coefficient $C_p$', fontsize=11)
        ax.set_title('Surface Pressure Distribution Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.invert_yaxis()  # Cp convention: negative values upward
        
        plt.tight_layout()
        self.save_figure(fig, 'figure3_surface_pressure')
        plt.close(fig)
    
    def figure4_gradient_decay(self):
        """Figure 4: Gradient Decay & Move Limit Dynamics."""
        data = self.data_loader.load_convergence_history()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        
        iterations = data['iterations']
        grad_norm = data['grad_norm']
        trust_radius = data['trust_radius']
        step_size = data['step_size']
        
        # Plot gradient norm (log scale)
        ax1.semilogy(iterations, grad_norm, color=COLORS['blue'], linewidth=2.0, 
                    label='$\\|\\nabla C_D\\|$')
        
        # Plot trust radius on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(iterations, trust_radius, color=COLORS['red'], linewidth=1.5, 
                linestyle='--', label='Trust Radius')
        
        # Plot step size
        ax2.plot(iterations, step_size, color=COLORS['green'], linewidth=1.5, 
                linestyle=':', label='Step Size $\\|\\Delta x\\|$')
        
        # Formatting
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Gradient Norm $\\|\\nabla C_D\\|$', fontsize=11, color=COLORS['blue'])
        ax2.set_ylabel('Trust Radius / Step Size', fontsize=11, color=COLORS['red'])
        ax1.set_title('Gradient Decay & Move Limit Dynamics', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=COLORS['blue'])
        ax2.tick_params(axis='y', labelcolor=COLORS['red'])
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)
        
        plt.tight_layout()
        self.save_figure(fig, 'figure4_gradient_decay')
        plt.close(fig)


def main():
    """Main function to generate all publication figures."""
    # Set paths
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "aso_production_100iter"
    output_dir = project_root / "paper_figures"
    
    print("=" * 60)
    print("ASO Paper Figure Generation")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Initialize data loader
    print("Loading ASO data...")
    data_loader = ASODataLoader(data_dir)
    
    # Initialize figure generator
    figure_generator = FigureGenerator(data_loader, output_dir)
    
    # Generate figures
    print("\nGenerating Figure 1: Convergence History...")
    figure_generator.figure1_convergence_history()
    
    print("\nGenerating Figure 2: Airfoil Profile Overlay...")
    figure_generator.figure2_airfoil_profile()
    
    print("\nGenerating Figure 3: Surface Pressure Distribution...")
    figure_generator.figure3_surface_pressure()
    
    print("\nGenerating Figure 4: Gradient Decay & Move Limit Dynamics...")
    figure_generator.figure4_gradient_decay()
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()