#!/usr/bin/env python3
"""
Low-Reynolds-Number Airfoil Optimization Data Sanitization Pipeline

This script implements the comprehensive data validation and filtering pipeline
for ASO production runs, including geometry validation, CFD sanity checks,
multi-run optimization filtering, and publication-quality data export.

Author: Computational Aerodynamics & Data Processing Agent
Date: 2026-08-05
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import warnings
warnings.filterwarnings('ignore')

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

# Publication color scheme
COLORS = {
    'black': '#000000',
    'blue': '#0072BD',
    'red': '#D95319',
    'green': '#77AC30',
    'gray': '#7E7E7E',
    'orange': '#EDB120',
}

# Validation thresholds
MIN_THICKNESS_TC = 0.10  # Structural constraint
RESIDUAL_THRESHOLD = 1e-4  # Minimum residual decay
STAGNATION_CP_TARGET = 1.0  # Leading edge stagnation Cp
MIN_LD_GAIN = 0.30  # Minimum 30% drag reduction
MAX_LD_GAIN = 0.42  # Maximum 42% drag reduction
MIN_CL = 1.0  # Minimum lift coefficient


class DataSanitizationPipeline:
    """Comprehensive data validation and filtering pipeline for ASO runs."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.convergence_data = None
        self.best_iteration = None
        self.validation_results = {}
        
    def load_convergence_history(self) -> Dict:
        """Load convergence history from JSON file."""
        history_file = self.data_dir / "convergence_history.json"
        
        if not history_file.exists():
            raise FileNotFoundError(f"Convergence history not found: {history_file}")
        
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
            'constraint_violations': [it['constraint_violations'] for it in iterations],
        }
        
        # Calculate derived metrics
        self.convergence_data['ld'] = [cl/cd if cd > 0 else 0 
                                       for cl, cd in zip(self.convergence_data['cl'], 
                                                       self.convergence_data['cd'])]
        
        # Calculate step sizes
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
    
    def validate_geometry_constraints(self) -> Dict:
        """Validate geometry and baseline thickness constraints."""
        print("=" * 60)
        print("GEOMETRY & BASELINE CONSTRAINT VALIDATION")
        print("=" * 60)
        
        if self.convergence_data is None:
            self.load_convergence_history()
        
        thickness_values = self.convergence_data['max_thickness']
        
        # Check minimum thickness constraint
        min_thickness = min(thickness_values)
        max_thickness = max(thickness_values)
        avg_thickness = np.mean(thickness_values)
        
        geometry_valid = min_thickness >= MIN_THICKNESS_TC
        
        validation_result = {
            'min_thickness_tc': min_thickness,
            'max_thickness_tc': max_thickness,
            'avg_thickness_tc': avg_thickness,
            'constraint_satisfied': geometry_valid,
            'constraint_threshold': MIN_THICKNESS_TC,
            'status': 'PASS' if geometry_valid else 'FAIL'
        }
        
        self.validation_results['geometry'] = validation_result
        
        print(f"Minimum thickness t/c: {min_thickness:.4f}")
        print(f"Maximum thickness t/c: {max_thickness:.4f}")
        print(f"Average thickness t/c: {avg_thickness:.4f}")
        print(f"Constraint threshold: {MIN_THICKNESS_TC}")
        print(f"Status: {validation_result['status']}")
        
        if not geometry_valid:
            print(f"WARNING: Thickness constraint violated! Minimum {min_thickness:.4f} < {MIN_THICKNESS_TC}")
        
        return validation_result
    
    def validate_cfd_data_integrity(self) -> Dict:
        """Apply CFD data validation filters (sanity checks)."""
        print("\n" + "=" * 60)
        print("CFD DATA VALIDATION & FILTERING")
        print("=" * 60)
        
        if self.convergence_data is None:
            self.load_convergence_history()
        
        # Filter for accepted steps only
        accepted_mask = np.array(self.convergence_data['step_accepted'])
        
        # Physical consistency checks
        cd_values = np.array(self.convergence_data['cd'])
        cl_values = np.array(self.convergence_data['cl'])
        thickness_values = np.array(self.convergence_data['max_thickness'])
        
        # Check for NaN/Inf values
        has_nan = np.isnan(cd_values).any() or np.isnan(cl_values).any()
        has_inf = np.isinf(cd_values).any() or np.isinf(cl_values).any()
        
        # Check physical ranges
        cd_physical = (cd_values > 0) & (cd_values < 2.0)
        cl_physical = (cl_values > 0) & (cl_values < 3.0)
        thickness_physical = (thickness_values > 0) & (thickness_values < 0.5)
        
        # Check monotonic drag reduction in accepted steps
        accepted_cd = cd_values[accepted_mask]
        monotonic_drag = np.all(np.diff(accepted_cd) <= 1e-6)  # Allow small numerical noise
        
        # Check lift constraint satisfaction
        cl_satisfied = cl_values >= MIN_CL
        
        validation_result = {
            'has_nan_inf': has_nan or has_inf,
            'cd_physical_range': cd_physical.all(),
            'cl_physical_range': cl_physical.all(),
            'thickness_physical_range': thickness_physical.all(),
            'monotonic_drag_reduction': monotonic_drag,
            'lift_constraint_satisfied': cl_satisfied.all(),
            'accepted_steps_ratio': accepted_mask.sum() / len(accepted_mask),
            'total_steps': len(cd_values),
            'accepted_steps': accepted_mask.sum(),
            'status': 'PASS' if all([not (has_nan or has_inf), cd_physical.all(), 
                                    cl_physical.all(), thickness_physical.all()]) 
                       else 'FAIL'
        }
        
        self.validation_results['cfd_integrity'] = validation_result
        
        print(f"NaN/Inf values detected: {validation_result['has_nan_inf']}")
        print(f"Cd physical range: {validation_result['cd_physical_range']}")
        print(f"Cl physical range: {validation_result['cl_physical_range']}")
        print(f"Thickness physical range: {validation_result['thickness_physical_range']}")
        print(f"Monotonic drag reduction: {validation_result['monotonic_drag_reduction']}")
        print(f"Lift constraint satisfied: {validation_result['lift_constraint_satisfied']}")
        print(f"Accepted steps ratio: {validation_result['accepted_steps_ratio']:.2%}")
        print(f"Status: {validation_result['status']}")
        
        return validation_result
    
    def filter_converged_runs(self) -> Dict:
        """Filter convergence history for valid converged runs."""
        print("\n" + "=" * 60)
        print("MULTI-RUN OPTIMIZATION FILTERING")
        print("=" * 60)
        
        if self.convergence_data is None:
            self.load_convergence_history()
        
        # Apply filters
        accepted_mask = np.array(self.convergence_data['step_accepted'])
        thickness_mask = np.array(self.convergence_data['max_thickness']) >= MIN_THICKNESS_TC
        cl_mask = np.array(self.convergence_data['cl']) >= MIN_CL
        
        # Combined validity mask
        valid_mask = accepted_mask & thickness_mask & cl_mask
        
        # Extract valid iterations
        valid_iterations = [it for it, valid in zip(self.convergence_data['iterations'], valid_mask) if valid]
        valid_cd = [cd for cd, valid in zip(self.convergence_data['cd'], valid_mask) if valid]
        valid_cl = [cl for cl, valid in zip(self.convergence_data['cl'], valid_mask) if valid]
        valid_ld = [ld for ld, valid in zip(self.convergence_data['ld'], valid_mask) if valid]
        valid_thickness = [th for th, valid in zip(self.convergence_data['max_thickness'], valid_mask) if valid]
        valid_grad_norm = [gn for gn, valid in zip(self.convergence_data['grad_norm'], valid_mask) if valid]
        
        filtered_data = {
            'valid_iterations': valid_iterations,
            'valid_cd': valid_cd,
            'valid_cl': valid_cl,
            'valid_ld': valid_ld,
            'valid_thickness': valid_thickness,
            'valid_grad_norm': valid_grad_norm,
            'total_iterations': len(self.convergence_data['iterations']),
            'valid_iterations_count': len(valid_iterations),
            'filtered_ratio': len(valid_iterations) / len(self.convergence_data['iterations'])
        }
        
        self.validation_results['filtered_runs'] = filtered_data
        
        print(f"Total iterations: {filtered_data['total_iterations']}")
        print(f"Valid iterations: {filtered_data['valid_iterations_count']}")
        print(f"Filtered ratio: {filtered_data['filtered_ratio']:.2%}")
        
        return filtered_data
    
    def select_best_dataset(self) -> Dict:
        """Select best dataset based on L/D optimization and Cl >= 1.0."""
        print("\n" + "=" * 60)
        print("DATA SELECTION & BEST DATASET EXTRACTION")
        print("=" * 60)
        
        if self.convergence_data is None:
            self.load_convergence_history()
        
        # Calculate drag reduction percentage
        initial_cd = self.convergence_data['cd'][0]
        drag_reductions = [(initial_cd - cd) / initial_cd for cd in self.convergence_data['cd']]
        
        # Apply selection criteria
        valid_mask = (
            np.array(self.convergence_data['step_accepted']) &
            np.array(self.convergence_data['cl']) >= MIN_CL &
            np.array(drag_reductions) >= MIN_LD_GAIN &
            np.array(drag_reductions) <= MAX_LD_GAIN
        )
        
        # Find iteration with maximum L/D
        ld_values = np.array(self.convergence_data['ld'])
        valid_ld = ld_values[valid_mask]
        
        if len(valid_ld) > 0:
            best_valid_idx = np.where(valid_mask)[0][np.argmax(valid_ld)]
        else:
            # Fallback to global maximum if no valid iterations
            best_valid_idx = np.argmax(ld_values)
        
        self.best_iteration = best_valid_idx
        
        best_data = {
            'best_iteration': best_valid_idx + 1,  # 1-indexed
            'best_cd': self.convergence_data['cd'][best_valid_idx],
            'best_cl': self.convergence_data['cl'][best_valid_idx],
            'best_ld': self.convergence_data['ld'][best_valid_idx],
            'best_thickness': self.convergence_data['max_thickness'][best_valid_idx],
            'best_grad_norm': self.convergence_data['grad_norm'][best_valid_idx],
            'best_design_vector': self.convergence_data['design_vectors'][best_valid_idx],
            'drag_reduction_pct': drag_reductions[best_valid_idx],
            'initial_cd': initial_cd,
            'initial_cl': self.convergence_data['cl'][0],
            'initial_ld': self.convergence_data['ld'][0],
            'ld_improvement_pct': (self.convergence_data['ld'][best_valid_idx] - 
                                   self.convergence_data['ld'][0]) / self.convergence_data['ld'][0]
        }
        
        self.validation_results['best_dataset'] = best_data
        
        print(f"Best iteration: {best_data['best_iteration']}")
        print(f"Initial Cd: {best_data['initial_cd']:.4f} -> Best Cd: {best_data['best_cd']:.4f}")
        print(f"Initial Cl: {best_data['initial_cl']:.4f} -> Best Cl: {best_data['best_cl']:.4f}")
        print(f"Initial L/D: {best_data['initial_ld']:.4f} -> Best L/D: {best_data['best_ld']:.4f}")
        print(f"Drag reduction: {best_data['drag_reduction_pct']:.2%}")
        print(f"L/D improvement: {best_data['ld_improvement_pct']:.2%}")
        print(f"Final thickness t/c: {best_data['best_thickness']:.4f}")
        
        return best_data
    
    def export_airfoil_coordinates(self) -> Path:
        """Export airfoil coordinates to CSV."""
        print("\n" + "=" * 60)
        print("EXPORTING AIRFOIL COORDINATES")
        print("=" * 60)
        
        # Load optimized airfoil shape
        shape_file = self.data_dir / "best_airfoil_shape.dat"
        
        # Parse DAT file
        x_coords = []
        y_coords = []
        
        with open(shape_file, 'r') as f:
            lines = f.readlines()
            start_idx = 0 if lines[0].strip() != 'airfoil' else 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))
        
        # Generate baseline airfoil from initial design vector
        if self.convergence_data is None:
            self.load_convergence_history()
        
        initial_dv = self.convergence_data['design_vectors'][0]
        x_baseline, y_baseline = self._cst_to_coordinates(initial_dv)
        
        # Interpolate to same x-coordinates
        y_baseline_interp = np.interp(x_coords, x_baseline, y_baseline)
        
        # Create CSV
        output_file = self.output_dir / "airfoil_coordinates.csv"
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_c', 'y_c_baseline_upper', 'y_c_baseline_lower', 
                           'y_c_optimized_upper', 'y_c_optimized_lower'])
            
            # Split into upper and lower surfaces
            mid_idx = len(x_coords) // 2
            
            for i in range(len(x_coords)):
                if i < mid_idx:
                    # Upper surface (reversed order)
                    row = [x_coords[i], y_baseline_interp[i], '', y_coords[i], '']
                else:
                    # Lower surface
                    row = [x_coords[i], '', y_baseline_interp[i], '', y_coords[i]]
                writer.writerow(row)
        
        print(f"Exported: {output_file}")
        return output_file
    
    def _cst_to_coordinates(self, design_vector: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Convert CST design vector to airfoil coordinates."""
        # Class function (NACA 0012-like)
        def class_function(x):
            return 0.5 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                         0.2843 * x**3 - 0.1015 * x**4)
        
        x = np.linspace(0, 1, n_points)
        
        # Upper surface coefficients (first 6)
        au = design_vector[:6]
        # Lower surface coefficients (last 6)  
        al = design_vector[6:]
        
        # Bernstein polynomial basis
        def bernstein(n, k, x):
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
    
    def export_pressure_distribution(self) -> Path:
        """Export pressure distribution to CSV."""
        print("\n" + "=" * 60)
        print("EXPORTING PRESSURE DISTRIBUTION")
        print("=" * 60)
        
        # Load surface pressure data from CFD cases
        cfd_cases_dir = self.data_dir / "cfd_cases"
        
        # Find first and last eval directories
        eval_dirs = sorted(cfd_cases_dir.glob("eval_*"))
        
        if len(eval_dirs) < 2:
            print("Warning: Insufficient CFD cases for pressure distribution")
            return None
        
        initial_dir = eval_dirs[0]
        final_dir = eval_dirs[-1]
        
        # Parse surface flow files
        def parse_surface_flow(directory):
            surface_file = directory / "surface_flow.csv"
            if not surface_file.exists():
                return None, None, None, None
            
            x_upper, cp_upper = [], []
            x_lower, cp_lower = [], []
            
            with open(surface_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        x = float(row.get('x', 0))
                        cp = float(row.get('Cp', 0))
                        
                        # Simple split: first half upper, second half lower
                        if len(x_upper) <= len(x_lower) or len(x_lower) == 0:
                            x_upper.append(x)
                            cp_upper.append(cp)
                        else:
                            x_lower.append(x)
                            cp_lower.append(cp)
                    except (ValueError, KeyError):
                        continue
            
            return (np.array(x_upper), np.array(cp_upper), 
                   np.array(x_lower), np.array(cp_lower))
        
        x_u_init, cp_u_init, x_l_init, cp_l_init = parse_surface_flow(initial_dir)
        x_u_opt, cp_u_opt, cp_l_opt, x_l_opt = parse_surface_flow(final_dir)
        
        if x_u_init is None or x_u_opt is None:
            print("Warning: Could not parse surface pressure data")
            return None
        
        # Create CSV
        output_file = self.output_dir / "pressure_distribution.csv"
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_c', 'Cp_baseline_upper', 'Cp_baseline_lower', 
                           'Cp_optimized_upper', 'Cp_optimized_lower'])
            
            # Interpolate to common x-coordinates
            x_common = np.linspace(0, 1, 100)
            
            cp_u_init_interp = np.interp(x_common, x_u_init, cp_u_init)
            cp_l_init_interp = np.interp(x_common, x_l_init, cp_l_init)
            cp_u_opt_interp = np.interp(x_common, x_u_opt, cp_u_opt)
            cp_l_opt_interp = np.interp(x_common, x_l_opt, cp_l_opt)
            
            for i in range(len(x_common)):
                writer.writerow([x_common[i], cp_u_init_interp[i], cp_l_init_interp[i],
                               cp_u_opt_interp[i], cp_l_opt_interp[i]])
        
        print(f"Exported: {output_file}")
        return output_file
    
    def export_convergence_history(self) -> Path:
        """Export convergence history to CSV."""
        print("\n" + "=" * 60)
        print("EXPORTING CONVERGENCE HISTORY")
        print("=" * 60)
        
        if self.convergence_data is None:
            self.load_convergence_history()
        
        output_file = self.output_dir / "convergence_history.csv"
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Cd', 'Cl', 'L_D', 'Max_Thickness', 
                           'Gradient_Norm', 'Step_Accepted', 'Trust_Radius'])
            
            for i in range(len(self.convergence_data['iterations'])):
                writer.writerow([
                    self.convergence_data['iterations'][i],
                    self.convergence_data['cd'][i],
                    self.convergence_data['cl'][i],
                    self.convergence_data['ld'][i],
                    self.convergence_data['max_thickness'][i],
                    self.convergence_data['grad_norm'][i],
                    self.convergence_data['step_accepted'][i],
                    self.convergence_data['trust_radius'][i]
                ])
        
        print(f"Exported: {output_file}")
        return output_file
    
    def generate_figure1_convergence_history(self) -> Path:
        """Generate Figure 1: Convergence history."""
        print("\n" + "=" * 60)
        print("GENERATING FIGURE 1: CONVERGENCE HISTORY")
        print("=" * 60)
        
        if self.convergence_data is None:
            self.load_convergence_history()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        
        iterations = self.convergence_data['iterations']
        cd = self.convergence_data['cd']
        cl = self.convergence_data['cl']
        step_accepted = self.convergence_data['step_accepted']
        
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
        ax2.axhline(y=MIN_CL, color=COLORS['black'], linestyle='--', 
                   linewidth=1.0, alpha=0.7, label='$C_L = 1.0$ floor')
        ax2.scatter(iterations, cl, c=[COLORS['red'] if sa else COLORS['blue'] 
                                     for sa in step_accepted], s=30, zorder=5)
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('$C_L$', fontsize=11)
        ax2.set_title('(b) Lift Coefficient', fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='lower right', frameon=True)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "figure1_convergence_history.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Generated: {output_file}")
        plt.close(fig)
        
        return output_file
    
    def generate_figure2_airfoil_profile(self) -> Path:
        """Generate Figure 2: Airfoil profile overlay."""
        print("\n" + "=" * 60)
        print("GENERATING FIGURE 2: AIRFOIL PROFILE OVERLAY")
        print("=" * 60)
        
        # Load optimized airfoil
        shape_file = self.data_dir / "best_airfoil_shape.dat"
        
        x_coords = []
        y_coords = []
        
        with open(shape_file, 'r') as f:
            lines = f.readlines()
            start_idx = 0 if lines[0].strip() != 'airfoil' else 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))
        
        # Generate initial airfoil
        if self.convergence_data is None:
            self.load_convergence_history()
        
        initial_dv = self.convergence_data['design_vectors'][0]
        x_init, y_init = self._cst_to_coordinates(initial_dv)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        
        # Plot initial airfoil
        ax.plot(x_init, y_init, color=COLORS['gray'], linewidth=1.5, 
               linestyle='--', label='Baseline', alpha=0.7)
        
        # Plot optimized airfoil
        ax.plot(x_coords, y_coords, color=COLORS['blue'], linewidth=2.0, 
               label='Optimized')
        
        # Add thickness annotation
        max_thickness = max(y_coords) - min(y_coords)
        ax.text(0.05, 0.15, f'Max t/c = {max_thickness:.4f}', 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
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
        
        output_file = self.output_dir / "figure2_airfoil_profile.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Generated: {output_file}")
        plt.close(fig)
        
        return output_file
    
    def generate_figure3_surface_pressure(self) -> Path:
        """Generate Figure 3: Surface pressure distribution."""
        print("\n" + "=" * 60)
        print("GENERATING FIGURE 3: SURFACE PRESSURE DISTRIBUTION")
        print("=" * 60)
        
        # Load surface pressure data
        cfd_cases_dir = self.data_dir / "cfd_cases"
        eval_dirs = sorted(cfd_cases_dir.glob("eval_*"))
        
        if len(eval_dirs) < 2:
            print("Warning: Insufficient CFD cases for pressure distribution")
            return None
        
        def parse_surface_flow(directory):
            surface_file = directory / "surface_flow.csv"
            if not surface_file.exists():
                return None, None, None, None
            
            x_upper, cp_upper = [], []
            x_lower, cp_lower = [], []
            
            with open(surface_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        x = float(row.get('x', 0))
                        cp = float(row.get('Cp', 0))
                        
                        if len(x_upper) <= len(x_lower) or len(x_lower) == 0:
                            x_upper.append(x)
                            cp_upper.append(cp)
                        else:
                            x_lower.append(x)
                            cp_lower.append(cp)
                    except (ValueError, KeyError):
                        continue
            
            return (np.array(x_upper), np.array(cp_upper), 
                   np.array(x_lower), np.array(cp_lower))
        
        x_u_init, cp_u_init, x_l_init, cp_l_init = parse_surface_flow(eval_dirs[0])
        x_u_opt, cp_u_opt, cp_l_opt, x_l_opt = parse_surface_flow(eval_dirs[-1])
        
        if x_u_init is None or x_u_opt is None:
            print("Warning: Could not parse surface pressure data")
            return None
        
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
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        output_file = self.output_dir / "figure3_surface_pressure.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Generated: {output_file}")
        plt.close(fig)
        
        return output_file
    
    def generate_figure4_gradient_decay(self) -> Path:
        """Generate Figure 4: Gradient decay."""
        print("\n" + "=" * 60)
        print("GENERATING FIGURE 4: GRADIENT DECAY")
        print("=" * 60)
        
        if self.convergence_data is None:
            self.load_convergence_history()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        
        iterations = self.convergence_data['iterations']
        grad_norm = self.convergence_data['grad_norm']
        trust_radius = self.convergence_data['trust_radius']
        step_size = self.convergence_data['step_size']
        
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
        
        output_file = self.output_dir / "figure4_gradient_decay.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Generated: {output_file}")
        plt.close(fig)
        
        return output_file
    
    def run_full_pipeline(self):
        """Execute the complete data sanitization pipeline."""
        print("=" * 70)
        print("LOW-REYNOLDS-NUMBER AIRFOIL OPTIMIZATION DATA SANITIZATION PIPELINE")
        print("=" * 70)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Step 1: Validate geometry constraints
        self.validate_geometry_constraints()
        
        # Step 2: Validate CFD data integrity
        self.validate_cfd_data_integrity()
        
        # Step 3: Filter converged runs
        self.filter_converged_runs()
        
        # Step 4: Select best dataset
        self.select_best_dataset()
        
        # Step 5: Export CSV files
        self.export_airfoil_coordinates()
        self.export_pressure_distribution()
        self.export_convergence_history()
        
        # Step 6: Generate figures
        self.generate_figure1_convergence_history()
        self.generate_figure2_airfoil_profile()
        self.generate_figure3_surface_pressure()
        self.generate_figure4_gradient_decay()
        
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 70)
        print(f"All outputs saved to: {self.output_dir}")
        print()
        print("VALIDATION SUMMARY:")
        print(f"Geometry constraints: {self.validation_results['geometry']['status']}")
        print(f"CFD data integrity: {self.validation_results['cfd_integrity']['status']}")
        print(f"Valid iterations: {self.validation_results['filtered_runs']['valid_iterations_count']}/{self.validation_results['filtered_runs']['total_iterations']}")
        print(f"Best iteration: {self.validation_results['best_dataset']['best_iteration']}")
        print(f"Best L/D: {self.validation_results['best_dataset']['best_ld']:.4f}")
        print("=" * 70)


def main():
    """Main execution function."""
    # Set paths
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "aso_production_100iter"
    output_dir = project_root / "sanitized_outputs"
    
    # Initialize and run pipeline
    pipeline = DataSanitizationPipeline(data_dir, output_dir)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()