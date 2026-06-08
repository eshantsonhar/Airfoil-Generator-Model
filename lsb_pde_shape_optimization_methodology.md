# Passive Suppression of Laminar Separation Bubbles in Low-Reynolds-Number Airfoils Using PDE-Constrained Aerodynamic Shape Optimization

## 1. Research Foundations

### 1.1 Introduction

Low-Reynolds-number airfoils operate in a regime where the boundary layer is thin, weakly energetic, and often laminar over a substantial fraction of the chord. For chord Reynolds numbers in the range

$$
5.0\times 10^4 \le Re_c=\frac{\rho_\infty U_\infty c}{\mu_\infty} \le 5.0\times 10^5,
$$

the laminar boundary layer is highly susceptible to adverse pressure gradients. When the local wall shear stress becomes zero and then negative, the laminar boundary layer separates. If the separated shear layer undergoes transition, turbulent mixing can re-energize the near-wall flow and cause reattachment, producing a laminar separation bubble (LSB). The bubble changes the effective aerodynamic contour seen by the inviscid outer flow, increases pressure drag, modifies suction-side pressure recovery, and can trigger abrupt stall when the bubble bursts or fails to reattach.

The LSB is not a minor viscous correction. It is a coupled viscous-inviscid-transition phenomenon governed by the pressure-gradient history, boundary-layer receptivity, free-stream turbulence intensity, surface roughness, acoustic disturbance environment, and airfoil geometry. A small change in leading-edge radius, suction-side curvature, or pressure-recovery slope can shift the transition point enough to alter bubble length, drag, maximum lift, and stall hysteresis. This makes low-Reynolds-number airfoil design difficult for micro air vehicles, small wind turbines, high-altitude propellers, unmanned aircraft, low-speed rotor sections, and bio-inspired lifting surfaces.

Conventional mitigation often uses forced transition devices such as trips, roughness strips, zigzag tape, or distributed grit. These devices can suppress long bubbles by forcing earlier transition, but they add parasitic drag outside the design condition, are sensitive to height and placement, may fail under contamination or manufacturing variation, and can be aerodynamically unstable across Reynolds number and angle of attack. A trip optimized at one operating point can become an unnecessary drag source at another or can force premature turbulent separation near stall.

Passive geometric control is valuable because it addresses the pressure-gradient mechanism that causes the bubble rather than adding an external disturbance to overwhelm the transition process. A geometry optimized to avoid an excessively steep adverse pressure gradient can delay laminar separation, promote benign short-bubble behavior, or distribute pressure recovery so that transition and reattachment occur without a large drag penalty. Shape-based control can be integrated into the airfoil itself, requires no actuator power, does not rely on fragile trip-height tuning, and can be optimized over multiple Reynolds numbers and angles of attack.

This research therefore frames LSB suppression as a PDE-constrained aerodynamic shape optimization problem. The flow equations, transition model, mesh deformation, and aerodynamic objectives are solved in a coupled computational framework. The design variables modify the airfoil geometry through a smooth Class-Shape Transformation (CST) parameterization, while constraints preserve lift, thickness, curvature, leading-edge feasibility, trailing-edge closure, and manufacturability.

### 1.2 Research Motivation

Low-Reynolds-number airfoils exhibit aerodynamic behavior that differs sharply from high-Reynolds-number attached-flow airfoils. The key motivation is that low-Re performance is often dominated by transition and separation rather than by inviscid loading alone. Specific motivations are:

1. LSB-induced drag rise: A long separation bubble produces elevated pressure drag and can degrade lift-to-drag ratio even when lift remains acceptable.
2. Low-Re stall behavior: Stall can occur through bubble growth, bubble bursting, or leading-edge separation rather than the gradual trailing-edge separation common in many higher-Re cases.
3. Hysteresis: During increasing and decreasing angle-of-attack sweeps, the separated shear layer and transition location may follow different states, producing different lift and drag at the same angle of attack.
4. Surface contamination sensitivity: Dust, insect residue, rain droplets, leading-edge erosion, and manufacturing waviness can move transition onset, which can either shorten an LSB or create premature turbulent drag.
5. Trip instability: Forced-transition devices depend on local boundary-layer thickness, Reynolds number, roughness Reynolds number, pressure gradient, and disturbance environment. Their effect is not invariant across the flight envelope.
6. Need for reproducible design: Airfoil shapes intended for low-Re applications must be evaluated using transition-aware CFD, mesh convergence, gradient verification, and uncertainty-aware post-processing rather than only inviscid or fully turbulent assumptions.

### 1.3 Problem Statement

The research problem is to determine whether passive geometric modification of a low-Reynolds-number airfoil can suppress or reduce deleterious laminar separation bubbles while preserving required lift and geometric feasibility. The design must be obtained using a reproducible PDE-constrained optimization framework in which the governing flow equations, transition closure, aerodynamic objectives, geometry constraints, adjoint sensitivities, and data-analysis pipeline are explicitly defined.

The central technical problem is:

$$
\begin{aligned}
\min_{\mathbf{a},\,\mathbf{w},\,\mathbf{x}} \quad & J(\mathbf{w},\mathbf{x},\mathbf{a}) \\
\text{subject to} \quad & \mathbf{R}(\mathbf{w},\mathbf{x},\mathbf{a};Re_c,M_\infty,\alpha)=\mathbf{0},\\
& \mathbf{G}(\mathbf{x},\mathbf{a})=\mathbf{0},\\
& C_L(\mathbf{w},\mathbf{x}) \ge C_{L,\min},\\
& t(x_i;\mathbf{a}) \ge t_{\min}(x_i), \quad i=1,\ldots,N_t,\\
& \kappa_{\min} \le \kappa(s;\mathbf{a}) \le \kappa_{\max},\\
& \mathbf{a}_{\min}\le \mathbf{a}\le \mathbf{a}_{\max},
\end{aligned}
$$

where $\mathbf{a}$ are CST design variables, $\mathbf{w}$ is the conservative flow-state vector, $\mathbf{x}$ is the computational mesh, $\mathbf{R}$ is the discretized RANS-transition residual, $\mathbf{G}$ is the mesh-deformation equation, $J$ is a drag, bubble, or robust multi-point objective, and the constraints enforce aerodynamic and geometric acceptability.

### 1.4 Research Gap

Previous low-Re airfoil studies have established the importance of laminar separation bubbles, but four gaps remain for publishable optimization work:

1. Many airfoil optimizations rely on XFOIL or panel-integral boundary-layer methods. These are computationally efficient and useful for preliminary screening, but they do not resolve the full separated shear-layer dynamics, three-dimensional breakdown mechanisms, or detailed turbulence-transition coupling.
2. Fully turbulent RANS optimization often suppresses the laminar physics by assuming turbulent boundary layers from the leading edge. This tends to overpredict attachment under adverse pressure gradients and can produce shapes that perform well numerically but fail in low-Re transitional flow.
3. Transition-aware RANS is frequently used for analysis rather than embedded consistently in a shape-optimization loop with verified adjoint or finite-difference gradients.
4. Published optimization workflows often report lift and drag improvements without a rigorous LSB diagnostic framework. Bubble length, transition onset, reattachment location, pressure recovery, wake thickness, hysteresis, and mesh sensitivity must be quantified directly rather than inferred from force coefficients alone.

This work fills the gap by combining transition-aware CFD, smooth CST geometry control, constrained gradient-based optimization, explicit LSB diagnostics, multi-fidelity screening, and an R-based reproducible statistical analysis pipeline.

### 1.5 Hypothesis

The primary hypothesis is:

> A CST-parameterized airfoil optimized using PDE-constrained, transition-aware aerodynamic shape optimization can reduce LSB length and drag at low Reynolds number without sacrificing lift, by redistributing suction-side pressure recovery and reducing the severity of the adverse pressure gradient upstream of transition.

Secondary hypotheses are:

1. A geometry-only passive design can produce more robust low-Re performance than a trip-dependent design when evaluated across multiple Reynolds numbers, angles of attack, and transition sensitivities.
2. A multi-point objective penalizing drag and LSB length over an operating envelope will produce shapes with weaker stall hysteresis than a single-point drag optimum.
3. Gradient consistency checks are necessary because transition-model nonlinearities and separation-induced intermittency can introduce nonsmooth objective behavior near bubble onset, bursting, or reattachment loss.

### 1.6 Research Objectives

1. Develop a reproducible CST-based airfoil parameterization suitable for low-Re shape optimization.
2. Generate high-quality low-Re airfoil meshes with leading-edge clustering, suction-side boundary-layer resolution, trailing-edge refinement, and verified $y^+$ control.
3. Solve compressible or weakly compressible RANS equations with transition modeling at specified low-Reynolds-number conditions.
4. Formulate single-point, multi-point, and robustness-based PDE-constrained optimization problems.
5. Implement MMA-based constrained design updates with move limits, feasibility restoration, and trust-region stabilization.
6. Verify adjoint sensitivities against finite differences using mesh-consistent perturbations.
7. Quantify LSB metrics from CFD fields: separation, transition, reattachment, bubble length, shape factor, skin friction, pressure recovery, wake thickness, stall onset, and hysteresis.
8. Establish a statistical R pipeline for ingestion, cleaning, diagnostics, visualization, uncertainty quantification, mixed-effects modeling, Bayesian comparison, and reproducible reporting.
9. Perform verification and validation using code verification, solution verification, grid convergence, Richardson extrapolation, and comparison with experimental or benchmark data.

### 1.7 Contributions

The expected contributions are:

1. A transition-aware PDE-constrained optimization framework for passive low-Re LSB suppression.
2. A CST-based geometry representation with explicit thickness, curvature, trailing-edge, and manufacturability constraints.
3. A rigorous CFD methodology for low-Re separated transitional flow, including mesh, solver, convergence, and validation requirements.
4. A complete LSB detection methodology based on surface and field quantities rather than only aerodynamic coefficients.
5. A reproducible R data-analysis framework for CFD campaign management, optimization history analysis, uncertainty quantification, and publication-quality visualization.
6. A V&V protocol suitable for defending the computational results in an aerospace thesis or journal paper.

### 1.8 Scope and Limitations

The scope is two-dimensional airfoil-section optimization for low-Mach-number external flow. The baseline Reynolds-number envelope is

$$
Re_c\in \{1.0\times10^5,\;2.0\times10^5,\;3.0\times10^5,\;5.0\times10^5\},
$$

with angle of attack

$$
\alpha \in [-4^\circ,14^\circ],
$$

sampled at $1^\circ$ increments for polar construction and at $0.25^\circ$ to $0.5^\circ$ increments near stall and hysteresis transitions. The baseline Mach number is

$$
M_\infty = 0.10
$$

unless incompressible settings are explicitly used. At $M_\infty=0.10$, compressibility effects are small but compressible RANS remains numerically convenient in SU2 and provides a consistent conservative formulation.

Limitations are:

1. Two-dimensional RANS cannot resolve spanwise LSB breakdown, secondary instability, or three-dimensional vortex dynamics.
2. Transition correlations require calibration against free-stream turbulence intensity, surface roughness, and facility disturbance environment.
3. URANS can capture large-scale unsteadiness but does not replace DNS, LES, or well-resolved experimental measurements.
4. Shape optimization may exploit model bias; therefore validation against independent data is mandatory.
5. Stall and hysteresis near bubble bursting may be multistable and path-dependent, so a single steady solution is insufficient for final claims.

## 2. Literature Review

This review is organized by research theme. It identifies achievements, remaining limitations, contradictions, failure modes, and the specific gap addressed by this work. Literature anchors include Gaster's classical separation-bubble work, Tani's separation studies, Horton and Lissaman's low-Re airfoil interpretations, Drela's XFOIL framework, Menter's SST model, Langtry and Menter's $\gamma-Re_{\theta}$ transition model, Kulfan's CST parameterization, Svanberg's MMA, Roache's verification methodology, and the SU2 discrete-adjoint framework.

### 2.1 Low-Reynolds-Number Aerodynamics

Previous researchers established that low-Re airfoils cannot be interpreted by inviscid loading and fully turbulent boundary-layer assumptions alone. Low-Re lift, drag, and stall depend on whether the laminar boundary layer remains attached, separates and reattaches as a bubble, or separates without reattachment. Experimental low-Re airfoil programs have shown large scatter among nominally similar tests because turbulence intensity, acoustic environment, model finish, tunnel blockage, and end effects alter transition.

Limitations remain because many low-Re databases provide force polars without synchronized surface-pressure, skin-friction, transition, and velocity-field measurements. Contradictions occur when one study reports an airfoil as high-performing due to a stable short bubble while another observes a long bubble or leading-edge stall at the same nominal Reynolds number. These contradictions are often traceable to disturbance environment or surface condition.

Current approaches fail when they treat $Re_c$ and $\alpha$ as sufficient descriptors. For LSB-dominated airfoils, the disturbance environment and pressure-gradient history are also state variables. This work fills the gap by requiring transition-aware modeling, explicit bubble diagnostics, and uncertainty-aware comparison rather than force-coefficient comparison alone.

### 2.2 Laminar Separation Bubble Physics

Previous research has described the LSB sequence: laminar boundary-layer growth, adverse-pressure-gradient separation, separated shear-layer amplification, transition, turbulent reattachment, and wake recovery. The separated shear layer is receptive to Kelvin-Helmholtz-type instabilities. Bubble bursting occurs when transition and turbulent mixing are insufficient to produce reattachment before the separation region grows catastrophically.

Limitations remain because LSBs can be short, long, closed, open, steady, quasi-periodic, or strongly unsteady depending on Reynolds number and pressure gradient. Contradictions exist over whether the bubble improves or degrades performance in a given case. A short bubble can act as an efficient transition mechanism with acceptable drag, while a long bubble causes pressure drag and stall vulnerability.

Conventional analysis fails when it identifies only the transition point but not separation and reattachment. Transition location alone does not define bubble severity. This work fills the gap by measuring separation, transition, reattachment, bubble length, pressure recovery, and wake response together.

### 2.3 Transition Modeling

Previous researchers developed empirical, semi-empirical, and transport-equation transition models to represent onset in RANS. The $\gamma-Re_{\theta}$ model introduced by Langtry and Menter is especially influential because it uses local transport equations compatible with unstructured CFD solvers. It links transition onset to local correlations for transition momentum-thickness Reynolds number and intermittency.

Limitations remain because transition models are calibrated over finite datasets and are sensitive to free-stream turbulence intensity, pressure gradient, roughness, separation, and model constants. Contradictions occur when different transition models predict similar forces but different transition locations, or similar transition locations but different reattachment behavior.

Fully turbulent RANS fails at low Re because it assumes turbulent eddy viscosity before physical transition. Laminar-only simulations fail after separation because they cannot model turbulent reattachment. This work uses transition modeling because LSB formation is governed by the relative ordering of separation onset, transition onset, and reattachment.

### 2.4 Passive Flow Control

Passive flow control includes geometric shaping, leading-edge tubercles, vortex generators, roughness elements, trips, slots, cavities, compliant surfaces, and pressure-gradient tailoring. Previous work shows that passive devices can delay stall, force transition, or energize the boundary layer without external power.

Limitations remain because many passive devices work by adding disturbances rather than removing the pressure-gradient cause. Trips and roughness strips increase drag when the flow would otherwise remain favorably laminar. Small vortex generators may be sensitive to installation tolerances and can be inappropriate for very small airfoils.

Contradictions arise because a forced-transition device can improve maximum lift but degrade cruise efficiency. This work fills the gap by treating the airfoil shape itself as the control mechanism and optimizing geometry over multiple operating points.

### 2.5 Active Flow Control

Active flow control includes suction, blowing, plasma actuators, synthetic jets, acoustic forcing, moving surfaces, and feedback-controlled actuation. Previous research shows that active forcing can alter shear-layer instability and close open separation.

Limitations are power consumption, actuator integration, reliability, bandwidth, mass, and robustness. At low Reynolds number, effective forcing frequencies and amplitudes can depend strongly on the separated shear-layer state. Contradictions occur when forcing improves one flow state but triggers another, especially in bistable or hysteretic regimes.

This work does not replace active control research. It fills a complementary gap: passive geometric suppression for systems where actuator complexity is undesirable.

### 2.6 Adjoint Optimization

Adjoint methods allow gradients of an objective with respect to many design variables at a cost largely independent of the number of variables. Previous aerodynamic shape optimization studies use continuous or discrete adjoints for drag reduction, lift constraints, inverse design, and multi-point objectives.

Limitations remain in separated transitional flows because the objective may be nonsmooth near separation onset, transition onset, reattachment, or stall. Adjoint gradients may be inconsistent if the transition model, limiters, mesh deformation, or turbulence closure are not differentiated consistently.

This work fills the gap by requiring gradient consistency verification and using trust-region stabilization when transition-induced nonsmoothness is detected.

### 2.7 PDE-Constrained Optimization

PDE-constrained optimization embeds the governing equations as constraints rather than treating CFD as an external black box. Previous work has demonstrated efficient shape optimization using RANS and adjoint solvers.

Limitations remain when the PDE residual is not converged tightly enough, when the mesh deformation equations are ignored in sensitivities, or when constraints are enforced through weak penalties that hide infeasibility. Current approaches fail when optimization iterations accept noisy force coefficients from underconverged separated flows.

This work fills the gap by defining direct residual targets, force convergence criteria, adjoint residual targets, finite-difference checks, and mesh-consistent design updates.

### 2.8 Airfoil Parameterization Methods

Airfoil parameterizations include Hicks-Henne bumps, B-splines, Bezier curves, PARSEC, free-form deformation, CST, and direct coordinate control. Previous work shows that smooth low-dimensional parameterizations reduce noisy shape changes and help maintain manufacturability.

Limitations remain because overly restrictive parameterizations cannot express necessary pressure-gradient redistribution, while overly flexible methods produce curvature oscillations and geometry ringing. Contradictions occur when a parameterization that performs well for transonic shock control is too global or too smooth for low-Re leading-edge and suction-side transition control.

This work uses CST because it represents airfoil-like geometry analytically with explicit class and shape functions, while allowing upper and lower surface coefficients to be constrained.

### 2.9 CST Parameterization

Kulfan's Class-Shape Transformation represents airfoil coordinates as the product of a class function and a Bernstein-polynomial shape function, optionally with a trailing-edge thickness term. Previous researchers have used CST for airfoil fitting, inverse design, and optimization because it provides smooth surfaces and compact design variables.

Limitations are that low-order CST can underfit local leading-edge features, while high-order CST can create oscillatory curvature if unconstrained. Contradictions arise when CST is described as universally smooth but used with insufficient curvature control.

This work fills the gap by pairing CST with curvature constraints, leading-edge radius checks, thickness constraints, and finite-difference geometry audits.

### 2.10 MMA Optimization

Svanberg's Method of Moving Asymptotes constructs convex separable approximations of nonlinear objectives and constraints. Previous researchers have used MMA widely in structural and aerodynamic optimization because it handles many variables and inequality constraints robustly.

Limitations include sensitivity to gradient noise, asymptote movement, scaling, and active-constraint inconsistency. In transitional CFD, a design step can move the flow across a nonsmooth transition state, invalidating local approximations.

This work uses MMA with nondimensional design variables, move limits, constraint scaling, trust-region rejection, and KKT monitoring.

### 2.11 Multi-Fidelity CFD Optimization

Multi-fidelity approaches combine inexpensive screening tools such as XFOIL, panel methods, or coarse RANS with higher-fidelity transition-aware CFD. Previous work demonstrates computational savings when low-fidelity models identify promising design regions.

Limitations are model-form bias and false optima. XFOIL, while foundational and useful, couples a panel method with an integral boundary-layer model and an $e^N$-type transition criterion; it can have poor resolution of small viscous features such as separation bubbles and cannot represent three-dimensional LSB breakdown. Fully turbulent coarse RANS can also rank designs incorrectly.

This work uses low-fidelity tools only for initialization, screening, and sanity checks. Final claims are based on transition-aware RANS/URANS, mesh-converged diagnostics, and validation comparison.

### 2.12 Existing Low-Re Airfoils

Existing low-Re airfoil families include Selig/University of Illinois low-Re airfoils, Eppler airfoils, Drela airfoils, Wortmann sections, and NACA sections used as baselines. Previous research achieved high lift-to-drag ratios at selected Reynolds numbers through careful pressure recovery and transition management.

Limitations remain because airfoils optimized for one Reynolds number, turbulence environment, or manufacturing quality may degrade sharply elsewhere. Contradictions occur when thin high-camber designs show excellent low-drag performance in one dataset but poor stall robustness in another.

This work does not claim one universal airfoil. It seeks a reproducible method for pressure-gradient-based LSB suppression and robust operating-envelope performance.

### 2.13 Experimental Validation Approaches

Experimental validation methods include force balances, surface pressure taps, oil-flow visualization, infrared thermography, hot-film sensors, particle image velocimetry, smoke visualization, and wake surveys. Previous researchers have used these methods to identify separation, transition, reattachment, and bubble bursting.

Limitations include tunnel-wall interference, blockage, end-plate effects, model roughness, pressure-tap spatial resolution, PIV near-wall uncertainty, and mismatch between tunnel turbulence intensity and CFD transition inputs. Hysteresis validation requires controlled increasing and decreasing angle-of-attack sweeps rather than isolated static points.

This work fills the gap by defining validation variables in advance: $C_L$, $C_D$, $C_m$, $C_p(x)$, $x_s/c$, $x_t/c$, $x_r/c$, bubble length, wake momentum thickness, and hysteresis loop area.

## 3. Governing Physics

### 3.1 Compressible RANS Equations

The compressible Reynolds-averaged Navier-Stokes equations are solved in conservative form:

$$
\frac{\partial \mathbf{Q}}{\partial t}
+ \nabla\cdot \mathbf{F}^{c}(\mathbf{Q})
- \nabla\cdot \mathbf{F}^{v}(\mathbf{Q},\nabla\mathbf{Q})
= \mathbf{S},
$$

where

$$
\mathbf{Q}=
\begin{bmatrix}
\rho\\
\rho \tilde{u}_1\\
\rho \tilde{u}_2\\
\rho \tilde{u}_3\\
\rho \tilde{E}
\end{bmatrix}.
$$

The Favre-averaged velocity is $\tilde{u}_i=\overline{\rho u_i}/\bar{\rho}$. The total energy per unit mass is

$$
\tilde{E}=\tilde{e}+\frac{1}{2}\tilde{u}_i\tilde{u}_i+k,
$$

where $k$ is turbulent kinetic energy when a two-equation turbulence model is used. For one-equation Spalart-Allmaras closure, $k$ is not a solved variable and the practical energy closure uses effective viscosity and turbulent heat flux.

### 3.2 Continuity Equation

$$
\frac{\partial \rho}{\partial t}+\frac{\partial(\rho \tilde{u}_j)}{\partial x_j}=0.
$$

For steady simulations, the temporal term is pseudo-time marched to convergence:

$$
\frac{\partial \rho}{\partial \tau}+\frac{\partial(\rho \tilde{u}_j)}{\partial x_j}=0.
$$

### 3.3 Momentum Equation

$$
\frac{\partial(\rho \tilde{u}_i)}{\partial t}
+\frac{\partial}{\partial x_j}
\left(\rho \tilde{u}_i\tilde{u}_j+p\delta_{ij}\right)
=
\frac{\partial}{\partial x_j}
\left(\tau_{ij}+\tau_{ij}^{R}\right),
$$

where $\tau_{ij}$ is the molecular viscous stress tensor and $\tau_{ij}^{R}=-\rho \widetilde{u_i''u_j''}$ is the Reynolds stress tensor.

### 3.4 Energy Equation

$$
\frac{\partial(\rho \tilde{E})}{\partial t}
+\frac{\partial}{\partial x_j}
\left[(\rho \tilde{E}+p)\tilde{u}_j\right]
=
\frac{\partial}{\partial x_j}
\left[
\tilde{u}_i(\tau_{ij}+\tau_{ij}^{R})
- q_j - q_j^t
\right].
$$

The equation of state for a calorically perfect gas is

$$
p=\rho R T,\qquad e=c_vT,\qquad h=c_pT.
$$

### 3.5 Stress Tensor

The molecular stress tensor is

$$
\tau_{ij}
=
\mu
\left(
\frac{\partial \tilde{u}_i}{\partial x_j}
+\frac{\partial \tilde{u}_j}{\partial x_i}
-\frac{2}{3}\frac{\partial \tilde{u}_k}{\partial x_k}\delta_{ij}
\right).
$$

The Boussinesq eddy-viscosity approximation models Reynolds stress as

$$
\tau_{ij}^{R}
=
2\mu_t S_{ij}
-\frac{2}{3}\rho k\delta_{ij},
$$

where

$$
S_{ij}=
\frac{1}{2}
\left(
\frac{\partial \tilde{u}_i}{\partial x_j}
+\frac{\partial \tilde{u}_j}{\partial x_i}
\right).
$$

### 3.6 Heat Flux

The molecular heat flux is

$$
q_j=-k_f\frac{\partial T}{\partial x_j}
=-\frac{\mu c_p}{Pr}\frac{\partial T}{\partial x_j}.
$$

The turbulent heat flux is modeled using a turbulent Prandtl number:

$$
q_j^t=-\frac{\mu_t c_p}{Pr_t}\frac{\partial T}{\partial x_j},
$$

with $Pr_t=0.90$ for baseline simulations unless validation against a specific solver benchmark requires another value.

### 3.7 Closure Problem

RANS averaging introduces unknown Reynolds stresses and turbulent heat fluxes. The closure problem is the need to express these quantities in terms of mean-flow variables. For low-Re LSB prediction, closure is not only a turbulence problem but also a transition problem. The closure must represent the laminar region, separated laminar shear layer, transition onset, turbulent reattachment, and downstream turbulent boundary layer.

### 3.8 Spalart-Allmaras Model

The Spalart-Allmaras (SA) model solves a transport equation for a modified turbulent kinematic viscosity $\tilde{\nu}$:

$$
\frac{\partial \tilde{\nu}}{\partial t}
+\tilde{u}_j\frac{\partial \tilde{\nu}}{\partial x_j}
=
c_{b1}(1-f_{t2})\tilde{S}\tilde{\nu}
+\frac{1}{\sigma}
\left[
\nabla\cdot((\nu+\tilde{\nu})\nabla\tilde{\nu})
+c_{b2}|\nabla\tilde{\nu}|^2
\right]
-\left[
c_{w1}f_w-\frac{c_{b1}}{\kappa^2}f_{t2}
\right]
\left(\frac{\tilde{\nu}}{d}\right)^2.
$$

The eddy viscosity is

$$
\nu_t=\tilde{\nu}f_{v1},\qquad
f_{v1}=\frac{\chi^3}{\chi^3+c_{v1}^3},\qquad
\chi=\frac{\tilde{\nu}}{\nu}.
$$

SA is robust and economical, but in its common fully turbulent form it is unsuitable for natural LSB prediction because it produces turbulent eddy viscosity upstream of the physical transition location. It can be useful for fully turbulent comparison, grid studies, or sensitivity bracketing, but it cannot be the primary model for transition-governed LSB optimization.

### 3.9 k-omega SST Model

The SST model blends $k-\omega$ behavior near the wall with $k-\epsilon$ behavior away from the wall. The transport equations are

$$
\frac{\partial(\rho k)}{\partial t}
+\frac{\partial(\rho \tilde{u}_j k)}{\partial x_j}
=
P_k-\beta^*\rho k\omega
+\frac{\partial}{\partial x_j}
\left[
\left(\mu+\sigma_k\mu_t\right)
\frac{\partial k}{\partial x_j}
\right],
$$

$$
\frac{\partial(\rho \omega)}{\partial t}
+\frac{\partial(\rho \tilde{u}_j \omega)}{\partial x_j}
=
\alpha\frac{\omega}{k}P_k
-\beta\rho\omega^2
+\frac{\partial}{\partial x_j}
\left[
\left(\mu+\sigma_\omega\mu_t\right)
\frac{\partial \omega}{\partial x_j}
\right]
+2(1-F_1)\rho\sigma_{\omega2}
\frac{1}{\omega}
\frac{\partial k}{\partial x_j}
\frac{\partial \omega}{\partial x_j}.
$$

The eddy viscosity is limited by

$$
\mu_t=\frac{\rho a_1 k}{\max(a_1\omega,SF_2)}.
$$

SST improves adverse-pressure-gradient separation prediction relative to many earlier eddy-viscosity models. However, the fully turbulent SST model still fails for natural low-Re LSB prediction because it assumes turbulent transport before transition.

### 3.10 gamma-Re_theta Transition Model

The $\gamma-Re_{\theta}$ transition model supplements SST with transport equations for intermittency $\gamma$ and transition momentum-thickness Reynolds number $\widetilde{Re}_{\theta t}$:

$$
\frac{\partial(\rho\gamma)}{\partial t}
+\frac{\partial(\rho u_j\gamma)}{\partial x_j}
=
P_\gamma-E_\gamma
+\frac{\partial}{\partial x_j}
\left[
\left(\mu+\frac{\mu_t}{\sigma_\gamma}\right)
\frac{\partial\gamma}{\partial x_j}
\right],
$$

$$
\frac{\partial(\rho\widetilde{Re}_{\theta t})}{\partial t}
+\frac{\partial(\rho u_j\widetilde{Re}_{\theta t})}{\partial x_j}
=
P_{\theta t}
+\frac{\partial}{\partial x_j}
\left[
\sigma_{\theta t}(\mu+\mu_t)
\frac{\partial\widetilde{Re}_{\theta t}}{\partial x_j}
\right].
$$

The intermittency variable modulates turbulence production so that eddy viscosity remains low in laminar regions and grows after transition onset. In separated-flow transition, the model permits the separated shear layer to transition and reattach if the predicted intermittency and turbulence production become large enough.

Transition modeling is required because LSB prediction depends on the ordering

$$
x_s < x_t < x_r,
$$

where $x_s$ is separation, $x_t$ is transition onset, and $x_r$ is reattachment. A fully turbulent model effectively moves $x_t$ to the leading edge, eliminating the laminar separation process. A laminar model has no turbulent reattachment mechanism. A transition model is therefore the minimum RANS-level closure capable of representing a closed LSB.

## 4. CFD Methodology

### 4.1 Geometry Parameterization

The airfoil is represented with separate CST functions for upper and lower surfaces. Let $\psi=x/c\in[0,1]$. The surface ordinate is

$$
\zeta(\psi)=\frac{z(x)}{c}=C(\psi)S(\psi)+\psi \zeta_{TE},
$$

where $C(\psi)$ is the class function, $S(\psi)$ is the shape function, and $\zeta_{TE}$ is the signed trailing-edge ordinate.

#### 4.1.1 Class Function

For a round-nose, sharp or finite-thickness trailing-edge airfoil:

$$
C(\psi)=\psi^{N_1}(1-\psi)^{N_2},
$$

with

$$
N_1=0.5,\qquad N_2=1.0.
$$

$N_1=0.5$ imposes square-root leading-edge behavior consistent with finite leading-edge radius. $N_2=1.0$ supports a sharp trailing-edge class when $\zeta_{TE}=0$ and finite trailing-edge thickness when upper and lower $\zeta_{TE}$ values differ.

#### 4.1.2 Bernstein Polynomials

The shape function is

$$
S(\psi)=\sum_{i=0}^{n} A_i B_i^n(\psi),
$$

where

$$
B_i^n(\psi)=\binom{n}{i}\psi^i(1-\psi)^{n-i}.
$$

Separate coefficients are used:

$$
\mathbf{a}=
\left[
A_{0,u},\ldots,A_{n,u},
A_{0,l},\ldots,A_{n,l},
\zeta_{TE,u},\zeta_{TE,l}
\right]^T.
$$

The baseline order is $n=7$, giving eight upper and eight lower coefficients. This provides enough flexibility to modify leading-edge suction peak, mid-chord pressure recovery, and aft loading without allowing uncontrolled high-frequency geometry oscillation.

#### 4.1.3 Thickness and Camber

Upper and lower surfaces are

$$
\zeta_u(\psi)=C(\psi)\sum_{i=0}^{n}A_{i,u}B_i^n(\psi)+\psi\zeta_{TE,u},
$$

$$
\zeta_l(\psi)=C(\psi)\sum_{i=0}^{n}A_{i,l}B_i^n(\psi)+\psi\zeta_{TE,l}.
$$

Thickness and camber are

$$
t(\psi)=\zeta_u(\psi)-\zeta_l(\psi),
$$

$$
m(\psi)=\frac{1}{2}\left[\zeta_u(\psi)+\zeta_l(\psi)\right].
$$

Thickness constraints are enforced at

$$
\psi_i\in\{0.02,0.05,0.10,0.20,0.30,0.40,0.60,0.80,0.95\}.
$$

The leading-edge radius is computed from the near-leading-edge curvature of the CST curve and constrained to

$$
0.004c \le r_{LE}\le 0.030c.
$$

### 4.2 Mesh Generation Methodology

#### 4.2.1 GMSH Workflow

The mesh is generated using GMSH with a C-type farfield topology. The computational domain extends:

$$
R_{far}=50c
$$

from the airfoil reference point. The wake block extends downstream to $60c$ to prevent wake truncation from contaminating drag and pressure recovery. The airfoil is placed with leading edge at $(0,0)$ and trailing edge at $(1,0)$ in chord-normalized coordinates.

Workflow:

1. Export CST coordinates with cosine spacing:

   $$
   \psi_j=\frac{1}{2}\left[1-\cos\left(\frac{j\pi}{N_s-1}\right)\right],
   \qquad N_s=401
   $$

   per surface before duplicate trailing-edge cleanup.

2. Build spline curves for upper and lower surfaces.
3. Define leading-edge refinement field with target size $2.5\times10^{-4}c$.
4. Define suction-side refinement field from $x/c=0.02$ to $0.80$ with target surface spacing $5.0\times10^{-4}c$.
5. Define trailing-edge refinement field with target size $2.0\times10^{-4}c$ in a radius $0.03c$.
6. Generate boundary-layer inflation using first-cell height computed from target $y^+$.
7. Export SU2 mesh with markers:

   ```
   airfoil
   farfield
   wake
   ```

#### 4.2.2 Structured vs Unstructured Meshes

Structured O- or C-grids provide excellent boundary-layer control, low numerical diffusion, and predictable stretching. They are preferred for final verification studies. Unstructured hybrid meshes with prism or quadrilateral inflation layers and triangular outer cells are easier to automate during optimization. This work uses automated hybrid meshes during optimization and structured C-grid confirmation for final designs.

The optimization mesh is accepted only if:

1. Orthogonality in the first 30 boundary-layer layers remains above 0.20 using the GMSH quality metric.
2. Maximum cell growth ratio normal to the wall is less than 1.18 in the first 40 layers.
3. Surface spacing changes by less than a factor of 1.25 between adjacent surface segments outside the trailing-edge cusp.
4. No negative Jacobian or folded cell occurs after mesh deformation.

#### 4.2.3 Inflation Layers and y-plus Requirements

The first-cell height is computed from

$$
y_1=\frac{y^+\mu_w}{\rho_w u_\tau},
$$

where

$$
u_\tau=\sqrt{\frac{\tau_w}{\rho_w}},
\qquad
\tau_w=\frac{1}{2}\rho_\infty U_\infty^2 C_f.
$$

For pre-mesh estimation, use the laminar flat-plate estimate at the leading half chord:

$$
C_{f,\ell}(x)=\frac{0.664}{\sqrt{Re_x}},
$$

with $Re_x$ evaluated at $x/c=0.05$ for conservative leading-edge spacing. The target is

$$
y^+_{\max}<1.0,\qquad y^+_{\text{mean}}<0.5
$$

on the airfoil. The baseline first-cell height is

$$
y_1/c = 1.0\times10^{-6}
$$

for $Re_c=2.0\times10^5$, adjusted linearly with $1/Re_c$ for other Reynolds numbers. Use 45 inflation layers, growth ratio 1.12, and total boundary-layer height at least

$$
\delta_{BL}/c=0.08
$$

to contain separated shear-layer development.

#### 4.2.4 Leading-Edge and Trailing-Edge Refinement

Leading-edge clustering is mandatory because transition and suction-peak behavior depend on the pressure-gradient history immediately after stagnation. The leading edge uses at least 80 points over $0\le x/c\le0.05$. The trailing edge uses at least 60 points over $0.95\le x/c\le1.00$ because pressure drag, wake momentum thickness, and Kutta-condition sensitivity are affected by trailing-edge resolution.

### 4.3 Solver Settings

The primary solver is SU2 using the RANS solver with SST and the Langtry-Menter transition model. If the installed SU2 build does not expose the transition-model option, that build is not used for final LSB claims; it may only be used for fully turbulent sensitivity bracketing. The exact SU2 option names must be checked against the local `config_template.cfg` because SU2 option names can change by version. The numerical settings below define the required content and tolerances.

#### 4.3.1 Baseline Flow Configuration

```
SOLVER= RANS
MATH_PROBLEM= DIRECT
KIND_TURB_MODEL= SST
KIND_TRANS_MODEL= LM
MACH_NUMBER= 0.10
AOA= 4.0
REYNOLDS_NUMBER= 200000.0
REYNOLDS_LENGTH= 1.0
FREESTREAM_TEMPERATURE= 288.15
FREESTREAM_PRESSURE= 101325.0
REF_LENGTH= 1.0
REF_AREA= 1.0
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )
```

This block is the baseline design-point configuration for $Re_c=2.0\times10^5$ and $\alpha=4^\circ$. Campaign execution modifies only `AOA` and `REYNOLDS_NUMBER` according to the case table defined in Sections 4.5 and 4.6. For adiabatic low-speed external flow, zero wall heat flux is used. If an incompressible solver is selected, match $Re_c$, $\alpha$, density, and dynamic viscosity explicitly.

#### 4.3.2 CFL Strategy

Use pseudo-time continuation:

1. Iterations 1-500: CFL = 0.5 to stabilize transition and separation onset.
2. Iterations 501-2000: ramp CFL linearly from 0.5 to 10.
3. Iterations 2001 onward: CFL = 10 to 30 if residuals decrease monotonically.
4. If lift oscillation amplitude exceeds $\Delta C_L=2.0\times10^{-4}$ over 500 iterations, reduce CFL by 50%.

Low-Re separated flows can converge to incorrect steady states if CFL is increased too early. The ramp prevents the solver from numerically bypassing the laminar separation and transition adjustment.

#### 4.3.3 Convergence Criteria

A steady solution is accepted only when all conditions are satisfied:

1. Density residual decreases by at least five orders of magnitude:

   $$
   \log_{10}\left(\frac{\|R_\rho^N\|_2}{\|R_\rho^0\|_2}\right)\le -5.
   $$

2. Momentum and turbulence residuals decrease by at least four orders of magnitude.
3. Rolling 500-iteration standard deviation satisfies:

   $$
   \sigma(C_D)<5.0\times10^{-6},\qquad
   \sigma(C_L)<2.0\times10^{-5}.
   $$

4. Rolling 500-iteration slope satisfies:

   $$
   \left|\frac{dC_D}{dN}\right|<1.0\times10^{-8}\ \text{per iteration},
   \qquad
   \left|\frac{dC_L}{dN}\right|<5.0\times10^{-8}\ \text{per iteration}.
   $$

5. Separation and reattachment locations vary by less than $0.002c$ over the final 500 iterations.

#### 4.3.4 Discretization

Use second-order spatial accuracy for final results:

```
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= YES
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
```

Roe flux with MUSCL reconstruction is selected for low dissipation in attached and mildly separated regions. Venkatakrishnan limiting reduces oscillations near strong gradients while preserving smooth pressure recovery. First-order solutions may be used only for initialization and never for final force or bubble diagnostics.

### 4.4 Boundary Conditions

Airfoil wall:

$$
\mathbf{u}=0,\qquad \frac{\partial T}{\partial n}=0.
$$

Farfield:

$$
M_\infty=0.10,\quad
Re_c\in\{1.0,2.0,3.0,5.0\}\times10^5,\quad
\alpha=\alpha_i.
$$

Reference quantities:

$$
c=1.0,\qquad A_{ref}=1.0,\qquad \mathbf{x}_{moment}=(0.25c,0,0).
$$

Transition inlet quantities must correspond to the experimental or assumed disturbance environment. Baseline free-stream turbulence intensity is

$$
Tu_\infty=0.10\%
$$

for low-turbulence wind-tunnel comparison. Sensitivity studies use

$$
Tu_\infty\in\{0.05\%,0.10\%,0.25\%,0.50\%,1.00\%\}.
$$

### 4.5 Reynolds Number Selection

The primary Reynolds numbers are selected to cover common low-Re airfoil regimes:

1. $Re_c=1.0\times10^5$: strong LSB sensitivity; bubble bursting likely.
2. $Re_c=2.0\times10^5$: transitional MAV and small UAV regime.
3. $Re_c=3.0\times10^5$: common low-Re airfoil test condition.
4. $Re_c=5.0\times10^5$: upper low-Re regime where transition remains important but bubbles are often shorter.

### 4.6 Angle-of-Attack Sweep Methodology

Static polars are computed using both cold-start and continuation methods:

1. Cold-start sweep: each $\alpha$ initialized from uniform free stream.
2. Increasing sweep: initialize each $\alpha_i$ from converged $\alpha_{i-1}$.
3. Decreasing sweep: initialize each $\alpha_i$ from converged $\alpha_{i+1}$.

The hysteresis indicator is

$$
\Delta C_L^{hys}(\alpha)=C_L^{up}(\alpha)-C_L^{down}(\alpha),
$$

and loop area is

$$
A_{hys}^{CL}=\int_{\alpha_1}^{\alpha_2}
\left|C_L^{up}(\alpha)-C_L^{down}(\alpha)\right|d\alpha.
$$

### 4.7 Mesh Independence Study

Use at least four systematically refined meshes:

| Mesh | Surface points | Inflation layers | First cell height | Approximate cells |
|---|---:|---:|---:|---:|
| M1 coarse | 401 | 35 | $2.0\times10^{-6}c$ | $8.0\times10^4$ |
| M2 medium | 601 | 45 | $1.0\times10^{-6}c$ | $1.8\times10^5$ |
| M3 fine | 901 | 55 | $5.0\times10^{-7}c$ | $4.0\times10^5$ |
| M4 extra fine | 1201 | 65 | $2.5\times10^{-7}c$ | $8.0\times10^5$ |

Report grid convergence for:

$$
C_L,\ C_D,\ C_m,\ x_s/c,\ x_t/c,\ x_r/c,\ L_b/c.
$$

### 4.8 Time-Step Sensitivity for URANS

URANS is required when steady residuals stagnate with periodic force oscillations or when separated-flow hysteresis is being studied. Use nondimensional time step:

$$
\Delta t^*=\frac{\Delta t U_\infty}{c}\in\{0.0025,0.005,0.010\}.
$$

Each URANS case must run for at least

$$
T^*=\frac{tU_\infty}{c}=100
$$

after initial transients. Statistics are computed over the final $T^*=50$. Acceptable time-step sensitivity requires:

$$
\left|\overline{C_D}_{\Delta t^*/2}-\overline{C_D}_{\Delta t^*}\right|<1.0\times10^{-4},
$$

and bubble-length RMS difference below $0.01c$.

### 4.9 Validation Strategy

Validation compares CFD with independent experimental or benchmark data using:

1. Force coefficients: $C_L(\alpha)$, $C_D(\alpha)$, $C_m(\alpha)$.
2. Surface pressure: $C_p(x/c)$ on upper and lower surfaces.
3. Transition location: hot-film, infrared, oil-flow, or PIV-derived location.
4. Separation and reattachment: oil-flow, skin-friction inference, near-wall PIV, or pressure plateau.
5. Wake: momentum thickness and velocity deficit at $x/c=1.5$ and $2.0$.

### 4.10 Experimental Comparability

CFD inputs must match experimental conditions:

$$
Re_c,\ M_\infty,\ \alpha,\ Tu_\infty,\ c,\ T_\infty,\ p_\infty,\ surface\ roughness,\ blockage,\ wall\ correction.
$$

If experimental turbulence intensity is unknown, do not tune transition parameters to match forces alone. Instead report a transition-sensitivity band over $Tu_\infty=\{0.05,0.10,0.25,0.50,1.00\}\%$.

## 5. Optimization Methodology

### 5.1 PDE-Constrained Optimization Formulation

For $N_p$ operating points, define

$$
\mathcal{P}=\{(Re_j,\alpha_j,M_j,w_j)\}_{j=1}^{N_p},
\qquad
\sum_{j=1}^{N_p}w_j=1.
$$

The optimization problem is

$$
\begin{aligned}
\min_{\mathbf{a},\{\mathbf{w}_j\},\{\mathbf{x}_j\}}
\quad &
J(\mathbf{a})=
\sum_{j=1}^{N_p}w_j
\left[
\frac{C_{D,j}(\mathbf{w}_j,\mathbf{x}_j)}{C_{D,j}^{base}}
+\lambda_b\frac{L_{b,j}}{c}
+\lambda_h\frac{H_j}{H^{base}}
\right]\\
\text{subject to}\quad
& \mathbf{R}_j(\mathbf{w}_j,\mathbf{x}_j,\mathbf{a})=\mathbf{0},
\quad j=1,\ldots,N_p,\\
& \mathbf{G}_j(\mathbf{x}_j,\mathbf{a})=\mathbf{0},
\quad j=1,\ldots,N_p,\\
& C_{L,j}\ge C_{L,j}^{target},\\
& t(\psi_i)\ge t_{\min}(\psi_i),\\
& |\kappa(\psi_i)-\kappa^{base}(\psi_i)|\le \Delta\kappa_{\max}(\psi_i),\\
& \mathbf{a}_{min}\le \mathbf{a}\le \mathbf{a}_{max}.
\end{aligned}
$$

$H_j$ is a hysteresis metric used only when sweep-based optimization is performed. For steady optimization without hysteresis, $\lambda_h=0$.

### 5.2 Objective Functions

#### 5.2.1 Drag Minimization

Single-point drag objective:

$$
J_D=\frac{C_D}{C_D^{base}}.
$$

Use this only for diagnostic studies because single-point drag minimization can create shapes that overfit one Reynolds number and angle of attack.

#### 5.2.2 Multi-Point Objective

Recommended objective:

$$
J_{MP}=
\sum_{j=1}^{N_p}w_j
\frac{C_{D,j}}{C_{D,j}^{base}},
$$

with operating set:

$$
\mathcal{P}=
\{(2.0\times10^5,2^\circ),(2.0\times10^5,4^\circ),(2.0\times10^5,6^\circ),
(3.0\times10^5,2^\circ),(3.0\times10^5,4^\circ),(3.0\times10^5,6^\circ)\}.
$$

Weights are uniform unless a mission analysis provides time fractions.

#### 5.2.3 Robustness Objective

Robust objective:

$$
J_R=\mathbb{E}[C_D]+\beta \sqrt{\mathrm{Var}(C_D)}+\lambda_b\mathbb{E}\left[\frac{L_b}{c}\right],
$$

where uncertainty variables are

$$
\xi=\left(Re_c,\alpha,Tu_\infty,\delta_{LE}\right),
$$

and $\delta_{LE}$ is a leading-edge roughness or shape perturbation amplitude. Use $\beta=1.0$ for balanced mean-variance optimization and $\beta=2.0$ for conservative robustness.

### 5.3 Constraints

#### 5.3.1 Lift Constraint

$$
C_L(Re_j,\alpha_j,\mathbf{a})\ge C_L^{base}(Re_j,\alpha_j)-0.01.
$$

The tolerance $0.01$ prevents accepting drag reductions produced by unloading the airfoil.

#### 5.3.2 Thickness Constraints

At the specified thickness stations:

$$
t(\psi_i;\mathbf{a})\ge 0.95\,t_{base}(\psi_i).
$$

If structural packaging is known, use absolute lower bounds instead:

$$
t(0.30)\ge0.10c,\quad t(0.50)\ge0.075c,\quad t(0.80)\ge0.025c.
$$

#### 5.3.3 Curvature Constraints

Curvature is

$$
\kappa(\psi)=\frac{x'(\psi)z''(\psi)-z'(\psi)x''(\psi)}
{\left[x'(\psi)^2+z'(\psi)^2\right]^{3/2}}.
$$

Constrain curvature change:

$$
|\kappa(\psi_i)-\kappa_{base}(\psi_i)|\le 25/c
$$

for $\psi_i$ spaced every $0.005$ over $0.01\le\psi\le0.99$. This prevents geometry ringing that can create mesh artifacts and unrealistic pressure spikes.

#### 5.3.4 Manufacturability Constraints

Manufacturability constraints:

$$
t_{TE}\ge0.001c,\qquad t_{TE}\le0.006c,
$$

$$
|\theta_{TE,u}-\theta_{TE,l}|\le18^\circ,
$$

$$
r_{LE}\ge0.004c,
$$

and no surface self-intersection:

$$
\zeta_u(\psi_i)-\zeta_l(\psi_i)>0
\quad\forall \psi_i\in(0,1).
$$

### 5.4 MMA Algorithm

MMA solves a sequence of convex approximating subproblems. At iteration $k$, define lower and upper moving asymptotes:

$$
L_i^{(k)}<a_i^{(k)}<U_i^{(k)}.
$$

The objective approximation has the form

$$
\hat{J}^{(k)}(\mathbf{a})=
r_0^{(k)}
\sum_{i=1}^{n_a}
\left[
\frac{p_{0i}^{(k)}}{U_i^{(k)}-a_i}
+\frac{q_{0i}^{(k)}}{a_i-L_i^{(k)}}
\right],
$$

with analogous approximations for constraints. Coefficients $p$ and $q$ are chosen from local gradients so that the approximation matches first-order behavior at $\mathbf{a}^{(k)}$ and remains convex.

### 5.5 Reciprocal Approximation Theory

The reciprocal form is effective when responses behave nonlinearly with respect to shape variables. If increasing a CST coefficient steepens adverse pressure recovery, drag may change rapidly near separation onset. Reciprocal asymptotes restrict the local approximation near bounds, preventing steps that would cross into invalid or highly nonlinear geometry-flow states.

### 5.6 Move Limits

Design variables are nondimensionalized:

$$
\hat{a}_i=\frac{a_i-a_i^{base}}{s_i}.
$$

Use initial move limit:

$$
|\Delta \hat{a}_i|\le0.05.
$$

If the objective decreases and all constraints remain feasible for three consecutive iterations, increase to 0.075. If any CFD case fails convergence, any geometry constraint is violated, or finite-difference gradient mismatch exceeds 10%, reduce to 0.025.

Move limits exist because transition-dominated objectives are locally reliable only within small geometric perturbations. They also maintain mesh deformation quality.

### 5.7 KKT Optimality

The Lagrangian is

$$
\mathcal{L}(\mathbf{a},\boldsymbol{\lambda},\boldsymbol{\mu})
=J(\mathbf{a})
+\sum_{m=1}^{M}\lambda_m g_m(\mathbf{a})
+\sum_{i=1}^{n_a}
\mu_i^+(a_i-a_i^{max})
+\sum_{i=1}^{n_a}
\mu_i^-(a_i^{min}-a_i).
$$

KKT conditions:

$$
\nabla_{\mathbf{a}}\mathcal{L}=0,
$$

$$
g_m(\mathbf{a})\le0,
$$

$$
\lambda_m\ge0,\quad \lambda_m g_m(\mathbf{a})=0,
$$

and analogous bound complementarity. Optimization is considered converged when:

$$
\|\nabla_{\mathbf{a}}\mathcal{L}\|_\infty<1.0\times10^{-4},
$$

constraint violation is below $1.0\times10^{-5}$ in normalized units, and relative objective reduction over five iterations is below $1.0\times10^{-3}$.

### 5.8 Trust-Region Stabilization

Trust-region acceptance ratio:

$$
\rho^{(k)}=
\frac{J(\mathbf{a}^{(k)})-J(\mathbf{a}^{trial})}
{\hat{J}^{(k)}(\mathbf{a}^{(k)})-\hat{J}^{(k)}(\mathbf{a}^{trial})}.
$$

Rules:

1. Accept if $\rho^{(k)}\ge0.25$ and constraints are feasible.
2. Expand move limits by 20% if $\rho^{(k)}\ge0.75$ for two consecutive accepted steps.
3. Reject and halve move limits if $\rho^{(k)}<0.25$.
4. Reject if any bubble diagnostic jumps by more than $0.08c$ without a corresponding smooth pressure-distribution change.

Trust-region stabilization exists because local linear or reciprocal models can be invalid near separation-state changes.

### 5.9 Adjoint Sensitivity Derivation

Let the discrete residual be

$$
\mathbf{R}(\mathbf{w},\mathbf{x},\mathbf{a})=\mathbf{0}.
$$

The total derivative of objective $J$ is

$$
\frac{dJ}{d\mathbf{a}}
=
\frac{\partial J}{\partial \mathbf{a}}
+\frac{\partial J}{\partial \mathbf{w}}\frac{d\mathbf{w}}{d\mathbf{a}}
+\frac{\partial J}{\partial \mathbf{x}}\frac{d\mathbf{x}}{d\mathbf{a}}.
$$

Differentiate the residual:

$$
\frac{\partial \mathbf{R}}{\partial \mathbf{w}}\frac{d\mathbf{w}}{d\mathbf{a}}
+\frac{\partial \mathbf{R}}{\partial \mathbf{x}}\frac{d\mathbf{x}}{d\mathbf{a}}
+\frac{\partial \mathbf{R}}{\partial \mathbf{a}}=0.
$$

Define adjoint vector $\boldsymbol{\lambda}$:

$$
\left(\frac{\partial \mathbf{R}}{\partial \mathbf{w}}\right)^T
\boldsymbol{\lambda}
=
\left(\frac{\partial J}{\partial \mathbf{w}}\right)^T.
$$

Then

$$
\frac{dJ}{d\mathbf{a}}
=
\frac{\partial J}{\partial \mathbf{a}}
-\boldsymbol{\lambda}^T\frac{\partial \mathbf{R}}{\partial \mathbf{a}}
+\left(
\frac{\partial J}{\partial \mathbf{x}}
-\boldsymbol{\lambda}^T\frac{\partial \mathbf{R}}{\partial \mathbf{x}}
\right)
\frac{d\mathbf{x}}{d\mathbf{a}}.
$$

For discrete adjoint SU2 workflows, the direct solution, adjoint solution, mesh sensitivity, and projection onto design variables must all be included.

### 5.10 Gradient Consistency Verification

Use central finite differences for selected design variables:

$$
\left(\frac{dJ}{da_i}\right)_{FD}
=
\frac{J(a_i+h)-J(a_i-h)}{2h}.
$$

Step sizes:

$$
h/s_i\in\{10^{-2},10^{-3},10^{-4}\}.
$$

The accepted comparison uses the plateau region where truncation and iterative noise are both small. Relative gradient error:

$$
\epsilon_i=
\frac{|g_{adj,i}-g_{FD,i}|}
{\max(1,|g_{FD,i}|)}.
$$

Requirements:

$$
\mathrm{median}(\epsilon_i)<0.05,\qquad
\max(\epsilon_i)<0.20
$$

for non-negligible gradient components. If violated, check residual convergence, limiter differentiability, transition-model consistency, mesh deformation, and objective smoothness.

### 5.11 Multi-Fidelity Hierarchy

Fidelity levels:

1. Level 0: Geometry feasibility only; rejects self-intersection, excessive curvature, invalid thickness.
2. Level 1: XFOIL or panel-boundary-layer screening; used only to discard clearly poor shapes, not to certify LSB suppression.
3. Level 2: Coarse transition-aware RANS; used for early optimization iterations.
4. Level 3: Fine transition-aware RANS; used for final optimization and reporting.
5. Level 4: URANS or higher-fidelity comparison near stall and hysteresis.
6. Level 5: Experimental validation or benchmark comparison.

The hierarchy exists to reduce computational cost while preventing low-fidelity model bias from determining final conclusions.

## 6. LSB Detection and Flow Diagnostics

### 6.1 Separation Location

Surface skin friction coefficient:

$$
C_f(x)=\frac{\tau_w(x)}{\frac{1}{2}\rho_\infty U_\infty^2},
\qquad
\tau_w=\mu\left.\frac{\partial u_t}{\partial n}\right|_w.
$$

Separation occurs where $C_f$ changes from positive to negative on the suction side:

$$
x_s=\min_x\{x:C_f(x)=0,\ dC_f/dx<0\}.
$$

Compute by cubic interpolation between adjacent surface nodes with opposite $C_f$ sign. Require negative $C_f$ over at least five consecutive wall faces or over $0.002c$, whichever is larger, to reject numerical sign noise.

### 6.2 Reattachment Location

Reattachment occurs where $C_f$ changes from negative to positive downstream of separation:

$$
x_r=\min_{x>x_s}\{x:C_f(x)=0,\ dC_f/dx>0\}.
$$

If no reattachment occurs before the trailing edge, classify as open separation and set $x_r/c>1$ for plotting while reporting "no closed bubble".

### 6.3 Bubble Length

Closed-bubble length:

$$
L_b=x_r-x_s.
$$

Normalized:

$$
L_b^*=\frac{L_b}{c}.
$$

For multiple separated regions, report the primary suction-side bubble as the longest interval with $C_f<0$ and also archive all intervals.

### 6.4 Shape Factor

Boundary-layer displacement and momentum thickness:

$$
\delta^*(x)=\int_0^{\delta}
\left(1-\frac{u(y)}{U_e}\right)dy,
$$

$$
\theta(x)=\int_0^{\delta}
\frac{u(y)}{U_e}
\left(1-\frac{u(y)}{U_e}\right)dy.
$$

Shape factor:

$$
H(x)=\frac{\delta^*(x)}{\theta(x)}.
$$

$U_e(x)$ is extracted from the velocity maximum just outside the boundary layer or from inviscid edge reconstruction. The boundary-layer edge $\delta$ is the smallest wall-normal location satisfying

$$
\left|\frac{u(y)-U_e}{U_e}\right|<0.005
$$

for all subsequent points over at least three grid nodes. A sharp rise in $H$ indicates separation-prone laminar boundary-layer behavior.

### 6.5 Skin Friction Coefficient

Compute $C_f$ directly from wall shear:

$$
C_f=\frac{2\tau_w}{\rho_\infty U_\infty^2}.
$$

For curved surfaces, project viscous stress onto the local tangent:

$$
\tau_w=\mathbf{t}\cdot(\boldsymbol{\tau}\mathbf{n}),
$$

where $\mathbf{t}$ is the unit tangent and $\mathbf{n}$ is the outward wall normal.

### 6.6 Transition Onset

Transition onset is extracted using the first location downstream of the leading edge where intermittency exceeds a threshold:

$$
x_t=\min_x\{x:\gamma(x)\ge0.5,\ x>x_s\ \text{or}\ x>0.01c\}.
$$

If $\gamma$ is unavailable, use turbulent-to-laminar viscosity ratio:

$$
\frac{\mu_t}{\mu}>1
$$

combined with a positive spatial growth criterion. For validation with XFOIL-like outputs, archive $x_{tr,upper}$ and $x_{tr,lower}$ separately but do not equate them with separation or reattachment.

### 6.7 Pressure Recovery

Pressure coefficient:

$$
C_p(x)=\frac{p(x)-p_\infty}{\frac{1}{2}\rho_\infty U_\infty^2}.
$$

Pressure recovery over a suction-side interval $[x_a,x_b]$ is

$$
R_p(x_a,x_b)=C_p(x_b)-C_p(x_a).
$$

The adverse pressure-gradient metric is

$$
APG_{max}=\max_{x\in[x_{suction},x_{TE}]}\frac{dC_p}{d(x/c)}.
$$

Use Savitzky-Golay smoothing with polynomial order 3 and window length corresponding to $0.01c$ before differentiating $C_p$.

### 6.8 Wake Thickness

At wake station $x_w/c=1.5$ and $2.0$, compute momentum thickness:

$$
\theta_w=\int_{-\infty}^{\infty}
\frac{u(y)}{U_\infty}
\left(1-\frac{u(y)}{U_\infty}\right)dy.
$$

Wake deficit thickness:

$$
\delta_d=\int_{-\infty}^{\infty}
\left(1-\frac{u(y)}{U_\infty}\right)dy.
$$

Integrate over the region where velocity deficit exceeds 0.5% of $U_\infty$.

### 6.9 Stall Onset

Stall onset is identified by the first angle where any two conditions occur:

1. $dC_L/d\alpha$ falls below 50% of the linear-region slope.
2. $C_L$ decreases between adjacent increasing-$\alpha$ points.
3. $L_b/c$ increases by more than 0.10 over one degree.
4. Reattachment is lost on the suction side.
5. URANS lift RMS exceeds 2% of mean lift.

### 6.10 Hysteresis Loops

Compute increasing and decreasing sweeps with identical $\alpha$ grids. Interpolate $C_L$, $C_D$, $x_s$, $x_r$, and $L_b$ onto a common grid. Loop areas:

$$
A_{hys}^{CD}=\int_{\alpha_1}^{\alpha_2}|C_D^{up}-C_D^{down}|d\alpha,
$$

$$
A_{hys}^{Lb}=\int_{\alpha_1}^{\alpha_2}|L_b^{up}-L_b^{down}|d\alpha.
$$

Report the sign and magnitude of state dependence, not only the area.

## 7. Data Analysis Pipeline in R

### 7.1 R Environment Setup

#### 7.1.1 Package Stack

Install the mandatory packages:

```r
required_packages <- c(
  "tidyverse", "data.table", "ggplot2", "plotly", "patchwork",
  "broom", "readr", "tidyr", "stringr", "janitor", "scales",
  "viridis", "RColorBrewer", "zoo", "signal", "pracma",
  "mgcv", "nlme", "lme4", "forecast", "MASS", "car",
  "performance", "emmeans", "rstatix", "effectsize",
  "BayesianTools", "posterior", "brms", "loo", "bayesplot"
)

installed <- rownames(installed.packages())
to_install <- setdiff(required_packages, installed)
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

invisible(lapply(required_packages, library, character.only = TRUE))
```

For Bayesian modeling with `brms`, install CmdStanR:

```r
install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
cmdstanr::install_cmdstan(cores = parallel::detectCores())
```

#### 7.1.2 Project Folder Structure

```
airfoil_lsb_optimization/
  README.md
  renv.lock
  config/
    cases.csv
    mesh_levels.csv
    optimization_points.csv
    plotting_theme.yml
  geometry/
    baseline/
    optimized/
    cst_coefficients/
  meshes/
    M1_coarse/
    M2_medium/
    M3_fine/
    M4_extra_fine/
  cfd/
    raw/
      Re0200k_AOA04p00_M2_run001/
    processed/
    logs/
  optimization/
    history/
    gradients/
    constraints/
  diagnostics/
    cp/
    cf/
    transition/
    wake/
    bubble/
  analysis/
    scripts/
    cache/
    models/
  figures/
    draft/
    final_pdf/
    final_svg/
    final_png/
  tables/
  manuscript/
```

#### 7.1.3 Reproducibility Settings

Use `renv`:

```r
install.packages("renv", repos = "https://cloud.r-project.org")
renv::init()
renv::snapshot()
```

Record:

```r
sessionInfo()
Sys.info()
R.version.string
```

Store solver version:

```r
system("SU2_CFD --version", intern = TRUE)
system("gmsh --version", intern = TRUE)
```

#### 7.1.4 Seed Handling

Use deterministic seeds for stochastic sampling:

```r
set.seed(20260517)
RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
```

For parallel Bayesian chains:

```r
options(mc.cores = parallel::detectCores())
seed_master <- 20260517
chain_seeds <- seed_master + seq_len(4)
```

#### 7.1.5 File Naming Conventions

Use:

```
{airfoil_id}_Re{Re_k}_M{Mach}_AOA{alpha}_Tu{Tu}_mesh{mesh}_sweep{sweep}_run{run}_{quantity}.csv
```

Example:

```
OPT001_Re0200k_M0p10_AOA04p00_Tu0p10_meshM3_sweepup_run001_cp.csv
```

Rules:

1. Reynolds number in thousands with four digits: `0200k`.
2. Decimal points replaced with `p`.
3. Increasing sweep: `sweepup`; decreasing sweep: `sweepdown`; cold start: `sweepcold`.
4. Runs numbered with three digits.

#### 7.1.6 Metadata Standards

Each processed dataset must include:

| Field | Meaning |
|---|---|
| `airfoil_id` | baseline or optimized geometry identifier |
| `cst_hash` | SHA-256 hash of CST coefficient file |
| `mesh_id` | mesh level and mesh hash |
| `solver` | SU2 solver executable |
| `solver_version` | exact SU2 version string |
| `turb_model` | turbulence model |
| `transition_model` | transition model |
| `re_c` | chord Reynolds number |
| `mach` | free-stream Mach number |
| `alpha_deg` | angle of attack |
| `tu_percent` | free-stream turbulence intensity |
| `sweep_direction` | cold, up, or down |
| `run_id` | repeated run identifier |
| `converged` | logical convergence flag |
| `residual_drop` | density residual reduction |
| `cl_mean` | final or time-averaged lift |
| `cd_mean` | final or time-averaged drag |
| `xs_c` | separation location |
| `xt_c` | transition location |
| `xr_c` | reattachment location |
| `lb_c` | bubble length |

### 7.2 Data Pipeline

#### 7.2.1 CFD Result Ingestion

```r
library(tidyverse)
library(data.table)
library(janitor)

case_table <- readr::read_csv("config/cases.csv", show_col_types = FALSE) |>
  clean_names()

list_case_dirs <- function(root = "cfd/raw") {
  tibble(case_dir = list.dirs(root, recursive = FALSE, full.names = TRUE)) |>
    mutate(case_id = basename(case_dir))
}
```

#### 7.2.2 SU2 History Parsing

```r
parse_su2_history <- function(path) {
  raw <- readLines(path, warn = FALSE)
  header_line <- which(stringr::str_detect(raw, "^%")) |> tail(1)
  header <- raw[header_line] |>
    stringr::str_remove("^%") |>
    stringr::str_split(",") |>
    unlist() |>
    stringr::str_trim() |>
    janitor::make_clean_names()

  data.table::fread(path, skip = header_line, header = FALSE) |>
    as_tibble() |>
    setNames(header) |>
    mutate(iter = row_number())
}
```

Convergence metrics:

```r
summarise_history <- function(hist) {
  tail_window <- hist |> slice_tail(n = 500)
  hist |>
    summarise(
      n_iter = n(),
      cl_final = last(cl),
      cd_final = last(cd),
      cl_sd_500 = sd(tail_window$cl, na.rm = TRUE),
      cd_sd_500 = sd(tail_window$cd, na.rm = TRUE),
      cl_slope_500 = coef(lm(cl ~ iter, data = tail_window))[2],
      cd_slope_500 = coef(lm(cd ~ iter, data = tail_window))[2],
      residual_drop_density = first(res_flow_0) - last(res_flow_0)
    )
}
```

#### 7.2.3 Cp Distribution Parsing

```r
parse_surface_csv <- function(path) {
  readr::read_csv(path, show_col_types = FALSE) |>
    clean_names() |>
    mutate(
      x_c = x_coord / max(x_coord, na.rm = TRUE),
      surface = if_else(y_coord >= 0, "upper", "lower")
    )
}
```

Pressure coefficient must be normalized consistently:

```r
compute_cp <- function(surface, p_inf, rho_inf, u_inf) {
  surface |>
    mutate(cp = (pressure - p_inf) / (0.5 * rho_inf * u_inf^2))
}
```

#### 7.2.4 Residual Parsing

Residuals are parsed from history files and converted to normalized residual drops. If SU2 outputs logarithmic residuals, keep them in log form. If raw norms are output, convert:

```r
normalize_residuals <- function(hist, residual_cols) {
  hist |>
    mutate(across(all_of(residual_cols), ~ log10(.x / first(.x)), .names = "{.col}_logdrop"))
}
```

#### 7.2.5 Multi-Run Aggregation

```r
aggregate_runs <- function(metrics) {
  metrics |>
    group_by(airfoil_id, re_c, mach, alpha_deg, tu_percent, mesh_id, sweep_direction) |>
    summarise(
      n_runs = n(),
      cl_mean = mean(cl_final, na.rm = TRUE),
      cl_sd = sd(cl_final, na.rm = TRUE),
      cd_mean = mean(cd_final, na.rm = TRUE),
      cd_sd = sd(cd_final, na.rm = TRUE),
      lb_c_mean = mean(lb_c, na.rm = TRUE),
      lb_c_sd = sd(lb_c, na.rm = TRUE),
      .groups = "drop"
    )
}
```

#### 7.2.6 Optimization History Tracking

Optimization history file columns:

```
iteration, objective, cd_weighted, lb_weighted, max_constraint,
kkt_norm, trust_ratio, move_limit, accepted, mesh_failures,
gradient_median_error, gradient_max_error
```

Read and check:

```r
opt_hist <- readr::read_csv("optimization/history/opt_history.csv", show_col_types = FALSE) |>
  clean_names() |>
  mutate(accepted = as.logical(accepted))
```

#### 7.2.7 Mesh-Comparison Processing

```r
mesh_metrics <- readr::read_csv("diagnostics/mesh_convergence.csv", show_col_types = FALSE) |>
  clean_names() |>
  arrange(airfoil_id, re_c, alpha_deg, mesh_level)
```

Estimate apparent order:

```r
estimate_order_three_grids <- function(f1, f2, f3, r) {
  log(abs((f3 - f2) / (f2 - f1))) / log(r)
}
```

#### 7.2.8 Transition-Location Extraction

```r
extract_transition <- function(surface, threshold = 0.5) {
  surface |>
    filter(surface == "upper") |>
    arrange(x_c) |>
    mutate(gamma_lag = lag(intermittency)) |>
    filter(gamma_lag < threshold, intermittency >= threshold) |>
    slice_head(n = 1) |>
    transmute(xt_c = x_c)
}
```

If intermittency is absent:

```r
extract_transition_mut <- function(surface, mut_threshold = 1.0) {
  surface |>
    filter(surface == "upper") |>
    arrange(x_c) |>
    mutate(ratio = turbulent_viscosity / laminar_viscosity,
           ratio_lag = lag(ratio)) |>
    filter(ratio_lag < mut_threshold, ratio >= mut_threshold) |>
    slice_head(n = 1) |>
    transmute(xt_c = x_c)
}
```

#### 7.2.9 Hysteresis-Loop Generation

```r
make_hysteresis <- function(polar_data) {
  up <- polar_data |> filter(sweep_direction == "up")
  down <- polar_data |> filter(sweep_direction == "down")

  common_alpha <- intersect(up$alpha_deg, down$alpha_deg)

  up_i <- up |> filter(alpha_deg %in% common_alpha)
  down_i <- down |> filter(alpha_deg %in% common_alpha)

  inner_join(
    up_i, down_i,
    by = c("airfoil_id", "re_c", "mach", "tu_percent", "alpha_deg"),
    suffix = c("_up", "_down")
  ) |>
    mutate(
      delta_cl = cl_mean_up - cl_mean_down,
      delta_cd = cd_mean_up - cd_mean_down,
      delta_lb_c = lb_c_mean_up - lb_c_mean_down
    )
}
```

Loop integration:

```r
trapz_loop_area <- function(alpha, delta) {
  pracma::trapz(alpha, abs(delta))
}
```

#### 7.2.10 Statistical Repeatability Analysis

Repeatability is assessed from repeated solver runs with identical inputs:

```r
repeatability <- metrics |>
  group_by(airfoil_id, re_c, alpha_deg, mesh_id, sweep_direction) |>
  summarise(
    cd_repeat_sd = sd(cd_final),
    cl_repeat_sd = sd(cl_final),
    lb_repeat_sd = sd(lb_c),
    cd_repeat_cv = sd(cd_final) / mean(cd_final),
    .groups = "drop"
  )
```

### 7.3 Statistical Analysis

#### 7.3.1 Uncertainty Quantification

Use uncertainty quantification when comparing baseline and optimized designs under uncertain $Re_c$, $\alpha$, $Tu_\infty$, and geometry perturbations. Use Latin hypercube or Sobol sampling for input variables, then compute output distributions for $C_D$, $C_L$, and $L_b/c$.

For scalar output $Y$:

$$
\bar{Y}=\frac{1}{N}\sum_{i=1}^{N}Y_i,
\qquad
s_Y^2=\frac{1}{N-1}\sum_{i=1}^{N}(Y_i-\bar{Y})^2.
$$

Use when claiming robust improvement rather than single-condition improvement.

#### 7.3.2 Confidence Intervals

For repeated deterministic runs or experimental comparisons:

$$
\bar{Y}\pm t_{0.975,n-1}\frac{s}{\sqrt{n}}.
$$

For non-normal data or small repeated samples, use bootstrap confidence intervals:

```r
bootstrap_ci <- function(x, B = 5000, conf = 0.95) {
  boots <- replicate(B, mean(sample(x, replace = TRUE), na.rm = TRUE))
  quantile(boots, probs = c((1-conf)/2, 1-(1-conf)/2), na.rm = TRUE)
}
```

#### 7.3.3 Sensitivity Analysis

Use regression, Morris screening, or variance-based sensitivity depending on computational budget. For local sensitivity:

$$
S_i=\frac{\partial Y}{\partial \xi_i}\frac{\xi_i}{Y}.
$$

Use sensitivity analysis to identify whether optimized performance depends primarily on Reynolds number, turbulence intensity, or angle of attack.

#### 7.3.4 ANOVA

Use ANOVA for balanced factorial comparisons such as design $\times$ mesh $\times$ Reynolds number when residuals are approximately normal and independent:

```r
aov_cd <- aov(cd_mean ~ airfoil_id * mesh_id * factor(re_c), data = polar_summary)
car::Anova(aov_cd, type = 3)
```

Use ANOVA to test whether design differences exceed mesh and operating-condition variation. Do not use ANOVA for strongly correlated repeated angle sweeps without repeated-measures structure.

#### 7.3.5 Repeated-Measures Analysis

Use repeated-measures analysis when the same airfoil is evaluated across the same angles of attack:

```r
rstatix::anova_test(
  data = polar_summary,
  dv = cd_mean,
  wid = airfoil_id,
  within = alpha_deg
)
```

This accounts for correlation across angle samples.

#### 7.3.6 Mixed-Effects Modeling

Use mixed models when cases include repeated runs, mesh levels, airfoils, and Reynolds numbers:

```r
library(lme4)
model_cd <- lmer(
  cd_mean ~ airfoil_id * alpha_deg + factor(re_c) + (1 | mesh_id) + (1 | run_id),
  data = polar_summary
)
performance::check_model(model_cd)
emmeans::emmeans(model_cd, pairwise ~ airfoil_id | alpha_deg)
```

Mixed models are preferred when mesh or repeated-run effects are nuisance sources of variation.

#### 7.3.7 Bayesian Analysis

Use Bayesian modeling when uncertainty statements must propagate limited data, model hierarchy, or prior engineering constraints:

```r
library(brms)
bayes_cd <- brm(
  cd_mean ~ airfoil_id * alpha_deg + factor(re_c) + (1 | mesh_id),
  data = polar_summary,
  family = gaussian(),
  prior = c(
    prior(normal(0, 0.01), class = "b"),
    prior(exponential(100), class = "sigma")
  ),
  chains = 4,
  iter = 4000,
  warmup = 1000,
  seed = 20260517
)
loo::loo(bayes_cd)
bayesplot::mcmc_trace(as.array(bayes_cd))
```

Report posterior probability:

$$
P(C_{D,opt}<C_{D,base}\mid data).
$$

#### 7.3.8 Convergence-Rate Analysis

Fit residual decay:

$$
\log_{10}R(N)=a+bN
$$

over monotonic intervals. Use `forecast` or rolling-window regression to detect stagnation:

```r
hist |>
  mutate(res_roll_slope = zoo::rollapply(
    res_flow_0, width = 200,
    FUN = function(z) coef(lm(z ~ seq_along(z)))[2],
    fill = NA, align = "right"
  ))
```

Use when distinguishing converged steady solutions from pseudo-converged oscillatory separated solutions.

#### 7.3.9 Gradient Consistency Statistics

Analyze adjoint vs finite-difference gradients:

```r
grad_stats <- gradients |>
  mutate(
    rel_error = abs(g_adj - g_fd) / pmax(1, abs(g_fd)),
    sign_match = sign(g_adj) == sign(g_fd)
  ) |>
  summarise(
    median_rel_error = median(rel_error, na.rm = TRUE),
    max_rel_error = max(rel_error, na.rm = TRUE),
    sign_match_rate = mean(sign_match, na.rm = TRUE)
  )
```

Use Bland-Altman plots for gradient comparison:

$$
d_i=g_{adj,i}-g_{FD,i},\qquad
m_i=\frac{g_{adj,i}+g_{FD,i}}{2}.
$$

#### 7.3.10 Mesh-Sensitivity Quantification

For quantity $\phi$, compute relative mesh sensitivity:

$$
S_{mesh}=\frac{|\phi_{M3}-\phi_{M2}|}{|\phi_{M3}|}.
$$

Require:

$$
S_{mesh}(C_D)<0.02,\qquad
S_{mesh}(L_b)<0.05
$$

before using results for final conclusions.

### 7.4 Visualization Requirements

#### 7.4.1 General Plot Settings

Use:

```r
theme_paper <- function(base_size = 9) {
  ggplot2::theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.2, colour = "grey85"),
      axis.title = element_text(colour = "black"),
      axis.text = element_text(colour = "black"),
      legend.position = "top",
      legend.title = element_blank()
    )
}
```

Export:

```r
ggsave("figures/final_pdf/figure_name.pdf", width = 90, height = 65, units = "mm", device = cairo_pdf)
ggsave("figures/final_svg/figure_name.svg", width = 90, height = 65, units = "mm")
ggsave("figures/final_png/figure_name.png", width = 90, height = 65, units = "mm", dpi = 600)
```

Use vector PDF/SVG for line plots and 600 DPI PNG/TIFF for contours.

#### 7.4.2 Cp Plots

Plot $-C_p$ vs $x/c$:

1. x-axis: $0\le x/c\le1$.
2. y-axis: inverted pressure convention using $-C_p$ upward.
3. Separate upper and lower surfaces by line type.
4. Mark $x_s$, $x_t$, and $x_r$ with vertical dashed lines.
5. Use colorblind-safe palette: `viridis::scale_colour_viridis_d(option = "D")`.

#### 7.4.3 Cl/Cd Polar Plots

Plot $C_L$ vs $C_D$ and $C_L/C_D$ vs $\alpha$. Use identical axis limits for baseline and optimized designs. Include uncertainty bars when repeated runs or UQ samples exist.

#### 7.4.4 Drag Buckets

Plot $C_D$ vs $C_L$ across Reynolds numbers. Use faceting by $Re_c$ and color by airfoil. A drag bucket is reported only if the low-drag region persists across at least two adjacent $\alpha$ points and is not a single noisy minimum.

#### 7.4.5 Hysteresis Loops

Plot increasing and decreasing sweeps using arrows or line types. Report loop area directly in the panel subtitle:

$$
A_{hys}^{CL},\quad A_{hys}^{CD},\quad A_{hys}^{Lb}.
$$

#### 7.4.6 Residual Convergence Plots

Plot residuals on log scale:

$$
y=\log_{10}(R/R_0).
$$

Overlay $C_L$ and $C_D$ convergence on a secondary panel rather than a secondary y-axis.

#### 7.4.7 KKT Convergence

Plot objective, maximum constraint violation, KKT norm, move limit, and trust ratio vs iteration. Use log scale for KKT norm and constraint violation.

#### 7.4.8 Gradient Consistency

Use:

1. Scatter plot: $g_{FD}$ vs $g_{adj}$ with 1:1 line.
2. Relative error bar plot by design variable.
3. Finite-difference step-size convergence plot.

#### 7.4.9 Bubble-Length Maps

Plot filled contours of $L_b/c$ over $(Re_c,\alpha)$:

1. x-axis: $\alpha$.
2. y-axis: $Re_c$ on log or discrete scale.
3. color: $L_b/c$ using `viridis` option `C`.
4. mark open separation with hatching or point symbol.

#### 7.4.10 Contour Visualizations

For CFD contours:

1. Show Mach number, vorticity, intermittency, or $\mu_t/\mu$.
2. Use the same contour levels across baseline and optimized cases.
3. Crop to $-0.05\le x/c\le1.2$, $-0.2\le y/c\le0.2$ for LSB detail.
4. Include surface streamlines or skin-friction sign overlay.

#### 7.4.11 Pressure Recovery Plots

Plot $dC_p/d(x/c)$ vs $x/c$ using smoothed derivatives. Highlight regions where $dC_p/dx>0$ and annotate $APG_{max}$.

## 8. Validation and Verification

### 8.1 Code Verification

Code verification establishes that the numerical implementation solves the equations correctly. Use:

1. Manufactured solutions for laminar compressible Navier-Stokes if the local solver distribution includes a manufactured-solution test case; otherwise the code-verification evidence is limited to canonical regression cases and this limitation is stated explicitly.
2. Standard laminar flat-plate comparison for $C_f$:

   $$
   C_{f,x}=\frac{0.664}{\sqrt{Re_x}}.
   $$

3. Turbulent flat-plate comparison for fully turbulent model sanity checks.
4. Inviscid airfoil pressure comparison against panel-method trends at very high Reynolds number and attached low angle.

### 8.2 Solution Verification

Solution verification quantifies numerical error for the actual problem. It includes:

1. Iterative convergence study.
2. Mesh convergence study.
3. Time-step convergence study for URANS.
4. Sensitivity to limiter, gradient reconstruction, and farfield distance.

### 8.3 Validation Hierarchy

Validation levels:

1. Integral forces only: $C_L$, $C_D$, $C_m$.
2. Surface pressure: $C_p(x)$.
3. Skin friction or separation markers.
4. Transition location.
5. Velocity field: PIV or wake survey.
6. Hysteresis loops with matched sweep protocol.

Claims about LSB suppression require at least levels 1-4. Claims about wake-loss reduction require level 5.

### 8.4 Mesh Convergence Index

For three meshes with refinement ratio $r$ and solutions $\phi_1$ fine, $\phi_2$ medium, $\phi_3$ coarse, apparent order:

$$
p=\frac{\ln\left|\frac{\phi_3-\phi_2}{\phi_2-\phi_1}\right|}{\ln(r)}.
$$

Richardson extrapolated value:

$$
\phi_{ext}=\phi_1+\frac{\phi_1-\phi_2}{r^p-1}.
$$

Fine-grid GCI:

$$
GCI_{12}=F_s\frac{|\phi_1-\phi_2|}{|\phi_1|(r^p-1)},
$$

with safety factor

$$
F_s=1.25
$$

for at least three systematically refined grids.

### 8.5 Richardson Extrapolation

Use Richardson extrapolation only when solutions are in the asymptotic range. Check:

$$
\frac{GCI_{23}}{r^pGCI_{12}}\approx 1.
$$

If the ratio differs strongly from 1, report observed mesh sensitivity without claiming asymptotic convergence.

### 8.6 Adjoint Verification

Adjoint verification requires:

1. Direct residual convergence before adjoint solve.
2. Adjoint residual reduction by at least four orders of magnitude.
3. Finite-difference validation for at least 20% of active design variables.
4. Consistent sign for all dominant gradient components.
5. Repeat check after major topology changes in bubble state.

### 8.7 Repeatability

Repeat identical cases with the same mesh and solver settings. For deterministic solvers, nonzero variation indicates convergence tolerance, parallel reduction order, restart sensitivity, or flow multistability. Report repeatability standard deviation for $C_D$, $C_L$, and $L_b/c$.

### 8.8 Reproducibility

Archive:

1. CST coefficients.
2. Mesh generation scripts.
3. SU2 configuration files.
4. Solver version and build settings.
5. Raw history files.
6. Surface and volume outputs.
7. R scripts and `renv.lock`.
8. Figure generation scripts.
9. Case metadata.

### 8.9 Uncertainty Propagation

For independent uncertainty sources:

$$
u_Y^2=\sum_i
\left(\frac{\partial Y}{\partial X_i}u_{X_i}\right)^2.
$$

For nonlinear outputs, use Monte Carlo:

$$
Y^{(m)}=f(X_1^{(m)},\ldots,X_n^{(m)}),
\quad m=1,\ldots,N.
$$

Report 95% intervals from empirical quantiles.

## 9. Research Risks

### 9.1 Transition-Model Limitations

Transition models are correlation-based and may not represent the tested disturbance environment. If $Tu_\infty$ or roughness is unknown, predicted transition onset can be wrong even if forces appear plausible. Mitigation: conduct turbulence-intensity sensitivity studies and validate transition location independently.

### 9.2 RANS Limitations

RANS cannot resolve three-dimensional LSB breakdown, vortex shedding details, or broadband intermittency. It models averaged effects through closure. Mitigation: use URANS or higher-fidelity comparison near stall and avoid claiming resolved instability physics from steady RANS.

### 9.3 Low-Re Numerical Instability

Low-Re separated flows can exhibit residual stagnation, force oscillation, and multiple steady states. Mitigation: use continuation and cold starts, monitor force variance, perform URANS when steady assumptions fail, and report hysteresis.

### 9.4 Adjoint Inconsistency

Adjoint gradients can be inconsistent if limiters, transition equations, turbulence model terms, or mesh deformation are not differentiated consistently. Mitigation: finite-difference checks, smooth objectives, small move limits, and rejection of inconsistent steps.

### 9.5 Optimization Overfitting

Single-point optimization may create shapes that reduce drag by exploiting one pressure-gradient state while degrading off-design stall or robustness. Mitigation: multi-point objectives and independent validation polar.

### 9.6 Geometry Ringing

High-order CST coefficients can produce oscillatory curvature. Such shapes can create artificial pressure gradients and mesh artifacts. Mitigation: curvature constraints, leading-edge radius bounds, and surface smoothness audits.

### 9.7 False Minima

Transition-state discontinuities can create false local minima. Mitigation: restart optimization from multiple initial shapes, compare with derivative-free local perturbation tests, and verify objective smoothness around the final design.

### 9.8 Mesh-Induced Artifacts

LSB length and transition onset can shift with surface spacing, first-cell height, and inflation-layer growth. Mitigation: mesh convergence for bubble metrics, not only $C_L$ and $C_D$.

### 9.9 Stall-Regime Instability

Near stall, steady RANS may converge to nonphysical or path-dependent states. Mitigation: define stall using sweep protocols, URANS statistics, and hysteresis metrics.

## 10. Implementation and Reporting Requirements

### 10.1 Minimum Reproducible Case Definition

Every case must report:

$$
Re_c,\ M_\infty,\ \alpha,\ Tu_\infty,\ c,\ \rho_\infty,\ \mu_\infty,\ T_\infty,\ p_\infty.
$$

Every geometry must report:

$$
\mathbf{a}_u,\ \mathbf{a}_l,\ t_{TE},\ r_{LE},\ t/c\ \text{at defined stations}.
$$

Every CFD result must report:

$$
C_L,\ C_D,\ C_m,\ C_{D,p},\ C_{D,f},\ x_s/c,\ x_t/c,\ x_r/c,\ L_b/c.
$$

### 10.2 Acceptance Criteria for Final Optimized Airfoil

The optimized airfoil is accepted only if all conditions are met on the fine mesh:

1. Weighted drag objective decreases by at least 3%.
2. Mean bubble length decreases by at least 10% over the design operating points.
3. Lift constraint is satisfied at every design point.
4. No thickness or curvature constraint is active with violation greater than $10^{-5}$ normalized units.
5. Mesh sensitivity satisfies $S_{mesh}(C_D)<0.02$ and $S_{mesh}(L_b)<0.05$.
6. Gradient median relative error is below 5% for checked variables.
7. Off-design stall angle does not decrease by more than $0.5^\circ$.
8. Hysteresis loop area does not increase by more than 5%.

### 10.3 Source Anchors Used for Methodology

The methodology is grounded in established work and tool documentation, including:

1. Drela's XFOIL low-Re airfoil analysis framework and its viscous-inviscid/transition modeling basis: <https://link.springer.com/chapter/10.1007/978-3-642-84010-4_1>
2. XFOIL caveats regarding small viscous features and low-Re limitations: <https://v0xnihili.github.io/xfoil-docs/caveats/>
3. Langtry-Menter $\gamma-Re_{\theta}$ transition model concept and local transition transport approach: <https://en.wikipedia.org/wiki/Gamma-Re_Transition_Model>
4. SU2 PDE-constrained optimization and multiphysics simulation documentation: <https://su2code.github.io/docs_v7/home/>
5. SU2 discrete adjoint workflow and software components: <https://su2code.github.io/docs_v7/Software-Components/>
6. SU2 convective-scheme documentation for upwind/MUSCL and turbulence scalar discretization: <https://su2code.github.io/docs_v7/Convective-Schemes/>
7. SU2 execution documentation for discrete adjoint and finite-difference workflows: <https://su2code.github.io/docs/Execution/>
8. Kulfan CST implementation description in pyGeo documentation: <https://mdolab-pygeo.readthedocs-hosted.com/en/latest/DVGeometryCST.html>
9. Svanberg's Method of Moving Asymptotes bibliographic record and formulation anchor: <https://cir.nii.ac.jp/crid/1361418519342386048>
10. Recent discussion of XFOIL limitations for LSB prediction in low-Re airfoil cases: <https://link.springer.com/article/10.1007/s10494-025-00727-7>
