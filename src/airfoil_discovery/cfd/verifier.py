from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airfoil_discovery.cfd.extractor import SurfaceDistributions


@dataclass(slots=True)
class VerificationResult:
    lsb_detected: bool
    transition_inconsistent: bool
    unrealistic_early_transition: bool
    fully_laminar: bool
    physics_violation_penalty: float
    flags: list[str]


class TransitionVerifier:
    def verify(self, dist: SurfaceDistributions, reynolds: float) -> VerificationResult:
        dcp_dx = np.gradient(dist.upper_cp, dist.upper_x)
        plateau = np.abs(dcp_dx) < 0.5
        lsb_detected = False
        for start in np.flatnonzero(plateau):
            end = start
            while end + 1 < len(plateau) and plateau[end + 1]:
                end += 1
            if dist.upper_x[end] - dist.upper_x[start] >= 0.02:
                recovery = dcp_dx[end + 1 :] > 1.0
                if np.any(recovery):
                    first = np.flatnonzero(recovery)[0] + end + 1
                    if dist.upper_x[first] - dist.upper_x[end] >= 0.01 or first == end + 1:
                        lsb_detected = True
                        break
        transition_inconsistent = False
        if lsb_detected and None not in (dist.x_sep, dist.x_tr, dist.x_reat):
            transition_inconsistent = not (dist.x_sep < dist.x_tr < dist.x_reat)
        unrealistic_early_transition = bool(dist.x_tr is not None and dist.x_tr < 0.02 and reynolds < 30000.0)
        fully_laminar = dist.x_tr is None
        flags: list[str] = []
        if transition_inconsistent:
            flags.append("transition_inconsistent")
        if unrealistic_early_transition:
            flags.append("unrealistic_early_transition")
        if fully_laminar:
            flags.append("fully_laminar")
        return VerificationResult(
            lsb_detected=lsb_detected,
            transition_inconsistent=transition_inconsistent,
            unrealistic_early_transition=unrealistic_early_transition,
            fully_laminar=fully_laminar,
            physics_violation_penalty=float(sum([transition_inconsistent, unrealistic_early_transition])),
            flags=flags,
        )
