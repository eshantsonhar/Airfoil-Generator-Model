"""
Automated wording checks for CFD-only research language.

Ensures scientifically cautious language is used in reports.
Prevents implying experimental validation exists or fabricating
experimental agreement.
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WordingViolation:
    """Record of a wording violation."""
    
    violation_type: str
    forbidden_phrase: str
    suggested_replacement: str
    context: str
    line_number: Optional[int] = None


class WordingChecker:
    """
    Checks for inappropriate language in CFD-only research.
    
    Enforces use of scientifically cautious language:
    - "predicted" instead of "proved"
    - "simulated" instead of "confirmed"
    - "within the employed transitional RANS framework" instead of "experimentally verified"
    - "computationally observed" instead of "observed in reality"
    
    NEVER allows:
    - "proved"
    - "confirmed"
    - "experimentally verified"
    - "observed in reality"
    """
    
    # Forbidden phrases and their replacements
    FORBIDDEN_PHRASES = {
        "proved": {
            "replacement": "predicted",
            "context": "CFD-only research cannot prove physical phenomena",
        },
        "confirmed": {
            "replacement": "predicted",
            "context": "CFD-only research cannot confirm without experimental validation",
        },
        "experimentally verified": {
            "replacement": "computationally predicted",
            "context": "No experimental validation in this CFD-only study",
        },
        "observed in reality": {
            "replacement": "computationally observed",
            "context": "CFD-only research cannot observe reality",
        },
        "validated by experiment": {
            "replacement": "validated against literature benchmarks",
            "context": "CFD-only research uses literature validation, not experiments",
        },
        "matches experimental data": {
            "replacement": "agrees with literature benchmarks",
            "context": "CFD-only research compares to literature, not experiments",
        },
        "experimental agreement": {
            "replacement": "literature agreement",
            "context": "CFD-only research has no experimental data",
        },
        "real-world": {
            "replacement": "computational",
            "context": "CFD-only research is computational, not real-world",
        },
        "actual": {
            "replacement": "simulated",
            "context": "CFD-only research is simulated, not actual",
        },
        "true": {
            "replacement": "predicted",
            "context": "CFD-only research predictions, not truth",
        },
    }
    
    # Recommended cautious phrases
    RECOMMENDED_PHRASES = [
        "predicted",
        "simulated",
        "computationally observed",
        "within the employed transitional RANS framework",
        "numerically predicted",
        "CFD-predicted",
        "computationally estimated",
        "within the numerical framework",
        "according to the employed model",
    ]
    
    def __init__(self):
        """Initialize wording checker."""
        self.violations: List[WordingViolation] = []
    
    def check_text(self, text: str, filename: Optional[str] = None) -> List[WordingViolation]:
        """
        Check text for forbidden phrases.
        
        Args:
            text: Text to check
            filename: Optional filename for context
        
        Returns:
            List of wording violations
        """
        self.violations = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for forbidden, info in self.FORBIDDEN_PHRASES.items():
                # Case-insensitive search
                pattern = re.compile(re.escape(forbidden), re.IGNORECASE)
                matches = pattern.finditer(line)
                
                for match in matches:
                    violation = WordingViolation(
                        violation_type="forbidden_phrase",
                        forbidden_phrase=match.group(),
                        suggested_replacement=info["replacement"],
                        context=info["context"],
                        line_number=line_num if filename else None,
                    )
                    self.violations.append(violation)
        
        return self.violations
    
    def check_file(self, filepath: str) -> List[WordingViolation]:
        """
        Check a file for forbidden phrases.
        
        Args:
            filepath: Path to file to check
        
        Returns:
            List of wording violations
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.check_text(text, filename=filepath)
    
    def generate_report(self) -> str:
        """
        Generate a report of wording violations.
        
        Returns:
            Report string
        """
        if not self.violations:
            return "No wording violations found. Language is scientifically cautious."
        
        report = f"Found {len(self.violations)} wording violation(s):\n\n"
        
        for i, violation in enumerate(self.violations, 1):
            report += f"{i}. {violation.violation_type.upper()}\n"
            report += f"   Forbidden: '{violation.forbidden_phrase}'\n"
            report += f"   Suggested: '{violation.suggested_replacement}'\n"
            report += f"   Context: {violation.context}\n"
            if violation.line_number:
                report += f"   Line: {violation.line_number}\n"
            report += "\n"
        
        report += "\nRecommended cautious phrases:\n"
        for phrase in self.RECOMMENDED_PHRASES:
            report += f"  - {phrase}\n"
        
        return report
    
    def get_violation_count(self) -> int:
        """Get number of violations."""
        return len(self.violations)
    
    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0
    
    def reset(self):
        """Reset violations."""
        self.violations.clear()


def check_report_text(text: str) -> Tuple[bool, str]:
    """
    Check report text for CFD-only language compliance.
    
    Args:
        text: Report text to check
    
    Returns:
        (is_compliant, report)
    """
    checker = WordingChecker()
    violations = checker.check_text(text)
    
    if violations:
        return False, checker.generate_report()
    
    return True, "Report language is scientifically cautious and CFD-only compliant."
