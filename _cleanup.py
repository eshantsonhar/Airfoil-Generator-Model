"""Clear debug file for convergence check"""
import sys
sys.path.insert(0,'.')
from pathlib import Path

lines = Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_rel_check.py').read_text().splitlines()
lines2 = Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_check_abs.py').read_text().splitlines()
lines3 = Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_sweep_thresholds.py').read_text().splitlines()
lines4 = Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_verify_fix.py').read_text().splitlines()
lines5 = Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_all_combos.py').read_text().splitlines()
lines6 = Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_check_conv.py').read_text().splitlines()
lines7 = Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_test_sw.py').read_text().splitlines()
lines8 = Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_test_osc.py').read_text().splitlines()

Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_rel_check.py').write_text('')
Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_check_abs.py').write_text('')
Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_sweep_thresholds.py').write_text('')
Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_verify_fix.py').write_text('')
Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_all_combos.py').write_text('')
Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_check_conv.py').write_text('')
Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_test_sw.py').write_text('')
Path('C:/Eshant_Sonhar/airfoil research paper/airfoil generator model/_test_osc.py').write_text('')
print('Done')
