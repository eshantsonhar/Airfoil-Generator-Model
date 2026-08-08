import json

with open('aso_verification_v16_boundary_fixed/convergence_history.json', 'r') as f:
    data = json.load(f)

iterations = data['iterations']
accepted = sum(1 for it in iterations if it.get('step_accepted', False))
total = len(iterations)

print(f'Total iterations: {total}')
print(f'Accepted steps: {accepted}')
print(f'Rejection rate: {(total-accepted)/total*100:.1f}%')
print(f'Acceptance rate: {accepted/total*100:.1f}%')

print('\nCd progression:')
for it in iterations:
    print(f"Iter {it['iteration']}: Cd={it['cd']:.6f}, Cl={it['cl']:.4f}, accepted={it['step_accepted']}, t/c={it['max_thickness']:.4f}")
