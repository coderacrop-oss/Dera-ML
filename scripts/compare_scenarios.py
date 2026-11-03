import sys, io, contextlib, warnings
sys.path.insert(0, 'src')
sys.path.insert(0, '.')  # for models/ registry path resolution
import campaign_pipeline as cp
warnings.filterwarnings('ignore')

SCENARIOS = {
    'transactions' : ('data/transactions.csv', None),
    'ab_experiment': ('data/scenarios/ab_experiment.csv', 'data/scenarios/ab_experiment.csv'),
    'reactivation' : ('data/scenarios/reactivation.csv', None),
    'new_customers': ('data/scenarios/new_customers.csv', None),
    'low_season'   : ('data/scenarios/low_season.csv', None),
    'b2b_mix'      : ('data/scenarios/b2b_mix.csv', None),
}

HDR = '{:<15} {:>5} {:>10} {:>7}  {:<14} {}'.format(
    'scenario', 'คน', 'send', 'maxPSI', 'status', 'top action')
from model_registry import ModelRegistry as _MR
_champ = _MR('models').current()
_champ_label = f"Champion {_champ.version} ({_champ.provenance})" if _champ else "No champion"
print(_champ_label)
print(HDR)
print('-' * 95)

# Cross-scenario PSI thresholds (looser than production monitoring PSI_CRITICAL=0.25):
# When serving genuinely different customer populations (new vs wholesale vs reactivation)
# some distribution shift is expected and acceptable.  These thresholds reflect that.
_PSI_STABLE   = 0.5   # same population, normal variance
_PSI_MODERATE = 2.0   # different but related population, model likely still valid
# PSI >= 2.0 → DRIFT: very different population, predictions may be unreliable

for name, (csv, labels) in SCENARIOS.items():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        out = cp.run_layer4(csv, labels)
    log = buf.getvalue()
    send = out['should_send'].astype(bool)
    psi_lines = [l for l in log.splitlines() if 'max PSI=' in l]
    psi = float(psi_lines[0].split('max PSI=')[1].split(')')[0]) if psi_lines else 0.0
    if psi < _PSI_STABLE:
        status = 'stable'
    elif psi < _PSI_MODERATE:
        status = 'moderate'
    else:
        status = 'DRIFT'
    top = out[send]['recommended_action'].value_counts().index[0][:38] if send.sum() else '-'
    print('{:<15} {:>5} {:>4} ({:4.1%}) {:>7.3f}  {:<14} {}'.format(
        name, len(out), send.sum(), send.mean(), psi, status, top))
