#!/usr/bin/env bash
# Print the interpreters the recipes will use and check their key imports.
IA=${IA_PYTHON:-python}; HOD=${HOD_PYTHON:-python}
echo "IA_PYTHON  = $IA";  $IA  -c "import ia2pt, pyccl, fastpt, iminuit, nautilus, treecorr, pycorr, dsigma, healpy, pyarrow; print('  IA stack OK  (ia2pt', ia2pt.__version__ if hasattr(ia2pt,'__version__') else '', ')')" || echo "  IA stack: MISSING packages"
$IA -c "import cucount, lsstypes" 2>/dev/null && echo "  cucount + lsstypes present (GPU multipoles)" || echo "  cucount/lsstypes absent: measurement stage unavailable (fits and figures unaffected)"
echo "HOD_PYTHON = $HOD"; $HOD -c "from HOD_NRV.HOD_analytical.sampler import CSMFFitter; import jax, dark_emulator; print('  HOD stack OK')" || echo "  HOD stack: MISSING packages (analyses/csmf unavailable)"
