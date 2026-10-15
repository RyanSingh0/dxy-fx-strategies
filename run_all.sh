#!/usr/bin/env bash
# Reproduce every number in the paper. Expect two to three hours on two cores.
set -e
cd "$(dirname "$0")/src"
echo "Part 1: the data defect"
python3 part1_data_defect/diagnose_futures.py   > ../results/part1_futures_diagnostics.txt
python3 part1_data_defect/price_swap.py         > ../results/part1_price_swap.txt
python3 part1_data_defect/exchange_battery.py   > ../results/part1_exchange_battery.txt
python3 part1_data_defect/adversarial_checks.py > ../results/part1_adversarial.txt
echo "Part 2: the undeclared search"
python3 part2_declared_search/run_battery.py    > ../results/part2_declared_search.txt
python3 part2_declared_search/ensembles.py      > ../results/part2_ensembles.txt
echo "Part 3: the exhaustive search"
python3 part3_exhaustive/validate_harness.py    > ../results/part3_harness_validation.txt
python3 part3_exhaustive/run_search.py          > ../results/part3_search.txt
python3 part3_exhaustive/verify_search.py       > ../results/part3_verification.txt
echo "Part 4: strategies built from the structure"
python3 part4_reasoned/analyse_structure.py     > ../results/part4_structure.txt
python3 part4_reasoned/carry_futures.py         > ../results/part4_carry_futures.txt
python3 part4_reasoned/carry_dispersion.py      > ../results/part4_carry_dispersion.txt
python3 part4_reasoned/carry_mechanism.py       > ../results/part4_carry_mechanism.txt
python3 part4_reasoned/carry_long_sample.py     > ../results/part4_carry_long_sample.txt
python3 part4_reasoned/factors_long_sample.py   > ../results/part4_factors_long_sample.txt
python3 part4_reasoned/volatility_targeting.py  > ../results/part4_volatility_targeting.txt
echo "Revision analyses"
mkdir -p ../results/revision
python3 revision/fast_search.py                 > ../results/revision/fast_search.txt
python3 revision/run_ceiling.py boot 2000 21
python3 revision/run_ceiling.py gauss 1000 21
for L in 5 63 126; do python3 revision/run_ceiling.py boot 500 $L; done
python3 revision/summarise_ceiling.py           > ../results/revision/ceiling_summary.txt
python3 revision/naive_stats.py                 > ../results/revision/naive_stats.txt
python3 revision/screen_study.py                > ../results/revision/screen_study.txt
python3 revision/mechanism.py                   > ../results/revision/mechanism.txt
python3 revision/carry_instability.py           > ../results/revision/carry_instability.txt
python3 revision/part2_spot_tr.py               > ../results/revision/part2_spot_tr.txt
python3 revision/nonlinear_spreads.py           > ../results/revision/nonlinear_spreads.txt
python3 revision/text_numbers.py                > ../results/revision/text_numbers.txt
python3 revision/naive_tables.py                > ../results/revision/naive_tables.txt
python3 revision/npwhere_effect.py              > ../results/revision/npwhere_effect.txt
python3 revision/screen_dependence.py           > ../results/revision/screen_dependence.txt
python3 revision/figures_naive.py ../results/revision/figures
python3 revision/make_figures.py ../results/revision/figures
python3 -m pip freeze | grep -iE "^(numpy|pandas|scipy|statsmodels|matplotlib)==" > ../results/revision/environment.txt
python3 --version >> ../results/revision/environment.txt
echo "Done. All output is in results/"
