#!/usr/bin/env bash
# Reproduce every number in the paper. Expect roughly 60 to 90 minutes in total.
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
echo "Done. All output is in results/"
