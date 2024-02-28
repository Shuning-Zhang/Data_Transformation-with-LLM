#!/bin/bash

# for foofah dataset experiment 2-60

# for i in $(seq 2 51); do
#     for j in $(seq 1 5); do
#         echo "------------------------"
#         echo "experiment progress: ${i}_${j}"
#         echo "------------------------"
#         python foofah.py --input /tests/data/foofah/exp0_${i}_${j}.txt
#     done
# done

# for i in $(seq 1 5); do
#     echo "------------------------"
#     echo "experiment progress: ${i}"
#     echo "------------------------"
#     python foofah.py --input /tests/data/foofah/exp0_agriculture_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_craigslist_data_wrangler_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_crime_data_wrangler_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_potters_wheel_divide_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_crime_data_wrangler_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_potters_wheel_fold_2_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_potters_wheel_merge_split_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_potters_wheel_split_fold_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_potters_wheel_unfold_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_potters_wheel_unfold2_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_proactive_wrangling_complex_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_proactive_wrangling_fold_${i}.txt
#     python foofah.py --input /tests/data/foofah/exp0_reshape_table_structure_data_wrangler_${i}.txt
    
# done
# More shell script commands if needed

python foofah.py --input /tests/data/Transformation.Text/Address.000002.txt