for seed in 1234 2341 3412 4123; do
    python examples/sac_gaussian.py --env $1 --seed $seed --num_layer 2 --prob $2 --std $3
done
