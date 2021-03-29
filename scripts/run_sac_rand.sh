for seed in 1234 2341 3412 4123; do
    python examples/sac_random_fc.py --env $1 --seed $seed --num_layer 2 --single_flag $2  --equal_flag $3  --lower $4 --upper $5
done