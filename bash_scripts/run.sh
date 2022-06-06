# Arguments:
#   - GPU number
#   - Environment name
study_name=benchmark
for envname in SpaceInvaders Pong Seaquest Freeway Breakout
do
    for repo in "mb" "sb3"
    do
        python -m runs.${repo}.a2c.run \
            --seed 120213 \
            --device cuda:$1 \
            --log-dir ./logs/benchmark \
            --env-name ${envname}NoFrameskip-v4 \
            --save-model \
            --start-tune-with-default-params \
            --n-seeds 3 \
            --max-trials 3 \
            --tune \
            --study-name ${repo}-${envname}-${study_name} \
            --continue-study \
            --storage-url redis://172.30.1.56:6910
    done
done