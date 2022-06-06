#!/bin/bash
# Arguments:
#   - Number of parallel processes
#   - Number of CPU core per process
#   - Available GPU count
#   - script path
#   - Initial Core Index
trap terminate SIGINT
terminate(){
    pkill -SIGINT -P $$
    exit
}
for i in $(seq $1 $END)
do
    let j=$i-1
    let c_s=$j*$2+$5
    let c_e=$i*$2-1+$5
    let cuda_n=$j%$3
    # echo "$c_s $c_e", $cuda_n
    taskset -c $c_s-$c_e bash $4 $cuda_n &
    sleep 2
done
wait