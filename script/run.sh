#!/bin/bash

echo "Running CCML"
cd ./../code/

batch_size=32
epoch=1
arch=SCNN
dataset_path=../../../datasets/ireland12
channel=ALL
label=BEN-12
sigma=10000.0
swap=1
swap_rate=.75
lambda2=.25
lambda3=.50
flip_bound=.9
flip_per=0.01
miss_alpha=1.0
extra_beta=1.0
add_noise=1
noise_type=1
sample_rate=0.1
class_rate=0.1
metric=mmd
alpha=0.
test_=1

python main.py -b $batch_size -e $epoch -a $arch -d $dataset_path -ch $channel -lb $label -si $sigma -sw $swap -sr $swap_rate -lto $lambda2 -ltr $lambda3 -fb $flip_bound -fp $flip_per -ma $miss_alpha -eb $extra_beta -an $add_noise -nty $noise_type -sar $sample_rate -car $class_rate -dm $metric -alp $alpha -test $test_
