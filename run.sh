gpu=0
lr=1e-3
bs=20480
epochs=350
for full in 0 
do
	for factors in 1000
	do
output=output-bs-$bs-factors-$factors-epochs-$epochs-full-$full-$lr.txt
CUDA_VISIBLE_DEVICES=$gpu python ./train.py --factors $factors --lr $lr --bs $bs --epochs $epochs --full $full > $output
done
done
