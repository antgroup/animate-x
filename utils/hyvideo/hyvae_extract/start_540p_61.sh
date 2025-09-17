export PYTHONPATH=${PYTHONPATH}:`pwd`
for ((i=0;i<$HOST_GPU_NUM;++i)); do
    CUDA_VISIBLE_DEVICES=$i python3 -u hyvideo/hyvae_extract/run.py --local_rank $i --config 'hyvideo/hyvae_extract/vae_540p_61.yaml'&
done
# CUDA_VISIBLE_DEVICES=0 python3 -u hyvideo/hyvae_extract/run.py --local_rank 0 --config 'hyvideo/hyvae_extract/vae_540p_61.yaml'&
wait

echo "Finished."

# export HOST_GPU_NUM=8 
# cd /HunyuanVideo-I2V
# bash hyvideo/hyvae_extract/start_540p_61.sh