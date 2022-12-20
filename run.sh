

module load cuda11.1/blas/11.1.1
module load cuda11.1/fft/11.1.1
module load cuda11.1/nsight/11.1.1
module load cuda11.1/profiler/11.1.1
module load cuda11.1/toolkit/11.1.1


srun -p hgx2q -N 1 -n 8 --gres=gpu:1 --pty /bin/bash --login
ssh -t g002  'nvtop'



cd ~/D1/SP_guided_Noisy_Label_Seg/
source venv/bin/activate

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


python main.py --cfg  'exp/default.yaml'  --id default --parallel
python main.py --cfg 'exp/default.yaml' --id default --demo 'val' --weight_path './checkpoints/default/model_up_epoch_10.pth' --parallel --save_preds True


python main.py --cfg 'exp/ISIC_NoNoisyLabel.yaml' --id ISIC_NoNoisyLabel  --parallel


python main.py --cfg 'exp/KvasirSeg_NoNoisyLabel.yaml' --id KvasirSeg_NoNoisyLabel  --parallel
