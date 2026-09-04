python benchmark.py   --policy-type molmoact2   --policy-path allenai/MolmoAct2-SO100_101   --policy-from-hub   --task "Pick up the object."   --iterations 1   --policy-seconds 90   --fps 30   --eval-follower-calib-dirs calibration/robots/so101_follower   --eval-follower-ports /dev/ttyACM0   --eval-follower-ids so101_follower_arm   --eval-top-indexes 6   --eval-wrist-indexes 0   --reset-mode none   --molmoact2-enable-hardware-actions  --molmoact2-num-steps 10


python benchmark.py \
  --policy-type molmoact2 \
  --policy-path allenai/MolmoAct2-SO100_101 \
  --policy-from-hub \
  --task "Pick up the orange bread, lift the bread, and drop it on the red plate" \
  --iterations 1 \
  --policy-seconds 180 \
  --fps 30 \
  --zoom 1.0 \
  --top-width 640 \
  --top-height 480 \
  --wrist-width 640 \
  --wrist-height 480 \
  --wrist-fps 30 \
  --eval-follower-calib-dirs calibration/robots/so101_follower \
--eval-follower-ports /dev/ttyACM0 \
--eval-follower-ids so101_follower_arm \
--eval-top-indexes 6 \
--eval-wrist-indexes 0 \
  --reset-mode fixed \
  --reset-action-file arm_reset.json \
  --molmoact2-num-steps 9 \
  --molmoact2-actions-per-chunk 4 \
  --molmoact2-max-joint-step-deg 10 \
  --molmoact2-enable-hardware-actions \
  --molmoact2-enable-cuda-graph


python benchmark.py \
  --policy-type molmoact2 \
  --policy-path allenai/MolmoAct2-SO100_101 \
  --policy-from-hub \
  --task "put the bread on the red plate." \
  --iterations 1 \
  --policy-seconds 180 \
  --fps 30 \
  --zoom 1.0 \
  --top-width 640 \
  --top-height 480 \
  --wrist-width 640 \
  --wrist-height 480 \
  --wrist-fps 30 \
  --eval-follower-calib-dirs calibration/robots/so101_follower \
--eval-follower-ports /dev/ttyACM0 \
--eval-follower-ids so101_follower_arm \
--eval-top-indexes 6 \
--eval-wrist-indexes 0 \
  --reset-mode fixed \
  --reset-action-file arm_reset.json \
  --molmoact2-num-steps 9 \
  --molmoact2-actions-per-chunk 4 \
  --molmoact2-max-joint-step-deg 10 \
  --molmoact2-enable-hardware-actions \
  --molmoact2-enable-cuda-graph


this is the red plate swap successful run

python benchmark.py \
  --policy-type molmoact2 \
  --policy-path allenai/MolmoAct2-SO100_101 \
  --policy-from-hub \
  --task "Pick up the orange bread, lift the bread, and drop it on the red plate" \
  --iterations 1 \
  --policy-seconds 180 \
  --fps 30 \
  --zoom 1.0 \
  --top-width 640 \
  --top-height 480 \
  --wrist-width 640 \
  --wrist-height 480 \
  --wrist-fps 30 \
  --eval-follower-calib-dirs calibration/robots/so101_follower \
--eval-follower-ports /dev/ttyACM0 \
--eval-follower-ids so101_follower_arm \
--eval-top-indexes 6 \
--eval-wrist-indexes 0 \
  --reset-mode fixed \
  --reset-action-file arm_reset.json \
  --molmoact2-num-steps 9 \
  --molmoact2-actions-per-chunk 4 \
  --molmoact2-max-joint-step-deg 10 \
  --molmoact2-enable-hardware-actions \
  --molmoact2-enable-cuda-graph




  Fine tuning commands

  testing
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-SO100_101 \
  vlareplica \
  --max_duration=20 \
  --device_batch_size=1 \
  --global_batch_size=2 \
  --num_workers=0 \
  --pin_memory=false \
  --save_folder=/metadisk/may/molmoact2-checkpoints/smoke-2gpu \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --lora_enable=false


## MolmoAct2 VLAReplica benchmark-aligned 40K run (effective batch 16)

This uses the complete `HenryZhang/VLAReplica_SFT_data` dataset, a 32-action
chunk, 40,000 optimizer steps, and action-expert-only fine-tuning from the
SO100/SO101 checkpoint. With two GPUs and device batch 1, global batch 16 uses
gradient accumulation.

### Start training

```bash
cd ~/Desktop/Github/VLAReplica/molmoact2/experiments

GPU_IDS=0,1 \
NPROC_PER_NODE=2 \
DEVICE_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=16 \
bash run_vlareplica_benchmark.sh
```

For four GPUs:

```bash
GPU_IDS=0,1,2,3 NPROC_PER_NODE=4 DEVICE_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=16 bash run_vlareplica_benchmark.sh
```

If the new GPUs have enough memory for two examples each, set
`DEVICE_BATCH_SIZE=2`. Start with 1 and increase only after a short memory test.

### Live dashboard and raw log

Run the dashboard in a second terminal:

```bash
cd ~/Desktop/Github/VLAReplica/molmoact2/experiments
python scripts/monitor_vlareplica_training.py \
  --log /metadisk/may/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16.log \
  --save-folder /metadisk/may/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16
```

Or follow only the raw log:

```bash
tail -F /metadisk/may/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16.log
```

### Resume after an interruption

The launcher refuses to overwrite an existing run. This underlying command
resumes from the newest `stepX` checkpoint:

```bash
cd ~/Desktop/Github/VLAReplica/molmoact2/experiments
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc-per-node=2 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-SO100_101 \
  vlareplica \
  --max_duration=40000 \
  --device_batch_size=1 \
  --global_batch_size=16 \
  --num_workers=4 \
  --pin_memory=true \
  --log_interval=10 \
  --save_folder=/metadisk/may/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16 \
  --save_interval=5000 \
  --save_num_checkpoints_to_keep=2 \
  --eval_interval=-1 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --lora_enable=false \
  --action_expert_learning_rate=5e-5 \
  --save_final_optim=false \
  --save_final_unsharded_checkpoint=true 2>&1 | tee -a \
  /metadisk/may/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16.log
```

### Convert and push the finished model to Hugging Face

The run saves `step40000-unsharded` for conversion. Authenticate once and then
run the prepared conversion/upload script. The upload is private by default.

```bash
hf auth login
cd ~/Desktop/Github/VLAReplica/molmoact2/experiments
bash scripts/convert_and_push_vlareplica.sh YOUR_HF_USERNAME/MolmoAct2-VLAReplica
```

To make the new repository public:

```bash
PRIVATE=false bash scripts/convert_and_push_vlareplica.sh YOUR_HF_USERNAME/MolmoAct2-VLAReplica
```

Converted Hugging Face model output:

```text
/metadisk/may/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16/step40000-hf
```
