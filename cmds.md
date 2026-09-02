python benchmark.py   --policy-type molmoact2   --policy-path allenai/MolmoAct2-SO100_101   --policy-from-hub   --task "Pick up the object."   --iterations 1   --policy-seconds 90   --fps 30   --eval-follower-calib-dirs calibration/robots/so101_follower   --eval-follower-ports /dev/ttyACM0   --eval-follower-ids so101_follower_arm   --eval-top-indexes 6   --eval-wrist-indexes 0   --reset-mode none   --molmoact2-enable-hardware-actions  --molmoact2-num-steps 10


python benchmark.py \
  --policy-type molmoact2 \
  --policy-path allenai/MolmoAct2-SO100_101 \
  --policy-from-hub \
  --task "Pick up the orange bread and lift" \
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
  --molmoact2-max-joint-step-deg 15 \
  --molmoact2-enable-hardware-actions \
  --molmoact2-enable-cuda-graph
