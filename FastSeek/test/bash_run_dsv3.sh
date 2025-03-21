# nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
#  -f true  -o /app/sglang_workspace/profile_log/sglang043_sh_64bs_256i_10o_nsys \
#  python3 -m sglang.bench_one_batch --model /app/models/DeepSeek-R1-Zero \
#  --tensor-parallel-size 8 \
#  --context-length 266 --cuda-graph-bs 64 \
#  --mem-fraction-static 0.88 \
#  --max-running-requests 64 --max-total-tokens 17024 \
#  --batch-size 64 --input-len 256 --output-len 10 \
#  --trust-remote-code --enable-flashinfer-mla --disable-radix-cache > /app/sglang_workspace/profile_log/run_log.txt # 2>&1 



python3 run_dsv3.py --model /home/private_xgpt_repo/FastSeek/models/model_fp8_4layer.pt \
 --tensor-parallel-size 8 \
 --context-length 256 --cuda-graph-bs 4 \
 --mem-fraction-static 0.90 \
 --max-running-requests 4 --max-total-tokens 2048 \
 --batch-size 4 --input-len 128 --output-len 128 \
 --load-format pt \
 --trust-remote-code --enable-flashinfer-mla --disable-radix-cache > /home/private_xgpt_repo/FastSeek/log/run_log.txt 2>&1 