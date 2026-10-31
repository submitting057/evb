#!/bin/bash
set -e
cd /e/evb/df
python scripts/run_df.py --dataset 10s --fusion branch --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --dt --tag df_branch_all_dt
python scripts/run_df.py --dataset 5s --fusion branch --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --tag df5_branch_all_e100
