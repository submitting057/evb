#!/bin/bash
set -e
cd /e/evb/df
python scripts/run_df.py --dataset 10s --fusion branch --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --tag df_branch_all_e100
python scripts/run_df.py --dataset 10s --fusion branch --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --side seg --tag df_branch_all_seg
python scripts/run_df.py --dataset 10s --fusion branch --channels video_cdn,api,channel_info,thumbnail,other --epochs 100 --dt --tag df_branch_all_dt
