#!/bin/bash
set -e
cd /e/evb/df
python scripts/run_df.py --dataset 10s --fusion merge  --channels video_cdn,thumbnail,channel_info --tag df_merge_vtc
python scripts/run_df.py --dataset 10s --fusion stack  --channels video_cdn --tag df_stack_v
python scripts/run_df.py --dataset 10s --fusion stack  --channels video_cdn,thumbnail,channel_info --tag df_stack_vtc
python scripts/run_df.py --dataset 10s --fusion stack  --channels video_cdn,api,channel_info,thumbnail,other --tag df_stack_all
python scripts/run_df.py --dataset 10s --fusion branch --channels video_cdn,thumbnail,channel_info --tag df_branch_vtc
python scripts/run_df.py --dataset 10s --fusion branch --channels video_cdn,api,channel_info,thumbnail,other --tag df_branch_all
