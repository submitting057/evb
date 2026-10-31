#!/bin/bash
set -e
cd /e/evb/df
python scripts/run_df.py --dataset 10s --fusion branch --channels video_cdn,thumbnail,channel_info --tag df_branch_vtc
python scripts/run_df.py --dataset 10s --fusion branch --channels video_cdn,api,channel_info,thumbnail,other --tag df_branch_all
python scripts/run_df.py --dataset 10s --fusion merge --channels video_cdn,thumbnail,channel_info --epochs 100 --tag df_merge_vtc100
python scripts/run_df.py --dataset 10s --fusion merge --channels video_cdn,thumbnail,channel_info --chan-emb --epochs 100 --tag df_mce_vtc
python scripts/run_df.py --dataset 10s --fusion merge --channels video_cdn,api,channel_info,thumbnail,other --chan-emb --epochs 100 --tag df_mce_all
