import argparse
import os

parser = argparse.ArgumentParser(description='Scene Sketch Text')

# ----------------------------
# Dataloader Options
# ----------------------------

# For Our Dataset:
# ------------------

parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data',
	help='Enter root directory of OurScene Dataset')
parser.add_argument('--max_len', type=int, default=224, help='Max Edge length of images')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--workers', type=int, default=12, help='Num of workers in dataloader')
parser.add_argument('--log_online', action='store_true',
					help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally '
						 'be set.')
parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')
parser.add_argument('--project', default='Sample_Project', type=str,
					help='Name of the project - relates to W&B project names. In --savename default setting part of '
						 'the savename.')
parser.add_argument('--group', default='', type=str, help='Name of the group - relates to W&B group names - all runs '
														  'with same setup but different seeds are logged into one '
														  'group. In --savename default setting part of the savename. '
														  'Name is created as model_dataset_group')
parser.add_argument('--savename', default='group_plus_seed', type=str, help='Run savename - if default, the savename'
																			' will comprise the project and group name')
parser.add_argument('--path_aux', type=str, default=os.getcwd())
opts = parser.parse_args()
