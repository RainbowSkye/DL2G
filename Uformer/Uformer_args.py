import os, sys
import argparse

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))

parser = argparse.ArgumentParser(description='Image denoising evaluation on SIDD')

parser.add_argument('--input_dir', default='../dataset/highlight/Face_Gla_Spec/Val/Specular',
                    type=str, help='Directory of test images')
parser.add_argument('--mask_dir', default=None,
                    type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='../dataset/highlight/Face_Gla_Spec/Val/Degra',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='../logs/denoising/highlight/Uformer_Btorch1.8.0_11/models/model_best.pth',
                    type=str, help='Path to weights')

parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')

U_args = parser.parse_args()
