"""
VIPT 标准评估脚本 - 使用 lib.test.evaluation 框架计算 SR/PR/AUC
输出格式兼容 OTB/LaSOT 等标准跟踪基准的评估结果，全量测试评估

使用方法:
    python for_test/analysis_results.py --dataset_name lasher --tracker_param exp1_baseline_standard
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
import argparse


def main():
    parser = argparse.ArgumentParser(description='VIPT Standard Evaluation (OTB-style)')
    parser.add_argument('--tracker_name', default='vipt', help='Tracker name (default: vipt)')
    parser.add_argument('--tracker_param', type=str, required=True,
                        help='Config/experiment name, e.g. exp1_baseline_standard, vit_base')
    parser.add_argument('--dataset_name', type=str, default='lasher',
                        help='Dataset name: lasher, vtuav_st, vtuav_lt, etc.')
    parser.add_argument('--runid', type=int, default=None, help='Run ID for multi-run evaluation')
    parser.add_argument('--plot', action='store_true', help='Also generate plots')
    args = parser.parse_args()

    trackers = []
    dataset_name = args.dataset_name

    trackers.extend(trackerlist(
        name=args.tracker_name,
        parameter_name=args.tracker_param,
        dataset_name=dataset_name,
        run_ids=args.runid,
        display_name=f"VIPT-{args.tracker_param}",
        result_only=True
    ))

    dataset = get_dataset(dataset_name)

    print("\n" + "=" * 60)
    print(f"  VIPT Evaluation: {args.tracker_param} on {dataset_name}")
    print("=" * 60)

    print_results(
        trackers,
        dataset,
        dataset_name,
        merge_results=True,
        plot_types=('success', 'norm_prec', 'prec')
    )

    if args.plot:
        plot_results(
            trackers,
            dataset,
            dataset_name,
            merge_results=True,
            plot_types=('success', 'prec'),
            plot_bin_gap=0.05
        )
        plot_path = os.path.join(project_root, 'for_test', f'eval_plot_{args.tracker_param}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")


if __name__ == '__main__':
    main()
