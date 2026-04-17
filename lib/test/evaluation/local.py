from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/apulis-dev/code/VIPT_gai/data/got10k_lmdb'
    settings.got10k_path = '/home/apulis-dev/code/VIPT_gai/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/apulis-dev/code/VIPT_gai/data/itb'
    settings.lasot_extension_subset_path_path = '/home/apulis-dev/code/VIPT_gai/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/apulis-dev/code/VIPT_gai/data/lasot_lmdb'
    settings.lasot_path = '/home/apulis-dev/code/VIPT_gai/data/lasot'

    settings.lasher_path = '/home/apulis-dev/code/VIPT_gai/data/lasher'  # 替换为你的实际绝对路径
    settings.network_path = '/home/apulis-dev/code/VIPT_gai/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/apulis-dev/code/VIPT_gai/data/nfs'
    settings.otb_path = '/home/apulis-dev/code/VIPT_gai/data/otb'
    settings.prj_dir = '/home/apulis-dev/code/VIPT_gai'
    settings.result_plot_path = '/home/apulis-dev/code/VIPT_gai/output/test/result_plots'
    settings.results_path = '/home/apulis-dev/code/VIPT_gai/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/apulis-dev/code/VIPT_gai/output'
    settings.segmentation_path = '/home/apulis-dev/code/VIPT_gai/output/test/segmentation_results'
    settings.tc128_path = '/home/apulis-dev/code/VIPT_gai/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/apulis-dev/code/VIPT_gai/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/apulis-dev/code/VIPT_gai/data/trackingnet'
    settings.uav_path = '/home/apulis-dev/code/VIPT_gai/data/uav'
    settings.vot18_path = '/home/apulis-dev/code/VIPT_gai/data/vot2018'
    settings.vot22_path = '/home/apulis-dev/code/VIPT_gai/data/vot2022'
    settings.vot_path = '/home/apulis-dev/code/VIPT_gai/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.results_path = '/home/apulis-dev/code/VIPT_gai/output/test/tracking_results'
    return settings
