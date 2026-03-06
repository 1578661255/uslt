mt5_path = "./pretrained_weight/mt5-base"

# label paths
train_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "How2Sign": "./data/How2Sign/labels.train",
                    "OpenASL": "./data/OpenASL/labels.train",
                    }

dev_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev",
                    "How2Sign": "",
                    "OpenASL": "./data/OpenASL/labels.dev",
                    }

test_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test",
                    "How2Sign": "./data/How2Sign/labels.test",
                    "OpenASL": "./data/OpenASL/labels.test",
}


# video paths
rgb_dirs = {
            "CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "How2Sign": "./dataset/How2Sign/rgb_format",
            "OpenASL": "./dataset/OpenASL/rgb_format",
            }

# pose paths
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "How2Sign": "./dataset/WLASL/pose_format",
            "OpenASL": "./dataset/WLASL/pose_format",
}

# 动作描述文件路径（Stage 3 多模态融合）
description_dirs = {
            "CSL_Daily": "./description/CSL-Daily/split_data",
}

# 预编码描述特征文件路径（BERT 编码后的特征，加速训练）
desc_feat_paths = {
            "CSL_Daily": "./script/desc_bert_features.pkl",
            "CSL_News": "./script/desc_bert_features_csl_news.pkl",
            "WLASL": "./script/desc_bert_features_wlasl.pkl",
            "How2Sign": "./script/desc_bert_features_how2sign.pkl",
            "OpenASL": "./script/desc_bert_features_openasl.pkl",
}
