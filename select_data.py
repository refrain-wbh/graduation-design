import numpy as np
import pandas as pd
import settings
from pathlib import Path

import time
from loguru import logger

cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time() + 8 * 60 * 60))
file_name = Path(__file__).name.split(".")[0]
log_file = f"log/{file_name}/{cur_time}.log"
logger.add(
    log_file,
    rotation="100 MB",
    diagnose=True,
    backtrace=True,
    enqueue=True,
    retention="10 days",
    compression="zip",
)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


@logger.catch
def get_df_from_statistics_file(statistics_file_path):
    robust_statistics_list = np.load(statistics_file_path, allow_pickle=True).item()
    if isinstance(robust_statistics_list, dict):
        robust_statistics_list = sorted(
            robust_statistics_list.items(), key=lambda x: x[0]
        )
        robust_statistics_list = [item[1] for item in robust_statistics_list]
    # 一个list中是不同interval所采集的样本
    keys1 = robust_statistics_list[0].keys()  # idx
    keys2 = list(robust_statistics_list[0].values())[0].keys()  # fields
    robust_statistics = [
        {
            **{"idx": key1},
            **{
                key2: [item[key1][key2] for item in robust_statistics_list]
                for key2 in keys2
            },
        }
        for key1 in keys1
    ]
    # 统计列表，每个元素中包含了idx字段

    df = pd.DataFrame(robust_statistics)
    logger.info(f"{df.head()}")

    return df


def sort_df_by_metric(df, select_metric):
    if "+" in select_metric:
        # metric的中文：指标
        mertic_x, mertic_y = select_metric.split("+")
        x_destination = 0 if mertic_x.endswith("_r") else 1
        y_destination = 0 if mertic_y.endswith("_r") else 1
        mertic_x, mertic_y = mertic_x.removesuffix("_r"), mertic_y.removesuffix("_r")

        df["sort_condition"] = pow(df[mertic_x] - x_destination, 2) + pow(
            df[mertic_y] - y_destination, 2
        )
        df = df.sort_values(by="sort_condition")
    else:
        if select_metric.endswith("_r"):  # 从小到大
            df = df.sort_values(by=select_metric, ascending=True)
        else:  # 从大到小
            df = df.sort_values(by=select_metric, ascending=False)
    return df


def select_data(df, select_metric, select_ratio, do_balance_labels, num_labels):
    df = sort_df_by_metric(df, select_metric)
    total_chose = int(df.shape[0] * select_ratio)

    selected_label_nums = {i: 0 for i in range(num_labels)}
    select_data_idx = []
    if do_balance_labels:
        each_label_chose = total_chose // num_labels
        for _, idx, label in df[["idx", "label"]].itertuples():
            if selected_label_nums[label] < each_label_chose:
                selected_label_nums[label] += 1
                select_data_idx.append(idx)
            if all(
                [selected_label_nums[i] >= each_label_chose for i in range(num_labels)]
            ):
                break
    else:
        for each in df["label"].iloc[:total_chose]:
            selected_label_nums[each] += 1
        select_data_idx = df["idx"].to_list()[:total_chose]

    return select_data_idx, selected_label_nums


def add_field(df):
    # golden label is a list 但是有相同的值
    df["label"] = df["golden_label"].apply(lambda x: x[0])

    df["Flip Rate"] = df["after_perturb_pred"].apply(lambda x: sum(x))
    df["Flip Rate"] = normalization(df["Flip Rate"])


def generate_data_indices(select_setting, df, num_labels):
    add_field(df)

    if select_setting.select_metric == "random":
        result_data_indices = np.random.choice(
            df.shape[0], size=df.shape[0] * select_setting.select_ratio, replace=False
        )
        selected_label_nums = {i: 0 for i in range(num_labels)}
        for idx in result_data_indices:
            selected_label_nums[df.iloc[idx]["label"]] += 1
        return result_data_indices, selected_label_nums

    if select_setting.select_from_two_end:
        select_metric = select_setting.select_metric
        select_metric_r = (
            select_metric[:-2] if select_metric[-2:] == "_r" else select_metric + "_r"
        )
        idx1, nums1 = select_data(
            df,
            select_metric,
            select_setting.select_ratio * (1 - select_setting.ratio_proportion),
            select_setting.do_balance_labels,
            num_labels,
        )
        idx2, nums2 = select_data(
            df,
            select_metric_r,
            select_setting.select_ratio * (select_setting.ratio_proportion),
            select_setting.do_balance_labels,
            num_labels,
        )
        result_data_indices = idx1 + idx2
        selected_label_nums = {
            i: nums1.get(i, 0) + nums2.get(i, 0) for i in range(num_labels)
        }
    else:
        result_data_indices, selected_label_nums = select_data(
            df,
            select_setting.select_metric,
            select_setting.select_ratio,
            select_setting.do_balance_labels,
            num_labels,
        )
    return result_data_indices, selected_label_nums


import plot


@logger.catch
def draw_picture1():
    static_path = "statistics/npys/2023-04-12-13:23:52.npy"
    df = get_df_from_statistics_file(static_path)
    df["original_correctness"] = df["original_pred"].apply(lambda x: sum(x))
    list_sub = lambda x, y: [x_i - y_i for x_i, y_i in zip(x, y)]
    df["loss_diff"] = df[["original_loss", "after_perturb_loss"]].apply(
        lambda x: list_sub(x[1], x[0]), axis=1
    )
    df["loss_diff_mean"] = df["loss_diff"].apply(lambda x: np.mean(x))
    df["loss_diff_std"] = df["loss_diff"].apply(lambda x: np.std(x))
    df["Sensitivity"] = df["after_perturb_logit"].apply(lambda x: np.mean(x))
    df["Variability"] = df["after_perturb_logit"].apply(lambda x: np.std(x))

    df["Sensitivity"] = normalization(df["Sensitivity"])
    df["Variability"] = normalization(df["Variability"])

    df["Flip Rate"] = df["after_perturb_pred"].apply(lambda x: sum(x))
    plot.plot_map(
        df,
        df.shape[0],
        hue_metric="Flip Rate",
        main_metric="Sensitivity",
        other_metric="Variability",
    )


if __name__ == "__main__":
    draw_picture1()
