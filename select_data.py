
import numpy as np
import pandas as pd
import settings
logger = settings.get_logger(__file__)

@logger.catch
def get_df_from_statistics_file(statistics_file_path):
    robust_statistics_list,config = np.load(statistics_file_path, allow_pickle=True)
    if isinstance(robust_statistics_list, dict):
        robust_statistics_list = sorted(robust_statistics_list.items(), key=lambda x: x[0])
        robust_statistics_list = [item[1] for item in robust_statistics_list]
    # robust_statistics_list is list of dict of dict
    #list表示不同的interval，dict是idx到统计数据的映射，dict是统计数据到值的映射
    # 需要首先产生一个dict，key是idx，value是一个dict，key是统计数据，value是一个列表，列表的长度是interval的数量
    # list of dict of dict to dict of list of dict 
    keys1 = robust_statistics_list[0].keys()
    keys2 = list(robust_statistics_list[0].values())[0].keys()
    robust_statistics = [{'idx':key1}|{key2: [item[key1][key2] for item in robust_statistics_list] for key2 in keys2} for key1 in keys1]
    # 统计列表，每个元素中包含了idx字段
    
    df = pd.DataFrame(robust_statistics)
    logger.info(f"{df.head()}") 
       
    return df
def sort_df_by_metric(df, select_metric):
    if "+" in select_metric:
        # metric的中文：指标
        mertic_x,mertic_y = select_metric.split("+")
        x_destination = 0 if mertic_x.endswith("_r") else 1
        y_destination = 0 if mertic_y.endswith("_r") else 1
        mertic_x,mertic_y = mertic_x.removesuffix("_r"),mertic_y.removesuffix("_r")
        
        df["sort_condition"] = pow(df[mertic_x] - x_destination,2) + pow(df[mertic_y] - y_destination,2)
        df = df.sort_values(by="sort_condition")
    else:  # 从大到小
        df = df.sort_values(by=select_metric, ascending=False)
    return df

def select_data(df,select_metric,select_ratio,do_balance_labels, num_labels):
    df = sort_df_by_metric(df, select_metric)
    total_chose = int(df.shape[0] * select_ratio)
    
    selected_label_nums = {i:0  for i in range(num_labels)}
    select_data_idx = []
    if do_balance_labels:
        each_label_chose = total_chose // num_labels
        for idx,label in df[["idx","label"]].to_list():
            if selected_label_nums[label] < each_label_chose:
                selected_label_nums[label] += 1
                select_data_idx.append(idx)
            if all([selected_label_nums[i] >= each_label_chose for i in range(num_labels)]):
                break
    else :
        for each in df['label'].to_list()[:total_chose]:
            selected_label_nums[each] += 1
        select_data_idx = df['idx'].to_list()[:total_chose]
    
    return select_data_idx,selected_label_nums


def generate_data_indices(statistics_setting, df,num_labels):
    if statistics_setting.select_metric == "random":
        result_data_indices = np.random.choice(df.shape[0], statistics_setting.select_ratio, replace=False)
        selected_label_nums = {i:0  for i in range(num_labels)}
        for idx in result_data_indices:
            selected_label_nums[df.iloc[idx]["label"]] += 1
        return result_data_indices, selected_label_nums
    
    if statistics_setting.select_from_two_end:
        select_metric = statistics_setting.select_metric
        select_metric_r = select_metric[:-2] if select_metric[-2:] == "_r" else select_metric + "_r"
        idx1,nums1 = select_data(df,
                                 select_metric,
                                 statistics_setting.select_ratio*(1 - statistics_setting.ratio_proportion),
                                 statistics_setting.do_balance_labels, 
                                 num_labels)
        idx2,nums2 = select_data(df,
                                 select_metric_r,
                                 statistics_setting.select_ratio*(statistics_setting.ratio_proportion),
                                 statistics_setting.do_balance_labels, 
                                 num_labels)
        result_data_indices = idx1 + idx2
        selected_label_nums = {i:nums1.get(i,0)+nums2.get(i,0)  for i in range(num_labels)}
    else :
        result_data_indices,selected_label_nums = select_data(df,
                                                             statistics_setting.select_metric,
                                                             statistics_setting.select_ratio,
                                                             statistics_setting.do_balance_labels, 
                                                             num_labels)
    return result_data_indices, selected_label_nums

import plot
@logger.catch
def main():
    static_path = "statistics/robust_statistics_dict_2023-04-11-01-13-33.npy"
    df = get_df_from_statistics_file(static_path)
    df['original_correctness'] = df['original_pred'].apply(lambda x: sum(x))
    list_sub = lambda x,y: [x_i-y_i for x_i,y_i in zip(x,y)]
    df['loss_diff'] = df[['original_loss','after_perturb_loss']].apply(lambda x: list_sub(x[1],x[0]),axis=1)
    df['loss_diff_mean'] = df['loss_diff'].apply(lambda x: np.mean(x))
    df['loss_diff_std'] = df['loss_diff'].apply(lambda x: np.std(x))
    df['Sensitivity'] = df['original_loss'].apply(lambda x: np.mean(x))
    df['Variability'] = df['original_loss'].apply(lambda x: np.std(x))
    df['Flip Rate'] = df['original_pred'].apply(lambda x: sum(x))
    plot.plot_map(df,df.shape[0],hue_metric="original_correctness",main_metric="Sensitivity",other_metric="Variability")
    
if __name__ == '__main__':
    main()
