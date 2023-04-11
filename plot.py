import seaborn as sns
import matplotlib.pyplot as plt

def plot_map(dataframe,max_instances_to_plot,hue_metric="original_correctness",main_metric="loss_diff_mean",other_metric="loss_diff_std",do_round=True,show_hist=True,model="BERT",dataset="SST2"):
    # 设置seaborn的样式，字体大小，字体类型和上下文
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')
    # 从数据框中随机抽取一部分数据，以便绘制的散点图不会太拥挤
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # normalize
    # 对数据框中的original_correctness列进行归一化处理，得到original_corr_frac列
    # 将original_corr_frac列的值转换为字符串，并保留一位小数，赋值给original_correctness列
    #dataframe = dataframe.assign(original_corr_frac = lambda d: d.original_correctness / d.original_correctness.max())
    #dataframe['original_correctness'] = [f"{x:.1f}" for x in dataframe['original_corr_frac']]

    # main_metric = 'loss_diff_mean'
    # other_metric = 'loss_diff_std'
    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues <= 11 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])
    #
    # colors = [
    #     "#ea7a66",
    #     "#ed5f4b",
    #     "#f14924",
    #     "#d13123",
    #     "#ea514c",
    #     "#983025",
    #     "#2b66c0",
    #     "#4855b5",
    #     "#2e3178",
    #     "#92b36f",
    #     "#59b351"
    # ]
    # colors.reverse()
    # pal = sns.color_palette(
    #     colors
    # )

    pal = sns.diverging_palette(260, 20, n=num_hues, sep=20, center="dark")
    # pal = sns.diverging_palette(15,320, n=num_hues, sep=20, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    # else:
    #     plot.legend(fancybox=True, shadow=True, ncol=1)
    if show_hist:
        # plot.set_title("{}-{} Robust Data Map".format(dataset,model))

        # histograms
        # ax1 =
        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        # plott0 = dataframe.hist(column=['Perturbed Loss Mean'], ax=ax1, color='#622a87')
        plott0 = dataframe.hist(column=['Sensitivity'], ax=ax1, color='#816e9c')
        plott0[0].set_title('')
        plott0[0].set_xlabel('Sensitivity')
        plott0[0].set_ylabel('Density')

        # plott1 = dataframe.hist(column=['Perturbed Loss Std'], ax=ax2, color='teal')
        plott1 = dataframe.hist(column=['Variability'], ax=ax2, color='#6989b9')
        plott1[0].set_title('')
        plott1[0].set_xlabel('Variability')
        plott1[0].set_ylabel('Density')

        # plott2 = dataframe.hist(column=['Flip Rate'], ax=ax3, color='#86bf91')
        plott2 = dataframe.hist(column=['Flip Rate'], ax=ax3, color='#a9c1a3')
        plott2[0].set_title('')
        plott2[0].set_xlabel('Flip Rate')
        plott2[0].set_ylabel('Density')

        # plot2 = sns.countplot(x="Flip Rate", data=dataframe, ax=ax3, color='#86bf91')
        # # ax3.xaxis.grid(True)  # Show the vertical gridlines
        #
        # plot2.set_title('')
        # plot2.set_xlabel('Flip Rate')
        # plot2.set_ylabel('Density')

    plt.savefig("{}.pdf".format(dataset),pad_inches=0)
    plt.show()

    print()
