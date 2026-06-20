"""
数据分析和可视化脚本
功能：生成各类图表用于数据分析和结果展示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

class DataVisualizer:
    def __init__(self):
        """
        初始化可视化器
        """
        self.processed_data = None
        self.model_results = None
        self.output_dir = 'figures'

        # 创建输出目录
        import os
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        加载数据
        """
        print("加载数据...")
        self.processed_data = pd.read_csv('processed_data.csv')

        try:
            with open('model_results.json', 'r') as f:
                self.model_results = json.load(f)
            print("模型结果已加载")
        except FileNotFoundError:
            print("警告: model_results.json 未找到，部分图表将无法生成")
            self.model_results = None

    def plot_sampling_statistics(self):
        """
        1. 绘制抽样统计图
        """
        print("\n生成抽样统计图...")

        stats = pd.read_csv('sampling_statistics.csv')

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1) 每月原始数据量
        ax1 = axes[0, 0]
        x = range(len(stats))
        ax1.bar(x, stats['original_total'], alpha=0.7, color='skyblue')
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Number of Records', fontsize=12)
        ax1.set_title('Original Data Volume by Month', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        # 将月份转为字符串再提取
        month_labels = [str(m)[4:6] for m in stats['month']]
        ax1.set_xticklabels(month_labels, rotation=0)
        ax1.grid(axis='y', alpha=0.3)

        # 2) 每月原始数据的类别分布
        ax2 = axes[0, 1]
        width = 0.35
        ax2.bar([i - width/2 for i in x], stats['original_member'], width, label='Member', alpha=0.8)
        ax2.bar([i + width/2 for i in x], stats['original_casual'], width, label='Casual', alpha=0.8)
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Number of Records', fontsize=12)
        ax2.set_title('Original Data Distribution by User Type', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(month_labels, rotation=0)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3) 抽样后数据量对比
        ax3 = axes[1, 0]
        ax3.bar(x, stats['sampled_total'], alpha=0.7, color='coral')
        ax3.set_xlabel('Month', fontsize=12)
        ax3.set_ylabel('Number of Records', fontsize=12)
        ax3.set_title('Sampled Data Volume by Month', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(month_labels, rotation=0)
        ax3.grid(axis='y', alpha=0.3)

        # 4) 抽样后的类别平衡性
        ax4 = axes[1, 1]
        ax4.bar([i - width/2 for i in x], stats['sampled_member'], width, label='Member', alpha=0.8)
        ax4.bar([i + width/2 for i in x], stats['sampled_casual'], width, label='Casual', alpha=0.8)
        ax4.set_xlabel('Month', fontsize=12)
        ax4.set_ylabel('Number of Records', fontsize=12)
        ax4.set_title('Sampled Data Distribution (Balanced)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(month_labels, rotation=0)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_sampling_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {self.output_dir}/01_sampling_statistics.png")

    def plot_temporal_analysis(self):
        """
        2. 绘制时间维度分析图
        """
        print("\n生成时间维度分析图...")

        df = self.processed_data

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1) 小时分布
        ax1 = axes[0, 0]
        hour_member = df[df['member_casual']=='member']['hour'].value_counts().sort_index()
        hour_casual = df[df['member_casual']=='casual']['hour'].value_counts().sort_index()
        ax1.plot(hour_member.index, hour_member.values, marker='o', label='Member', linewidth=2)
        ax1.plot(hour_casual.index, hour_casual.values, marker='s', label='Casual', linewidth=2)
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Number of Trips', fontsize=12)
        ax1.set_title('Trip Distribution by Hour', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))

        # 2) 星期分布
        ax2 = axes[0, 1]
        weekday_data = df.groupby(['day_of_week', 'member_casual']).size().unstack(fill_value=0)
        weekday_data.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Number of Trips', fontsize=12)
        ax2.set_title('Trip Distribution by Day of Week', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
        ax2.legend(['Casual', 'Member'])
        ax2.grid(axis='y', alpha=0.3)

        # 3) 时段分布
        ax3 = axes[1, 0]
        time_period_data = df.groupby(['time_period', 'member_casual']).size().unstack(fill_value=0)
        time_period_data = time_period_data.reindex(['morning_rush', 'work_hours', 'evening_rush', 'other'])
        time_period_data.plot(kind='bar', ax=ax3, alpha=0.8, width=0.7)
        ax3.set_xlabel('Time Period', fontsize=12)
        ax3.set_ylabel('Number of Trips', fontsize=12)
        ax3.set_title('Trip Distribution by Time Period', fontsize=14, fontweight='bold')
        ax3.set_xticklabels(['Morning\nRush', 'Work\nHours', 'Evening\nRush', 'Other'], rotation=0)
        ax3.legend(['Casual', 'Member'])
        ax3.grid(axis='y', alpha=0.3)

        # 4) 月份分布
        ax4 = axes[1, 1]
        month_data = df.groupby(['month_num', 'member_casual']).size().unstack(fill_value=0)
        month_data.plot(kind='line', ax=ax4, marker='o', linewidth=2)
        ax4.set_xlabel('Month', fontsize=12)
        ax4.set_ylabel('Number of Trips', fontsize=12)
        ax4.set_title('Trip Distribution by Month', fontsize=14, fontweight='bold')
        ax4.legend(['Casual', 'Member'])
        ax4.grid(alpha=0.3)
        ax4.set_xticks(range(1, 13))

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {self.output_dir}/02_temporal_analysis.png")

    def plot_spatial_analysis(self):
        """
        3. 绘制空间维度分析图
        """
        print("\n生成空间维度分析图...")

        df = self.processed_data

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1) 起点位置分布
        ax1 = axes[0]
        member_data = df[df['member_casual']=='member']
        casual_data = df[df['member_casual']=='casual']

        ax1.scatter(member_data['start_lng'], member_data['start_lat'],
                   alpha=0.3, s=1, label='Member', c='blue')
        ax1.scatter(casual_data['start_lng'], casual_data['start_lat'],
                   alpha=0.3, s=1, label='Casual', c='orange')
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.set_title('Trip Start Locations by User Type', fontsize=14, fontweight='bold')
        ax1.legend(markerscale=10)
        ax1.grid(alpha=0.3)

        # 2) 终点位置分布
        ax2 = axes[1]
        ax2.scatter(member_data['end_lng'], member_data['end_lat'],
                   alpha=0.3, s=1, label='Member', c='blue')
        ax2.scatter(casual_data['end_lng'], casual_data['end_lat'],
                   alpha=0.3, s=1, label='Casual', c='orange')
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Latitude', fontsize=12)
        ax2.set_title('Trip End Locations by User Type', fontsize=14, fontweight='bold')
        ax2.legend(markerscale=10)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_spatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {self.output_dir}/03_spatial_analysis.png")

    def plot_trip_characteristics(self):
        """
        4. 绘制行程特征分析图
        """
        print("\n生成行程特征分析图...")

        df = self.processed_data

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1) 行程时长分布
        ax1 = axes[0, 0]
        df[df['member_casual']=='member']['trip_duration'].hist(bins=50, alpha=0.6,
                                                                  label='Member', ax=ax1, color='blue')
        df[df['member_casual']=='casual']['trip_duration'].hist(bins=50, alpha=0.6,
                                                                 label='Casual', ax=ax1, color='orange')
        ax1.set_xlabel('Trip Duration (minutes)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Trip Duration Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.set_xlim(0, 60)

        # 2) 行程距离分布
        ax2 = axes[0, 1]
        df[df['member_casual']=='member']['trip_distance'].hist(bins=50, alpha=0.6,
                                                                  label='Member', ax=ax2, color='blue')
        df[df['member_casual']=='casual']['trip_distance'].hist(bins=50, alpha=0.6,
                                                                 label='Casual', ax=ax2, color='orange')
        ax2.set_xlabel('Trip Distance (km)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Trip Distance Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.set_xlim(0, 10)

        # 3) 速度分布
        ax3 = axes[0, 2]
        df[df['member_casual']=='member']['speed'].hist(bins=50, alpha=0.6,
                                                         label='Member', ax=ax3, color='blue')
        df[df['member_casual']=='casual']['speed'].hist(bins=50, alpha=0.6,
                                                        label='Casual', ax=ax3, color='orange')
        ax3.set_xlabel('Speed (km/h)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Speed Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.set_xlim(0, 25)

        # 4) 车辆类型分布
        ax4 = axes[1, 0]
        bike_type_data = df.groupby(['is_electric', 'member_casual']).size().unstack(fill_value=0)
        bike_type_data.plot(kind='bar', ax=ax4, alpha=0.8)
        ax4.set_xlabel('Bike Type', fontsize=11)
        ax4.set_ylabel('Number of Trips', fontsize=11)
        ax4.set_title('Bike Type Distribution', fontsize=12, fontweight='bold')
        ax4.set_xticklabels(['Classic', 'Electric'], rotation=0)
        ax4.legend(['Casual', 'Member'])
        ax4.grid(axis='y', alpha=0.3)

        # 5) 周末vs工作日
        ax5 = axes[1, 1]
        weekend_data = df.groupby(['is_weekend', 'member_casual']).size().unstack(fill_value=0)
        weekend_data.plot(kind='bar', ax=ax5, alpha=0.8)
        ax5.set_xlabel('Day Type', fontsize=11)
        ax5.set_ylabel('Number of Trips', fontsize=11)
        ax5.set_title('Weekday vs Weekend', fontsize=12, fontweight='bold')
        ax5.set_xticklabels(['Weekday', 'Weekend'], rotation=0)
        ax5.legend(['Casual', 'Member'])
        ax5.grid(axis='y', alpha=0.3)

        # 6) 往返行程
        ax6 = axes[1, 2]
        roundtrip_data = df.groupby(['is_round_trip', 'member_casual']).size().unstack(fill_value=0)
        roundtrip_data.plot(kind='bar', ax=ax6, alpha=0.8)
        ax6.set_xlabel('Trip Type', fontsize=11)
        ax6.set_ylabel('Number of Trips', fontsize=11)
        ax6.set_title('Round Trip Distribution', fontsize=12, fontweight='bold')
        ax6.set_xticklabels(['One-way', 'Round-trip'], rotation=0)
        ax6.legend(['Casual', 'Member'])
        ax6.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_trip_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {self.output_dir}/04_trip_characteristics.png")

    def plot_feature_comparison(self):
        """
        5. 绘制特征对比箱线图
        """
        print("\n生成特征对比箱线图...")

        df = self.processed_data

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        features = ['trip_duration', 'trip_distance', 'speed']
        titles = ['Trip Duration (minutes)', 'Trip Distance (km)', 'Speed (km/h)']

        for ax, feature, title in zip(axes, features, titles):
            data_to_plot = [df[df['member_casual']=='casual'][feature],
                           df[df['member_casual']=='member'][feature]]
            ax.boxplot(data_to_plot, labels=['Casual', 'Member'], patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title} by User Type', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {self.output_dir}/05_feature_comparison.png")

    def plot_confusion_matrix(self):
        """
        6. 绘制混淆矩阵
        """
        if self.model_results is None:
            print("\n跳过混淆矩阵图（模型结果未找到）")
            return

        print("\n生成混淆矩阵图...")

        cm = np.array(self.model_results['evaluation']['confusion_matrix_test'])

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Casual', 'Member'],
                   yticklabels=['Casual', 'Member'],
                   ax=ax, cbar_kws={'label': 'Count'},
                   annot_kws={'size': 16, 'weight': 'bold'})

        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_title('Confusion Matrix (Test Set)', fontsize=15, fontweight='bold', pad=20)

        # 计算准确率等指标
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision = cm[1,1] / (cm[0,1] + cm[1,1])
        recall = cm[1,1] / (cm[1,0] + cm[1,1])

        # 添加统计信息
        stats_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}'
        ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {self.output_dir}/06_confusion_matrix.png")

    def plot_roc_curve(self):
        """
        7. 绘制ROC曲线
        """
        if self.model_results is None:
            print("\n跳过ROC曲线图（模型结果未找到）")
            return

        print("\n生成ROC曲线图...")

        y_test = np.array(self.model_results['predictions']['y_test'])
        y_proba = np.array(self.model_results['predictions']['y_test_proba'])

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=15, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {self.output_dir}/07_roc_curve.png")

    def plot_model_performance(self):
        """
        8. 绘制模型性能对比图
        """
        if self.model_results is None:
            print("\n跳过模型性能对比图（模型结果未找到）")
            return

        print("\n生成模型性能对比图...")

        metrics = self.model_results['evaluation']['metrics']

        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        train_scores = [metrics['train']['accuracy'], metrics['train']['precision'],
                       metrics['train']['recall'], metrics['train']['f1_score'],
                       metrics['train']['auc']]
        test_scores = [metrics['test']['accuracy'], metrics['test']['precision'],
                      metrics['test']['recall'], metrics['test']['f1_score'],
                      metrics['test']['auc']]

        x = np.arange(len(metric_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8, color='coral')

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim([0.7, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {self.output_dir}/08_model_performance.png")

    def run(self):
        """
        执行所有可视化
        """
        print("="*60)
        print("数据分析和可视化流程")
        print("="*60)

        self.load_data()

        self.plot_sampling_statistics()
        self.plot_temporal_analysis()
        self.plot_spatial_analysis()
        self.plot_trip_characteristics()
        self.plot_feature_comparison()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_model_performance()

        print("\n" + "="*60)
        print(f"所有图表已保存到 {self.output_dir}/ 目录")
        print("="*60)


if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.run()
