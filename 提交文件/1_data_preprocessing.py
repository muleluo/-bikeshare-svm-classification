"""
数据预处理脚本
功能：从12个月的Capital Bikeshare数据中进行分层抽样，处理缺失值和异常值，进行特征工程
"""

import pandas as pd
import numpy as np
import zipfile
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BikeShareDataProcessor:
    def __init__(self, data_dir='实验课数据', target_total_samples=60000,
                 sampling_method='proportional'):
        """
        初始化数据处理器

        参数:
            data_dir: 数据文件夹路径
            target_total_samples: 目标总样本数（默认60000）
            sampling_method: 抽样方法
                - 'proportional': 按月数据量比例抽样（推荐）
                - 'equal': 每月抽取相同数量
        """
        self.data_dir = data_dir
        self.target_total_samples = target_total_samples
        self.sampling_method = sampling_method
        self.months = [f'202501', f'202502', f'202503', f'202504',
                       f'202505', f'202506', f'202507', f'202508',
                       f'202509', f'202510', f'202511', f'202512']

    def load_and_sample_data(self):
        """
        从12个月的数据中进行科学的分层抽样

        抽样策略：
        1. 第一遍扫描：统计每月数据量和类别分布
        2. 计算抽样比例：按月数据量比例分配样本数
        3. 第二遍抽样：在每月内部按类别进行分层抽样，保证类别平衡
        """
        print("开始数据抽样...")
        print(f"目标总样本数: {self.target_total_samples}")
        print(f"抽样方法: {self.sampling_method}")

        # 第一步：扫描所有月份，统计数据量和类别分布
        print("\n第一步：扫描所有月份数据...")
        monthly_info = []

        for month in self.months:
            zip_file = f"{self.data_dir}/{month}-capitalbikeshare-tripdata.zip"

            try:
                with zipfile.ZipFile(zip_file, 'r') as z:
                    csv_filename = f"{month}-capitalbikeshare-tripdata.csv"
                    with z.open(csv_filename) as f:
                        df_month = pd.read_csv(f)

                total = len(df_month)
                member_count = len(df_month[df_month['member_casual'] == 'member'])
                casual_count = len(df_month[df_month['member_casual'] == 'casual'])

                monthly_info.append({
                    'month': month,
                    'total': total,
                    'member': member_count,
                    'casual': casual_count,
                    'member_ratio': member_count / total if total > 0 else 0
                })

                print(f"  {month}: 总计{total:>7}条 (会员{member_count:>6}, 散户{casual_count:>6}, 会员比例{member_count/total:.2%})")

            except Exception as e:
                print(f"  ⚠️  读取{month}数据出错: {e}")
                continue

        # 计算总数据量
        total_records = sum(info['total'] for info in monthly_info)
        print(f"\n原始数据总计: {total_records:,} 条")

        # 第二步：计算每月应抽取的样本数
        print("\n第二步：计算每月抽样配额...")

        if self.sampling_method == 'proportional':
            # 按比例抽样：每月抽样数 = 总样本数 × (该月数据量 / 总数据量)
            # 四舍五入到整百
            for info in monthly_info:
                proportion = info['total'] / total_records
                raw_samples = self.target_total_samples * proportion
                # 四舍五入到最近的100
                info['target_samples'] = round(raw_samples / 100) * 100
                print(f"  {info['month']}: 占比{proportion:.2%} → 抽样{info['target_samples']:>5}条")
        else:
            # 均等抽样：每月相同数量（已经是整百）
            samples_per_month = (self.target_total_samples // len(monthly_info) // 100) * 100
            for info in monthly_info:
                info['target_samples'] = samples_per_month
                print(f"  {info['month']}: 抽样{info['target_samples']:>5}条")

        # 调整总和，确保接近目标值
        total_target = sum(info['target_samples'] for info in monthly_info)
        print(f"\n调整后总目标: {total_target:,} 条 (目标: {self.target_total_samples:,})")

        # 第三步：在每月内部进行分层抽样，确保类别平衡
        print("\n第三步：执行分层抽样...")
        all_data = []
        monthly_stats = []

        for info in monthly_info:
            month = info['month']
            target_samples = info['target_samples']

            if target_samples == 0:
                continue

            zip_file = f"{self.data_dir}/{month}-capitalbikeshare-tripdata.zip"

            try:
                with zipfile.ZipFile(zip_file, 'r') as z:
                    csv_filename = f"{month}-capitalbikeshare-tripdata.csv"
                    with z.open(csv_filename) as f:
                        df_month = pd.read_csv(f)

                # 在该月内部进行分层抽样：会员和散户各抽50%
                member_target = target_samples // 2
                casual_target = target_samples - member_target

                sampled_data = []

                # 抽样会员
                member_data = df_month[df_month['member_casual'] == 'member']
                if len(member_data) > 0:
                    n_member = min(len(member_data), member_target)
                    sampled_member = member_data.sample(n=n_member, random_state=42)
                    sampled_data.append(sampled_member)

                # 抽样散户
                casual_data = df_month[df_month['member_casual'] == 'casual']
                if len(casual_data) > 0:
                    n_casual = min(len(casual_data), casual_target)
                    sampled_casual = casual_data.sample(n=n_casual, random_state=42)
                    sampled_data.append(sampled_casual)

                df_sampled = pd.concat(sampled_data, ignore_index=True)
                df_sampled['month'] = month

                actual_member = len(df_sampled[df_sampled['member_casual'] == 'member'])
                actual_casual = len(df_sampled[df_sampled['member_casual'] == 'casual'])

                print(f"  {month}: 目标{target_samples:>5}条 → 实际{len(df_sampled):>5}条 (会员{actual_member}, 散户{actual_casual})")

                monthly_stats.append({
                    'month': month,
                    'original_total': info['total'],
                    'original_member': info['member'],
                    'original_casual': info['casual'],
                    'target_samples': target_samples,
                    'sampled_total': len(df_sampled),
                    'sampled_member': actual_member,
                    'sampled_casual': actual_casual,
                    'sampling_rate': len(df_sampled) / info['total']
                })

                all_data.append(df_sampled)

            except Exception as e:
                print(f"  ⚠️  处理{month}数据时出错: {e}")
                continue

        # 合并所有月份的数据
        df_combined = pd.concat(all_data, ignore_index=True)

        # 保存抽样统计信息
        stats_df = pd.DataFrame(monthly_stats)
        stats_df.to_csv('sampling_statistics.csv', index=False)

        # 打印最终统计
        print("\n" + "="*70)
        print("抽样完成统计:")
        print("="*70)
        total_sampled = len(df_combined)
        total_member = len(df_combined[df_combined['member_casual'] == 'member'])
        total_casual = len(df_combined[df_combined['member_casual'] == 'casual'])

        print(f"总样本数: {total_sampled:,} 条")
        print(f"  会员(member): {total_member:,} 条 ({total_member/total_sampled:.2%})")
        print(f"  散户(casual): {total_casual:,} 条 ({total_casual/total_sampled:.2%})")
        print(f"类别平衡度: {min(total_member, total_casual) / max(total_member, total_casual):.2%}")
        print(f"总体抽样率: {total_sampled / total_records:.2%}")
        print(f"\n抽样统计已保存: sampling_statistics.csv")

        return df_combined

    def extract_features(self, df):
        """
        特征工程：从原始数据中提取有用的特征
        """
        print("\n开始特征工程...")

        df = df.copy()

        # 1. 时间特征提取
        df['started_at'] = pd.to_datetime(df['started_at'])
        df['ended_at'] = pd.to_datetime(df['ended_at'])

        # 提取时间特征
        df['hour'] = df['started_at'].dt.hour
        df['day_of_week'] = df['started_at'].dt.dayofweek  # 0=周一, 6=周日
        df['day'] = df['started_at'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # 时段分类（早高峰、晚高峰、工作时间、其他）
        df['time_period'] = 'other'
        df.loc[(df['hour'] >= 7) & (df['hour'] < 9), 'time_period'] = 'morning_rush'
        df.loc[(df['hour'] >= 17) & (df['hour'] < 19), 'time_period'] = 'evening_rush'
        df.loc[(df['hour'] >= 9) & (df['hour'] < 17), 'time_period'] = 'work_hours'

        # 2. 行程时长（分钟）
        df['trip_duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

        # 3. 行程距离（使用Haversine公式计算）
        df['trip_distance'] = self.haversine_distance(
            df['start_lat'], df['start_lng'],
            df['end_lat'], df['end_lng']
        )

        # 4. 车辆类型编码
        df['is_electric'] = (df['rideable_type'] == 'electric_bike').astype(int)

        # 5. 是否往返同一站点
        df['is_round_trip'] = (df['start_station_id'] == df['end_station_id']).astype(int)

        # 6. 月份编码
        df['month_num'] = df['month'].str[4:6].astype(int)

        print(f"特征工程完成，新增特征: trip_duration, trip_distance, hour, day_of_week, is_weekend, time_period, is_electric, is_round_trip")

        return df

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        计算两点间的距离（公里）
        """
        R = 6371  # 地球半径（公里）

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def clean_data(self, df):
        """
        数据清洗：处理缺失值和异常值
        """
        print("\n开始数据清洗...")

        initial_count = len(df)
        print(f"初始数据量: {initial_count}")

        # 1. 检查缺失值
        missing_info = df.isnull().sum()
        print("\n缺失值统计:")
        print(missing_info[missing_info > 0])

        # 2. 处理缺失值
        # 删除关键字段缺失的记录
        df = df.dropna(subset=['started_at', 'ended_at', 'start_lat', 'start_lng',
                                'end_lat', 'end_lng', 'member_casual'])

        # 站点名称和ID缺失用"Unknown"填充
        df['start_station_name'] = df['start_station_name'].fillna('Unknown')
        df['end_station_name'] = df['end_station_name'].fillna('Unknown')
        df['start_station_id'] = df['start_station_id'].fillna('Unknown')
        df['end_station_id'] = df['end_station_id'].fillna('Unknown')

        print(f"删除关键字段缺失后: {len(df)} 条 (减少 {initial_count - len(df)} 条)")

        # 3. 处理异常值
        # 删除行程时长异常的记录（<1分钟或>480分钟即8小时）
        before = len(df)
        df = df[(df['trip_duration'] >= 1) & (df['trip_duration'] <= 480)]
        print(f"删除行程时长异常值: {len(df)} 条 (减少 {before - len(df)} 条)")

        # 删除行程距离异常的记录（>50公里）
        before = len(df)
        df = df[df['trip_distance'] <= 50]
        print(f"删除行程距离异常值: {len(df)} 条 (减少 {before - len(df)} 条)")

        # 删除速度异常的记录（>40 km/h）
        df['speed'] = df['trip_distance'] / (df['trip_duration'] / 60)  # km/h
        before = len(df)
        df = df[df['speed'] <= 40]
        print(f"删除速度异常值: {len(df)} 条 (减少 {before - len(df)} 条)")

        print(f"\n数据清洗完成，最终数据量: {len(df)} 条")
        print(f"数据保留率: {len(df)/initial_count*100:.2f}%")

        return df

    def prepare_model_data(self, df):
        """
        准备用于建模的数据
        """
        print("\n准备建模数据...")

        # 选择用于建模的特征
        feature_columns = [
            'trip_duration', 'trip_distance', 'speed',
            'hour', 'day_of_week', 'is_weekend', 'month_num',
            'is_electric', 'is_round_trip',
            'start_lat', 'start_lng', 'end_lat', 'end_lng'
        ]

        # 添加time_period的one-hot编码
        time_period_dummies = pd.get_dummies(df['time_period'], prefix='time_period')

        # 合并特征
        X = pd.concat([df[feature_columns], time_period_dummies], axis=1)

        # 目标变量（1=member, 0=casual）
        y = (df['member_casual'] == 'member').astype(int)

        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"目标变量分布: member={y.sum()}, casual={len(y)-y.sum()}")

        return X, y, df

    def process(self):
        """
        执行完整的数据处理流程
        """
        print("="*60)
        print("开始数据预处理流程")
        print("="*60)

        # 1. 加载和抽样
        df = self.load_and_sample_data()

        # 2. 特征工程
        df = self.extract_features(df)

        # 3. 数据清洗
        df = self.clean_data(df)

        # 4. 准备建模数据
        X, y, df_processed = self.prepare_model_data(df)

        # 5. 保存处理后的数据
        df_processed.to_csv('processed_data.csv', index=False)
        X.to_csv('features.csv', index=False)
        y.to_csv('labels.csv', index=False)

        print("\n数据预处理完成！")
        print(f"处理后的完整数据已保存: processed_data.csv")
        print(f"特征矩阵已保存: features.csv")
        print(f"标签已保存: labels.csv")

        return X, y, df_processed


if __name__ == "__main__":
    # 创建数据处理器
    # sampling_method可选: 'proportional'(按比例抽样，推荐) 或 'equal'(均等抽样)
    processor = BikeShareDataProcessor(
        target_total_samples=60000,
        sampling_method='proportional'
    )

    # 执行数据处理
    X, y, df_processed = processor.process()

    print("\n" + "="*60)
    print("数据预处理流程全部完成！")
    print("="*60)
