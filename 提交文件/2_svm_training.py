"""
SVM分类模型训练脚本
功能：使用支持向量机(SVM)算法对用户类型进行分类预测，并评估模型性能
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import joblib
import json
import time

class SVMClassifier:
    def __init__(self, test_size=0.2, random_state=42):
        """
        初始化SVM分类器

        参数:
            test_size: 测试集比例，默认0.2（即80%训练，20%测试）
            random_state: 随机种子，保证可复现性
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model = None
        self.results = {}

    def load_data(self):
        """
        加载预处理后的数据
        """
        print("加载数据...")
        X = pd.read_csv('features.csv')
        y = pd.read_csv('labels.csv').values.ravel()

        print(f"特征矩阵形状: {X.shape}")
        print(f"标签形状: {y.shape}")
        print(f"类别分布: member={y.sum()}, casual={len(y)-y.sum()}")

        return X, y

    def split_data(self, X, y):
        """
        划分训练集和测试集（80:20）
        """
        print(f"\n划分数据集（训练集{int((1-self.test_size)*100)}% : 测试集{int(self.test_size*100)}%）...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"训练集大小: {X_train.shape[0]} 样本")
        print(f"测试集大小: {X_test.shape[0]} 样本")
        print(f"训练集类别分布: member={y_train.sum()}, casual={len(y_train)-y_train.sum()}")
        print(f"测试集类别分布: member={y_test.sum()}, casual={len(y_test)-y_test.sum()}")

        return X_train, X_test, y_train, y_test

    def normalize_data(self, X_train, X_test):
        """
        数据标准化（Z-score标准化）
        """
        print("\n数据标准化...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("标准化完成")

        return X_train_scaled, X_test_scaled

    def train_baseline_model(self, X_train, y_train, X_test, y_test):
        """
        训练基线SVM模型（使用默认参数）
        """
        print("\n" + "="*60)
        print("训练基线SVM模型（默认参数）")
        print("="*60)

        start_time = time.time()

        # 使用RBF核的SVM
        baseline_model = SVC(kernel='rbf', random_state=self.random_state, probability=True)
        baseline_model.fit(X_train, y_train)

        train_time = time.time() - start_time

        # 预测
        y_train_pred = baseline_model.predict(X_train)
        y_test_pred = baseline_model.predict(X_test)

        # 评估
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"训练时间: {train_time:.2f} 秒")
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        self.results['baseline'] = {
            'model': 'SVM (RBF kernel, default params)',
            'train_time': train_time,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'params': baseline_model.get_params()
        }

        return baseline_model

    def tune_hyperparameters(self, X_train, y_train):
        """
        使用网格搜索调优超参数
        """
        print("\n" + "="*60)
        print("超参数调优（网格搜索 + 5折交叉验证）")
        print("="*60)

        # 定义参数网格（简化版本，保证质量的同时提高速度）
        param_grid = {
            'C': [0.1, 1, 10],  # 正则化参数（减少到3个）
            'gamma': ['scale', 0.01],  # RBF核参数（减少到2个）
            'kernel': ['rbf', 'linear']  # 核函数类型
        }

        print("参数网格:")
        print(json.dumps(param_grid, indent=2))
        print(f"总计参数组合: {3 * 2 * 2} × 5折交叉验证 = {3 * 2 * 2 * 5}次训练")

        start_time = time.time()

        # 网格搜索
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)

        tune_time = time.time() - start_time

        print(f"\n超参数调优完成，耗时: {tune_time:.2f} 秒")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

        self.best_model = grid_search.best_estimator_

        self.results['tuning'] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'tune_time': tune_time
        }

        return self.best_model

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """
        全面评估模型性能
        """
        print("\n" + "="*60)
        print("模型评估")
        print("="*60)

        # 训练集预测
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]

        # 测试集预测
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # 计算各项指标
        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred),
                'recall': recall_score(y_train, y_train_pred),
                'f1_score': f1_score(y_train, y_train_pred),
                'auc': roc_auc_score(y_train, y_train_proba)
            },
            'test': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1_score': f1_score(y_test, y_test_pred),
                'auc': roc_auc_score(y_test, y_test_proba)
            }
        }

        # 混淆矩阵
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        # 打印结果
        print("\n训练集性能:")
        print(f"  准确率 (Accuracy):  {metrics['train']['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['train']['precision']:.4f}")
        print(f"  召回率 (Recall):    {metrics['train']['recall']:.4f}")
        print(f"  F1分数 (F1-Score):  {metrics['train']['f1_score']:.4f}")
        print(f"  AUC:                {metrics['train']['auc']:.4f}")

        print("\n测试集性能:")
        print(f"  准确率 (Accuracy):  {metrics['test']['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['test']['precision']:.4f}")
        print(f"  召回率 (Recall):    {metrics['test']['recall']:.4f}")
        print(f"  F1分数 (F1-Score):  {metrics['test']['f1_score']:.4f}")
        print(f"  AUC:                {metrics['test']['auc']:.4f}")

        print("\n测试集混淆矩阵:")
        print("                预测")
        print("              Casual  Member")
        print(f"实际 Casual   {cm_test[0, 0]:5d}   {cm_test[0, 1]:5d}")
        print(f"     Member   {cm_test[1, 0]:5d}   {cm_test[1, 1]:5d}")

        print("\n测试集分类报告:")
        print(classification_report(y_test, y_test_pred,
                                   target_names=['Casual', 'Member']))

        # 交叉验证
        print("\n5折交叉验证...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"交叉验证得分: {cv_scores}")
        print(f"平均得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # 保存结果
        self.results['evaluation'] = {
            'metrics': metrics,
            'confusion_matrix_train': cm_train.tolist(),
            'confusion_matrix_test': cm_test.tolist(),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std())
        }

        # 保存预测结果
        self.results['predictions'] = {
            'y_test': y_test.tolist(),
            'y_test_pred': y_test_pred.tolist(),
            'y_test_proba': y_test_proba.tolist()
        }

        return metrics, cm_test

    def save_model(self):
        """
        保存模型和结果
        """
        print("\n保存模型和结果...")

        # 保存模型
        joblib.dump(self.best_model, 'svm_model.pkl')
        print("模型已保存: svm_model.pkl")

        # 保存标准化器
        joblib.dump(self.scaler, 'scaler.pkl')
        print("标准化器已保存: scaler.pkl")

        # 保存结果
        with open('model_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print("结果已保存: model_results.json")

    def run(self):
        """
        执行完整的训练和评估流程
        """
        print("="*60)
        print("SVM分类模型训练流程")
        print("="*60)

        # 1. 加载数据
        X, y = self.load_data()

        # 2. 划分数据集
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # 3. 数据标准化
        X_train_scaled, X_test_scaled = self.normalize_data(X_train, X_test)

        # 4. 训练基线模型
        baseline_model = self.train_baseline_model(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

        # 5. 超参数调优
        best_model = self.tune_hyperparameters(X_train_scaled, y_train)

        # 6. 评估最佳模型
        metrics, cm = self.evaluate_model(
            best_model, X_train_scaled, y_train, X_test_scaled, y_test
        )

        # 7. 保存模型
        self.save_model()

        print("\n" + "="*60)
        print("模型训练和评估流程全部完成！")
        print("="*60)

        return self.best_model, metrics


if __name__ == "__main__":
    # 创建分类器实例
    classifier = SVMClassifier(test_size=0.2, random_state=42)

    # 运行训练和评估
    model, metrics = classifier.run()

    print(f"\n最终测试集准确率: {metrics['test']['accuracy']:.4f}")
