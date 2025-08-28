import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_predictions():
    """
    予測結果の精度を詳しく評価する
    """
    targets = ['来客組数', '売上']
    
    for target in targets:
        print(f"\n{'='*50}")
        print(f"【{target}の予測精度評価】")
        print(f"{'='*50}")
        
        # 学習データの予測結果
        train_file = f"./csv/predictions_{target}_train.csv"
        train_df = pd.read_csv(train_file)
        
        # 評価データの予測結果
        test_file = f"./csv/predictions_{target}_test.csv"
        test_df = pd.read_csv(test_file)
        
        # 学習データの評価
        print(f"\n--- 学習データ ({len(train_df)}件) ---")
        train_actual = train_df['actual_train']
        train_pred = train_df['predicted_train']
        
        train_mae = mean_absolute_error(train_actual, train_pred)
        train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
        train_r2 = r2_score(train_actual, train_pred)
        train_mape = np.mean(np.abs((train_actual - train_pred) / train_actual)) * 100
        
        print(f"MAE (平均絶対誤差): {train_mae:.2f}")
        print(f"RMSE (平方根平均二乗誤差): {train_rmse:.2f}")
        print(f"R² (決定係数): {train_r2:.3f}")
        print(f"MAPE (平均絶対パーセント誤差): {train_mape:.2f}%")
        
        # 評価データの評価
        print(f"\n--- 評価データ ({len(test_df)}件) ---")
        test_actual = test_df['actual_test']
        test_pred = test_df['predicted_test']
        
        test_mae = mean_absolute_error(test_actual, test_pred)
        test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        test_r2 = r2_score(test_actual, test_pred)
        test_mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100
        
        print(f"MAE (平均絶対誤差): {test_mae:.2f}")
        print(f"RMSE (平方根平均二乗誤差): {test_rmse:.2f}")
        print(f"R² (決定係数): {test_r2:.3f}")
        print(f"MAPE (平均絶対パーセント誤差): {test_mape:.2f}%")
        
        # 過学習の確認
        print(f"\n--- 過学習チェック ---")
        print(f"R²差 (学習 - 評価): {train_r2 - test_r2:.3f}")
        if train_r2 - test_r2 > 0.1:
            print("⚠️ 過学習の可能性があります")
        elif train_r2 - test_r2 > 0.05:
            print("⚡ 軽微な過学習の兆候があります")
        else:
            print("✅ 過学習は見られません")
        
        # 統計サマリー
        print(f"\n--- 統計サマリー ---")
        print(f"実際の値の範囲: {test_actual.min():.0f} ～ {test_actual.max():.0f}")
        print(f"予測値の範囲: {test_pred.min():.1f} ～ {test_pred.max():.1f}")
        print(f"実際の値の平均: {test_actual.mean():.1f}")
        print(f"予測値の平均: {test_pred.mean():.1f}")
        
        # 特徴量重要度の表示
        importance_file = f"./csv/feature_importance_{target}.csv"
        importance_df = pd.read_csv(importance_file)
        
        print(f"\n--- 重要な特徴量 Top 5 ---")
        for i, row in importance_df.head(5).iterrows():
            print(f"{i+1}. {row['feature']}: {row['importance']:.3f} ({row['importance']*100:.1f}%)")

def create_accuracy_summary():
    """
    精度サマリーをCSVファイルに保存
    """
    targets = ['来客組数', '売上']
    results = []
    
    for target in targets:
        # 評価データの予測結果を読み込み
        test_file = f"./csv/predictions_{target}_test.csv"
        test_df = pd.read_csv(test_file)
        
        test_actual = test_df['actual_test']
        test_pred = test_df['predicted_test']
        
        # 精度指標を計算
        mae = mean_absolute_error(test_actual, test_pred)
        rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        r2 = r2_score(test_actual, test_pred)
        mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100
        
        results.append({
            '目的変数': target,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE(%)': mape,
            'データ数': len(test_df),
            '実際値_平均': test_actual.mean(),
            '実際値_最小': test_actual.min(),
            '実際値_最大': test_actual.max()
        })
    
    # 結果をDataFrameに変換して保存
    summary_df = pd.DataFrame(results)
    summary_df.to_csv('./csv/accuracy_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n精度サマリーを保存しました: ./csv/accuracy_summary.csv")
    
    return summary_df

if __name__ == "__main__":
    print("予測精度の詳細評価を開始します...")
    
    # 詳細評価の実行
    evaluate_predictions()
    
    # サマリーの作成
    summary_df = create_accuracy_summary()
    
    print(f"\n{'='*50}")
    print("【精度サマリー】")
    print(f"{'='*50}")
    print(summary_df.round(3).to_string(index=False))
