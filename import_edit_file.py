import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def standardize_numeric_columns(df, exclude_columns=None):
    """
    数値列を標準化（Z-score正規化）する関数
    平均0、標準偏差1に変換
    
    Parameters:
    df: DataFrame
    exclude_columns: 標準化から除外する列名のリスト
    
    Returns:
    DataFrame: 標準化されたデータフレーム
    """
    if exclude_columns is None:
        exclude_columns = []
    
    df_standardized = df.copy()
    
    # 数値列を特定
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 除外列を除いた数値列のみを標準化
    columns_to_standardize = [col for col in numeric_columns if col not in exclude_columns]
    
    for column in columns_to_standardize:
        mean = df[column].mean()
        std = df[column].std()
        
        # 標準偏差が0の場合は標準化しない（除算エラー回避）
        if std != 0:
            df_standardized[column] = (df[column] - mean) / std
            print(f"標準化完了: {column} (平均: {mean:.3f}, 標準偏差: {std:.3f})")
        else:
            print(f"標準化スキップ: {column} (標準偏差が0)")
    
    return df_standardized

def edit(df):  # データ加工
    # 数値列を適切な型に変換
    numeric_columns = ['取引月', '曜日', '土日', '祝日', '平均気温', '合計降水量', 
                      '平均風速', '季節', '来客組数', '売上']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 取引日をdatetime型に変換
    df['取引日'] = pd.to_datetime(df['取引日'])
    
    # 土日と祝日の論理和を求めて休日カラムを作成
    # 土日または祝日のいずれかが1の場合、休日=1とする
    df['休日'] = ((df['土日'] == 1) | (df['祝日'] == 1)).astype(int)
    
    # 年始からの経過日数を計算（1月1日を1日目とする）
    df['day_of_year'] = df['取引日'].dt.dayofyear
    
    # 1年の周期性をsin/cos変換で表現
    # 365日で1周期として角度を計算
    angle = 2 * np.pi * df['day_of_year'] / 365
    
    # sin/cos変換して新しい列を作成
    df['取引日_sin'] = np.sin(angle)
    df['取引日_cos'] = np.cos(angle)
    
    # 曜日のsin/cos変換（1週間=7日の周期性）
    # 曜日は1～7の値を持つので、7で1周期として角度を計算
    dow_angle = 2 * np.pi * (df['曜日'] - 1) / 7  # 曜日を0～6に正規化してから角度計算
    df['曜日_sin'] = np.sin(dow_angle)
    df['曜日_cos'] = np.cos(dow_angle)
    
    # 季節のsin/cos変換（1年=4季節の周期性）
    # 季節は1～4の値を持つので、4で1周期として角度を計算
    season_angle = 2 * np.pi * (df['季節'] - 1) / 4  # 季節を0～3に正規化してから角度計算
    df['季節_sin'] = np.sin(season_angle)
    df['季節_cos'] = np.cos(season_angle)
    
    # 中間的な列（day_of_year）は削除
    df = df.drop('day_of_year', axis=1)
    
    # データ標準化を実行
    # 来客組数と売上は目的変数として除外、また既にsin/cos変換した列も除外
    exclude_cols = ['来客組数', '売上', '曜日', '休日', '季節', '取引日_sin', '取引日_cos', '曜日_sin', '曜日_cos', '季節_sin', '季節_cos']
    df = standardize_numeric_columns(df, exclude_columns=exclude_cols)
    
    return df

def analyze_data(df):
    """
    データを8:2に分割して、来客組数と売上を目的変数とした分析を行う
    
    Parameters:
    df: 前処理済みのDataFrame
    
    Returns:
    dict: 分析結果
    """
    print("\n=== データ分析開始 ===")
    
    # 目的変数の定義
    target_cols = ['来客組数', '売上']
    
    # 説明変数の準備（取引日、目的変数、非数値列を除外）
    exclude_features = ['取引日'] + target_cols
    feature_cols = [col for col in df.columns if col not in exclude_features and df[col].dtype in ['int64', 'float64']]
    
    print(f"説明変数: {feature_cols}")
    print(f"目的変数: {target_cols}")
    
    # データセットの準備
    X = df[feature_cols]
    
    results = {}
    
    # 各目的変数に対して分析を実行
    for target_col in target_cols:
        print(f"\n--- {target_col}の分析 ---")
        y = df[target_col]
        
        # 8:2でデータ分割（ランダムシード固定で再現性確保）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"学習データ数: {len(X_train)}")
        print(f"評価データ数: {len(X_test)}")
        
        # Random Forestで学習
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 予測
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 評価指標の計算
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"学習データ RMSE: {train_rmse:.2f}")
        print(f"評価データ RMSE: {test_rmse:.2f}")
        print(f"学習データ R²: {train_r2:.3f}")
        print(f"評価データ R²: {test_r2:.3f}")
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n特徴量重要度 Top 5:")
        print(feature_importance.head())
        
        # 結果を保存
        results[target_col] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    return results

def save_analysis_results(results, output_dir="./csv/"):
    """
    分析結果をCSVファイルに保存
    """
    print("\n=== 分析結果保存 ===")
    
    for target_col, result in results.items():
        # 特徴量重要度を保存
        importance_file = f"{output_dir}feature_importance_{target_col}.csv"
        result['feature_importance'].to_csv(importance_file, index=False, encoding='utf-8-sig')
        print(f"特徴量重要度保存: {importance_file}")
        
        # 予測結果を保存
        prediction_df = pd.DataFrame({
            'actual_train': result['y_train'],
            'predicted_train': result['y_train_pred']
        })
        pred_file = f"{output_dir}predictions_{target_col}_train.csv"
        prediction_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
        
        prediction_df_test = pd.DataFrame({
            'actual_test': result['y_test'],
            'predicted_test': result['y_test_pred']
        })
        pred_test_file = f"{output_dir}predictions_{target_col}_test.csv"
        prediction_df_test.to_csv(pred_test_file, index=False, encoding='utf-8-sig')
        
        print(f"予測結果保存: {pred_file}, {pred_test_file}")


